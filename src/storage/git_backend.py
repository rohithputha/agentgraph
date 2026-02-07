import os
import subprocess
from pathlib import Path
from typing import List, Optional

class GitBackend:
    """
    Uses git plumbing to snapshot project files into an isolated bare repo
    at .agentgit/snapshots.git.  Snapshots are chained as commits so git
    stores only deltas between them.
    """

    def __init__(self, project_dir: Path, git_dir: Path):
        self.project_dir = project_dir
        self.git_dir = git_dir
        if not (git_dir/ "HEAD").exists():
            self._init_bare_repo()
    
    def _init_bare_repo(self):
        subprocess.run(
            ["git", "init", "--bare", str(self.git_dir)],
            check=True,
            capture_output=True,
        )

    def _create_blob(self, file_path: Path) -> str:
        return self._run(["hash-object", "-w", str(file_path)]).stdout.strip()

    def _run(
        self,
        args: List[str],
        stdin: Optional[str] = None,
        env: Optional[dict] = None,
    ) -> subprocess.CompletedProcess:
        """Run a git plumbing command against the bare snapshot repo."""
        return subprocess.run(
            ["git", "--git-dir", str(self.git_dir)] + args,
            input=stdin,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

    def _write_tree(self, tree_dict: dict):
        """Recursively build git tree objects from nested dict structure."""
        lines: List[str] = []
        for name, value in sorted(tree_dict.items()):  # Sort for consistent tree hashing
            if isinstance(value, str):
                # It's a blob SHA
                lines.append(f"100644 blob {value}\t{name}")
            else:
                # It's a subtree
                sub_sha = self._write_tree(value)
                lines.append(f"040000 tree {sub_sha}\t{name}")
        
        tree_input = "\n".join(lines) + "\n" if lines else ""
        result = self._run(["mktree"], stdin=tree_input)
        return result.stdout.strip()


    def _build_tree(self, blob_shas: dict[str, str]) -> str:
        """Build a git tree from {relative_path: blob_sha}."""
        # Nest into a dict:  {"dir": {"sub": {"file.py": "<sha>"}}}
        root: dict = {}
        for path, sha in blob_shas.items():
            parts = Path(path).parts
            node = root
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = sha
        return self._write_tree(root)

    def _get_last_snapshot(self) -> Optional[str]:
        ref = self.git_dir / "refs" / "heads" / "snapshots"
        return ref.read_text().strip() if ref.exists() else None
    
    def _set_last_snapshot(self, commit_sha: str):
        """Record this commit as the latest snapshot so the next one
        chains off it (enabling git delta storage)."""
        self._run(["update-ref", "refs/heads/snapshots", commit_sha])

    def _create_commit(self, tree_sha: str, message: str, parent: Optional[str] = None) -> str:
        cmd = ["commit-tree", tree_sha, "-m", message]
        if parent:
            cmd.extend(["-p", parent])
        # commit-tree needs author/committer identity; set it explicitly
        # so the snapshot repo works regardless of the user's git config.
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "AgentGit",
            "GIT_AUTHOR_EMAIL": "agentgit@snapshot",
            "GIT_COMMITTER_NAME": "AgentGit",
            "GIT_COMMITTER_EMAIL": "agentgit@snapshot",
        }
        return self._run(cmd, env=env).stdout.strip()

    def create_snapshot(self, message: str = "Checkpoint"):
        """Snapshot all tracked project files. Returns the commit SHA."""
        blob_sha: dict[str, str] = {}
        for path in self._get_tracked_files():
            rel = str(path.relative_to(self.project_dir))
            blob_sha[rel] = self._create_blob(path)
        
        tree_sha = self._build_tree(blob_sha)
        
        parent = self._get_last_snapshot()
        commit_sha = self._create_commit(tree_sha, message, parent)
        self._set_last_snapshot(commit_sha)
        return commit_sha
    
    def restore_snapshot(self, commit_sha: str):
        self._run(["read-tree", commit_sha])
        # materialise index → project working directory
        env = {
            **os.environ,
            "GIT_DIR": str(self.git_dir),
            "GIT_WORK_TREE": str(self.project_dir),
        }
        subprocess.run(
            ["git", "checkout-index", "-a", "-f"],
            env=env,
            cwd=self.project_dir,
            check=True,
            capture_output=True,
            text=True,
        )

    def get_snapshot_files(self, commit_sha: str) -> List[str]:
        """Return the list of file paths recorded in a snapshot."""
        result = self._run(["ls-tree", "-r", "--name-only", commit_sha])
        text = result.stdout.strip()
        return text.split("\n") if text else []
    
    def _get_tracked_files(self) -> List[Path]:
        """Get all files in the project directory (excluding .agentgit and common ignores)."""
        tracked = []
        ignore_patterns = {'.agentgit', '.git', '__pycache__', 'node_modules'}
        ignore_suffixes = {'.pyc', '.DS_Store'}
        
        for path in self.project_dir.rglob('*'):
            if path.is_file():
                try:
                    path.relative_to(self.git_dir)
                    continue
                except ValueError:
                    pass
                
                if any(pattern in path.parts for pattern in ignore_patterns):
                    continue
                
                if any(str(path).endswith(suffix) for suffix in ignore_suffixes):
                    continue
                    
                tracked.append(path)
        
        return tracked