import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

from agenttest.models.config import AgentTestConfig, ScenarioConfig, TestConfig


def load_config(path: Optional[str] = None) -> AgentTestConfig:
    config_path = _find_config_file(path)

    if not config_path:
        return AgentTestConfig()

    if not tomllib:
        raise ImportError(
            "TOML support requires Python 3.11+ or 'tomli' package.\n"
            "Install: pip install tomli"
        )

    try:
        with open(config_path, 'rb') as f:
            data = tomllib.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse TOML config at {config_path}: {e}")

    return _parse_config(data, config_path)


def _find_config_file(explicit_path: Optional[str] = None) -> Optional[Path]:
    search_paths = []

    if explicit_path:
        search_paths.append(Path(explicit_path))

    env_path = os.getenv('AGENTTEST_CONFIG')
    if env_path:
        search_paths.append(Path(env_path))

    search_paths.extend([
        Path.cwd() / 'agenttest.toml',
        Path.cwd() / '.agenttest' / 'config.toml',
    ])

    for path in search_paths:
        if path.exists() and path.is_file():
            return path

    if explicit_path:
        raise FileNotFoundError(f"Config file not found: {explicit_path}")

    return None


def _parse_config(data: Dict[str, Any], source_path: Path) -> AgentTestConfig:
    if 'agenttest' not in data:
        raise ValueError(
            f"Config file {source_path} missing [agenttest] section"
        )

    config_data = data['agenttest']

    similarity_threshold = config_data.get('similarity_threshold', 0.85)
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            f"similarity_threshold must be between 0.0 and 1.0, got {similarity_threshold}"
        )

    default_replay_mode = config_data.get('default_replay_mode', 'full')
    valid_modes = ['full', 'selective', 'locked']
    if default_replay_mode not in valid_modes:
        raise ValueError(
            f"default_replay_mode must be one of {valid_modes}, got '{default_replay_mode}'"
        )

    ignore_fields = config_data.get('ignore_fields', [])
    if not isinstance(ignore_fields, list):
        raise ValueError(
            f"ignore_fields must be a list, got {type(ignore_fields).__name__}"
        )

    tests = []
    for t in data.get('tests', []):
        if 'name' not in t:
            raise ValueError("Each [[tests]] entry must have a 'name' field")
        tier = t.get('tier', 'always')
        if tier not in ('always', 'local', 'ci-only'):
            raise ValueError(f"tier must be 'always', 'local', or 'ci-only', got '{tier}'")
        tests.append(TestConfig(
            name=t['name'],
            tier=tier,
            mode=t.get('mode'),
        ))

    scenarios = []
    for s in data.get("scenarios", []):
        if "name" not in s:
            raise ValueError("Each [[scenarios]] entry must have a 'name' field")
        if "entrypoint" not in s:
            raise ValueError("Each [[scenarios]] entry must have an 'entrypoint' field")
        scenarios.append(ScenarioConfig.from_dict(s))

    return AgentTestConfig(
        similarity_threshold=similarity_threshold,
        default_replay_mode=default_replay_mode,
        ignore_fields=ignore_fields,
        agentgit_dir=config_data.get('agentgit_dir', '.agentgit'),
        project_dir=config_data.get('project_dir', '.'),
        tests=tests,
        scenarios=scenarios,
    )


def save_config(config: AgentTestConfig, path: str) -> None:
    try:
        import tomli_w
    except ImportError:
        raise ImportError(
            "Saving TOML requires 'tomli_w' package.\n"
            "Install: pip install tomli-w"
        )

    data = {
        'agenttest': {
            'similarity_threshold': config.similarity_threshold,
            'default_replay_mode': config.default_replay_mode,
            'ignore_fields': config.ignore_fields,
            'agentgit_dir': config.agentgit_dir,
            'project_dir': config.project_dir,
        },
        'tests': [t.to_dict() for t in config.tests],
        'scenarios': [s.to_dict() for s in config.scenarios],
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        tomli_w.dump(data, f)
