import logging
from typing import Optional, Any
from contextlib import contextmanager

from agenttest.session import AgentTestSession
from agenttest.comparator import Comparison
from agenttest.models.comparison import ComparisonResult, StepStatus
from agenttest.models.config import AgentTestConfig
from agenttest.models.recording import Recording
from agenttest.interceptors.runtime import (
    install_global_runtime_interception,
    reset_active_recording_context,
    reset_active_replay_context,
    set_active_recording_context,
    set_active_replay_context,
)

try:
    from agenttest.interceptors.gatekeeper import LLMGatekeeper, ReplayMode
    INTERCEPTION_AVAILABLE = True
except ImportError:
    INTERCEPTION_AVAILABLE = False
    ReplayMode = None

logger = logging.getLogger(__name__)


class ReplayIntegrationError(RuntimeError):
    pass


class Replayer:
    def __init__(
        self,
        session: AgentTestSession,
        baseline_name: str,
        replay_name: Optional[str] = None,
        mode: str = "full",
        config: Optional[AgentTestConfig] = None
    ):
        self.session = session
        self.baseline_name = baseline_name
        self.replay_name = replay_name or f"{baseline_name}-replay"
        self.mode = mode.lower()
        self.config = config or session.config
        self.middleware = []

        if self.mode not in ["full", "selective", "locked"]:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be 'full', 'selective', or 'locked'"
            )

        if self.mode in ["selective", "locked"] and not INTERCEPTION_AVAILABLE:
            raise ValueError(
                f"Mode '{self.mode}' requires Phase 8.5 (LLM Interception).\n"
                f"Either install Phase 8.5 or use mode='full'"
            )

        self._baseline_recording: Optional[Recording] = None
        self._replay_recording: Optional[Recording] = None
        self._active_recording_id: Optional[str] = None
        self._gatekeeper: Optional[Any] = None
        self._wrapped_models_count = 0
        self._runtime_context_token = None
        self._runtime_recording_token = None

        self.comparison_result: Optional[ComparisonResult] = None
        self.baseline_details = []
        self.replay_details = []

    def __enter__(self):
        self._baseline_recording = self.session.get_recording_by_name(self.baseline_name)

        if not self._baseline_recording:
            raise ValueError(
                f"Baseline recording '{self.baseline_name}' not found.\n"
                f"Available recordings: {[r.name for r in self.session.list_recordings()]}"
            )

        self.baseline_details = self.session.get_recording_details(
            self._baseline_recording.recording_id
        )

        logger.info(
            "Loaded baseline '%s' (%d steps)",
            self.baseline_name,
            len(self.baseline_details),
        )

        if self.mode in ["selective", "locked"]:
            self._setup_interception()
            self._activate_runtime_context()

        self._start_replay_recording()
        self._activate_runtime_recording_context()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            is_locked_miss = False
            if exc_type is not None and INTERCEPTION_AVAILABLE:
                try:
                    from agenttest.interceptors.gatekeeper import LLMCacheMissError
                    if issubclass(exc_type, LLMCacheMissError):
                        is_locked_miss = True
                except ImportError:
                    pass

            completion_error = "Replay failed" if exc_type is not None else None
            self._complete_replay_recording(
                success=(exc_type is None),
                error_message=completion_error,
            )

            self._replay_recording = self.session.get_recording_by_name(self.replay_name)

            if self._replay_recording:
                self.replay_details = self.session.get_recording_details(
                    self._replay_recording.recording_id
                )
                if is_locked_miss:
                    logger.warning(
                        "Locked mode cache miss: captured %d of %d baseline steps; running comparison",
                        len(self.replay_details),
                        len(self.baseline_details),
                    )
                else:
                    logger.info(
                        "Replay '%s' captured %d steps",
                        self.replay_name,
                        len(self.replay_details),
                    )

                if exc_type is None or is_locked_miss:
                    self._assert_interception_integration()

                if self.mode == "locked":
                    live_steps = [d for d in self.replay_details if not d.was_cache_hit]
                    if live_steps:
                        raise ReplayIntegrationError(
                            "LOCKED mode executed live LLM call(s). "
                            "This indicates interception was bypassed. "
                            "Use AgentTest auto runtime interception or replayer.wrap_model(llm)."
                        )

                self._compare()
                if self.comparison_result:
                    self._store_comparison()

            if is_locked_miss:
                return True  # suppress LLMCacheMissError — outcome is in comparison_result

            return False
        finally:
            reset_active_replay_context(self._runtime_context_token)
            self._runtime_context_token = None
            reset_active_recording_context(self._runtime_recording_token)
            self._runtime_recording_token = None

    def _setup_interception(self):
        if not INTERCEPTION_AVAILABLE:
            return

        self._gatekeeper = LLMGatekeeper(self.session)
        self._gatekeeper.load_baseline_cache(self._baseline_recording.recording_id)

        mode_map = {
            "selective": ReplayMode.SELECTIVE,
            "locked": ReplayMode.LOCKED
        }

        replay_mode = mode_map[self.mode]
        self.middleware = self._gatekeeper.create_middleware(mode=replay_mode)

        logger.info(
            "LLM interception enabled (mode=%s, cache_entries=%d)",
            self.mode,
            len(self._gatekeeper._cache),
        )

    def _activate_runtime_context(self):
        if self.mode == "full" or not self._gatekeeper:
            return

        install_global_runtime_interception()
        self._runtime_context_token = set_active_replay_context(
            gatekeeper=self._gatekeeper,
            mode=self.mode,
            baseline_name=self.baseline_name,
            replay_name=self.replay_name,
        )

    def _activate_runtime_recording_context(self):
        install_global_runtime_interception()
        self._runtime_recording_token = set_active_recording_context(
            session=self.session,
            mode="replay",
            run_name=self.replay_name,
        )

    def _assert_interception_integration(self):
        if self.mode not in ["selective", "locked"] or not self._gatekeeper:
            return

        stats = self._gatekeeper.get_stats()
        expected_llm_activity = len(self.baseline_details) > 0 or len(self.replay_details) > 0
        if expected_llm_activity and stats.get("interception_attempts", 0) == 0:
            raise ReplayIntegrationError(
                "Replay captured LLM activity but no interception attempts were observed. "
                "Ensure AgentTest runtime interception is active, or wrap models with "
                "replayer.wrap_model(llm)."
            )

    def wrap_model(self, llm: Any, provider: Optional[str] = None, method: Optional[str] = None) -> Any:
        if self.mode == "full" or not self._gatekeeper:
            return llm
        self._wrapped_models_count += 1
        return self._gatekeeper.wrap_model(llm, provider=provider, method=method)

    def _start_replay_recording(self):
        recording = self.session.create_recording(
            name=self.replay_name,
            config=self.config,
            metadata={
                "recording_kind": "replay",
                "baseline_name": self.baseline_name,
            },
        )
        self._active_recording_id = recording.recording_id
        logger.info("Recording started: '%s'", self.replay_name)

    def _complete_replay_recording(self, success: bool, error_message: Optional[str] = None):
        if not self._active_recording_id:
            return

        if success:
            self.session.complete_recording(self._active_recording_id)
            status = "completed"
        else:
            self.session.complete_recording(
                self._active_recording_id,
                error=error_message or "Replay failed"
            )
            status = "failed"
        logger.info("Recording stopped: '%s' (%s)", self.replay_name, status)

    def _compare(self):
        logger.info("Comparing baseline vs replay")

        comparator = Comparison(
            similarity_threshold=self.config.similarity_threshold,
            ignore_fields=self.config.ignore_fields,
            semantic_model=None
        )

        self.comparison_result = comparator.compare_recordings(
            self.baseline_details,
            self.replay_details
        )

        if self.comparison_result.overall_pass:
            logger.info("No regressions detected (%d matched steps)", self.comparison_result.matched_steps)
        else:
            if self.comparison_result.has_regression:
                logger.warning(
                    "Regression detected (%d regression step(s))",
                    self.comparison_result.regression_steps,
                )
            if self.comparison_result.has_delta:
                logger.warning(
                    "Delta detected (%d new/changed step(s))",
                    self.comparison_result.delta_steps,
                )
            logger.info("Root cause at step %s", self.comparison_result.root_cause_index)

    def _store_comparison(self):
        self.session.store_comparison(self.comparison_result)
        logger.info("Comparison stored in database")

    @property
    def passed(self) -> bool:
        return (
            self.comparison_result is not None
            and self.comparison_result.overall_pass
        )


    @property
    def root_cause_summary(self) -> Optional[str]:
        if not self.comparison_result:
            return None

        if self.comparison_result.root_cause_index is None:
            return None

        root_step = self.comparison_result.step_comparisons[
            self.comparison_result.root_cause_index
        ]

        return f"Step {root_step.step_index}: {root_step.diff_summary}"

    @property
    def cache_stats(self) -> Optional[dict]:
        if self._gatekeeper:
            return self._gatekeeper.get_stats()
        return None

    def get_diverged_steps(self) -> list:
        if not self.comparison_result:
            return []

        return [
            step for step in self.comparison_result.step_comparisons
            if step.status == StepStatus.DIVERGE
        ]

    def print_report(self):
        if not self.comparison_result:
            print("No comparison result available")
            return

        print("\n" + "=" * 70)
        print(f"REPLAY REPORT: {self.replay_name}")
        print("=" * 70)

        result = self.comparison_result

        if result.overall_pass:
            status_label = "✅ PASS"
        elif result.has_regression and result.has_delta:
            status_label = "❌ REGRESSION  ⚠ DELTA"
        elif result.has_regression:
            status_label = "❌ REGRESSION"
        else:
            status_label = "⚠  DELTA — Review Required"

        print(f"\nOutcome: {status_label}")
        print(f"Mode:    {self.mode.upper()}")

        print(f"\nSteps:")
        print(f"  Total: {result.total_steps}")
        print(f"  Matched: {result.matched_steps}")
        print(f"  Diverged: {result.mismatched_steps}")
        print(f"  Added: {result.added_steps}")
        print(f"  Removed: {result.removed_steps}")

        if self.cache_stats:
            stats = self.cache_stats
            print(f"\nCache Performance:")
            print(f"  Hits: {stats['cache_hits']}")
            print(f"  Misses: {stats['cache_misses']}")
            print(f"  Live calls: {stats['live_calls']}")
            print(f"  Hit rate: {stats['cache_hit_rate']:.1%}")

        if result.root_cause_index is not None:
            print(f"\nRoot Cause:")
            print(f"  {self.root_cause_summary}")

        diverged = self.get_diverged_steps()
        if diverged:
            print(f"\nDivergences ({len(diverged)}):")
            for step in diverged[:5]:
                print(f"  Step {step.step_index}: {step.diff_summary}")
            if len(diverged) > 5:
                print(f"  ... and {len(diverged) - 5} more")

        print("=" * 70)


@contextmanager
def replay_against(
    session: AgentTestSession,
    baseline_name: str,
    mode: str = "full",
    **kwargs
):
    replayer = Replayer(
        session=session,
        baseline_name=baseline_name,
        mode=mode,
        **kwargs
    )

    with replayer:
        yield replayer
