# AgentTest Guarantee Test Matrix

This suite validates the service-level contract using mostly real `AgentTestSession` + `Recorder` + `Replayer` flows.

## Coverage

- `tests/guarantees/test_fingerprint_contract.py`
  - Fingerprint is structural (roles/tool names/model), not text-content based.
  - Tool-set/order and role-order changes alter fingerprint.

- `tests/guarantees/test_structural_hard_guarantees.py`
  - G1.1 add node detection
  - G1.2 remove node detection
  - G1.3 order change detection
  - G1.4 routing/branch change detection
  - G1.5 tool-set change detection

- `tests/guarantees/test_replay_mode_guarantees.py`
  - C1/C4 locked mode zero live calls
  - C2 selective mode live cost proportional to changed steps
  - G3.4 provider outage independence in locked mode (cache-first behavior)
  - G4.4 repeatability in locked mode with fixed baseline
  - Locked network guard for uninstrumented outbound LLM calls

- `tests/guarantees/test_process_cli_guarantees.py`
  - G4.1 accept flow requires explicit baseline promotion action
  - G4.2 baseline tags carry auditable recording metadata
  - CLI record path includes `--agenttest` to enforce instrumentation

- `tests/guarantees/test_pytest_plugin_auto_integration.py`
  - Plugin auto-wrap path works with marker-only tests (no manual replayer code)
  - Env-gated: `AGENTTEST_RUN_LANGCHAIN_RUNTIME=1`

- `tests/guarantees/test_explicit_non_guarantees.py`
  - X1/X2/X3 explicit non-guarantees: consistency does not imply correctness or production truth.

- `tests/integration/test_gemini_live_optional.py`
  - Optional real-world path with `gemini-2.0-flash` (env gated).

## Running

Local (no live APIs):

```bash
pytest tests/guarantees -v
```

Run env-gated LangChain runtime integration:

```bash
AGENTTEST_RUN_LANGCHAIN_RUNTIME=1 pytest tests/guarantees/test_pytest_plugin_auto_integration.py -v
```

Optional live Gemini:

```bash
AGENTTEST_RUN_LIVE=1 GOOGLE_API_KEY=... pytest tests/integration/test_gemini_live_optional.py -v
```
