import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agentgit.langgraph_callback import langgraph_callback
from agenttest.models.config import ScenarioConfig
from agenttest.session import AgentTestSession


@dataclass
class ScenarioContext:
    """
    Values injected into scenario callables.

    A scenario entrypoint can accept any subset of these parameters:
    - session
    - callback
    - invoke_config
    - payload
    """

    session: AgentTestSession
    callback: Any
    invoke_config: Dict[str, Any]
    payload: Any


def parse_entrypoint(entrypoint: str):
    if ":" not in entrypoint:
        raise ValueError(
            f"Invalid scenario entrypoint '{entrypoint}'. Expected '<module>:<callable>'."
        )
    module_name, attr_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name, None)
    if target is None:
        raise ValueError(f"Entrypoint '{entrypoint}' not found.")
    if not callable(target):
        raise ValueError(f"Entrypoint '{entrypoint}' must be callable.")
    return target


def load_scenario_payload(project_dir: str, scenario: ScenarioConfig) -> Any:
    if scenario.input_data is not None:
        return scenario.input_data

    if scenario.input_file:
        input_path = Path(project_dir) / scenario.input_file
        if not input_path.exists():
            raise FileNotFoundError(f"Scenario input file not found: {input_path}")
        raw = input_path.read_text(encoding="utf-8")
        # Keep input flexible; if parsing fails, pass through as raw text.
        try:
            return json.loads(raw)
        except Exception:
            return raw

    return None


def build_scenario_context(session: AgentTestSession, payload: Any) -> ScenarioContext:
    callback = langgraph_callback(session.ag.eventbus)
    invoke_config = {
        "callbacks": [callback],
        "configurable": {
            "user_id": session.user_id,
            "session_id": session.session_id,
        },
    }
    return ScenarioContext(
        session=session,
        callback=callback,
        invoke_config=invoke_config,
        payload=payload,
    )


def run_scenario_callable(entrypoint: str, context: ScenarioContext) -> Any:
    fn = parse_entrypoint(entrypoint)
    signature = inspect.signature(fn)
    params = signature.parameters

    call_kwargs: Dict[str, Any] = {}
    if "session" in params:
        call_kwargs["session"] = context.session
    if "callback" in params:
        call_kwargs["callback"] = context.callback
    if "invoke_config" in params:
        call_kwargs["invoke_config"] = context.invoke_config
    if "payload" in params:
        call_kwargs["payload"] = context.payload

    # If function can be satisfied by keyword injection, use it.
    if call_kwargs or len(params) == 0:
        return fn(**call_kwargs)

    # Fallback positional conventions for simple scenario functions.
    arity = len(params)
    if arity == 1:
        return fn(context.payload)
    if arity == 2:
        return fn(context.session, context.payload)

    raise ValueError(
        "Unsupported scenario entrypoint signature. "
        "Use zero args, a single payload arg, (session, payload), or named kwargs."
    )
