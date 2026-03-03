import contextvars
import copy
import functools
import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from agentgit.event import Event, EventType
from agenttest.fingerprint import compute_fingerprint

logger = logging.getLogger(__name__)


@dataclass
class ReplayRuntimeContext:
    gatekeeper: Any
    mode: str
    baseline_name: str
    replay_name: Optional[str] = None


@dataclass
class RecordingRuntimeContext:
    session: Any
    mode: str  # record | replay
    run_name: str


class LockedModeNetworkError(RuntimeError):
    pass


_ACTIVE_REPLAY_CONTEXT: contextvars.ContextVar[Optional[ReplayRuntimeContext]] = contextvars.ContextVar(
    "agenttest_active_replay_context",
    default=None,
)
_ACTIVE_RECORDING_CONTEXT: contextvars.ContextVar[Optional[RecordingRuntimeContext]] = contextvars.ContextVar(
    "agenttest_active_recording_context",
    default=None,
)

_PATCH_LOCK = threading.RLock()
_PATCHES: List[Tuple[Any, str, Any]] = []
_PATCHED_LANGCHAIN = False
_PATCHED_NETWORK = False

_AI_MESSAGE_CHUNK_CLASS = None

_BLOCKED_LLM_HOSTS = (
    "api.openai.com",
    "openai.azure.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
    "api.cohere.ai",
    "api.mistral.ai",
    "openrouter.ai",
    "api.together.xyz",
    "api.groq.com",
    "api.fireworks.ai",
    "api.deepseek.com",
)


def set_active_replay_context(
    gatekeeper: Any,
    mode: str,
    baseline_name: str,
    replay_name: Optional[str] = None,
):
    ctx = ReplayRuntimeContext(
        gatekeeper=gatekeeper,
        mode=mode,
        baseline_name=baseline_name,
        replay_name=replay_name,
    )
    return _ACTIVE_REPLAY_CONTEXT.set(ctx)


def reset_active_replay_context(token: Any) -> None:
    if token is None:
        return
    _ACTIVE_REPLAY_CONTEXT.reset(token)


def get_active_replay_context() -> Optional[ReplayRuntimeContext]:
    return _ACTIVE_REPLAY_CONTEXT.get()


def set_active_recording_context(session: Any, mode: str, run_name: str):
    ctx = RecordingRuntimeContext(
        session=session,
        mode=mode,
        run_name=run_name,
    )
    return _ACTIVE_RECORDING_CONTEXT.set(ctx)


def reset_active_recording_context(token: Any) -> None:
    if token is None:
        return
    _ACTIVE_RECORDING_CONTEXT.reset(token)


def get_active_recording_context() -> Optional[RecordingRuntimeContext]:
    return _ACTIVE_RECORDING_CONTEXT.get()


def install_global_runtime_interception() -> bool:
    global _PATCHED_LANGCHAIN, _PATCHED_NETWORK

    with _PATCH_LOCK:
        if _PATCHED_LANGCHAIN and _PATCHED_NETWORK:
            return True

        if not _PATCHED_LANGCHAIN:
            _PATCHED_LANGCHAIN = _install_langchain_model_patches()

        if not _PATCHED_NETWORK:
            _PATCHED_NETWORK = _install_network_guard_patches()

        return _PATCHED_LANGCHAIN or _PATCHED_NETWORK


def uninstall_global_runtime_interception() -> None:
    global _PATCHED_LANGCHAIN, _PATCHED_NETWORK

    with _PATCH_LOCK:
        while _PATCHES:
            target, attr_name, original = _PATCHES.pop()
            setattr(target, attr_name, original)

        _PATCHED_LANGCHAIN = False
        _PATCHED_NETWORK = False


def is_runtime_interception_installed() -> bool:
    return _PATCHED_LANGCHAIN or _PATCHED_NETWORK


def _install_langchain_model_patches() -> bool:
    global _AI_MESSAGE_CHUNK_CLASS

    try:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessageChunk
    except Exception:
        logger.info("LangChain runtime interception unavailable: langchain_core not installed")
        return False

    _AI_MESSAGE_CHUNK_CLASS = AIMessageChunk

    _patch_method(BaseChatModel, "invoke", _make_invoke_wrapper)
    _patch_method(BaseChatModel, "ainvoke", _make_ainvoke_wrapper)
    _patch_method(BaseChatModel, "stream", _make_stream_wrapper)
    _patch_method(BaseChatModel, "astream", _make_astream_wrapper)
    _patch_method(BaseChatModel, "batch", _make_batch_wrapper)
    _patch_method(BaseChatModel, "abatch", _make_abatch_wrapper)

    logger.info("Installed AgentTest runtime interception patches on BaseChatModel")
    return True


def _install_network_guard_patches() -> bool:
    installed_any = False

    try:
        import httpx

        _patch_method(httpx.Client, "request", _make_httpx_request_wrapper)
        _patch_method(httpx.AsyncClient, "request", _make_httpx_async_request_wrapper)
        installed_any = True
    except Exception:
        pass

    try:
        import requests

        _patch_method(requests.sessions.Session, "request", _make_requests_request_wrapper)
        installed_any = True
    except Exception:
        pass

    try:
        import aiohttp

        _patch_method(aiohttp.ClientSession, "_request", _make_aiohttp_request_wrapper)
        installed_any = True
    except Exception:
        pass

    if installed_any:
        logger.info("Installed AgentTest locked-mode outbound network guard")
    return installed_any


def _patch_method(target: Any, attr_name: str, wrapper_factory: Any) -> None:
    original = getattr(target, attr_name, None)
    if original is None:
        return

    if getattr(original, "__agenttest_runtime_patched__", False):
        return

    wrapped = wrapper_factory(original)
    setattr(wrapped, "__agenttest_runtime_patched__", True)

    setattr(target, attr_name, wrapped)
    _PATCHES.append((target, attr_name, original))


def _mode_is_interceptable(mode: str) -> bool:
    return str(mode).lower() in {"locked", "selective"}


def _mode_is_locked(mode: str) -> bool:
    return str(mode).lower() == "locked"


def _coerce_messages(raw_input: Any) -> List[Any]:
    if raw_input is None:
        return []

    if isinstance(raw_input, list):
        return raw_input

    if isinstance(raw_input, tuple):
        return list(raw_input)

    to_messages = getattr(raw_input, "to_messages", None)
    if callable(to_messages):
        try:
            messages = to_messages()
            if isinstance(messages, list):
                return messages
        except Exception:
            pass

    return [raw_input]


def _resolve_invocation(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    invocation = kwargs.get("invocation_params", {})
    if not isinstance(invocation, dict):
        return {}
    return invocation


def _resolve_model_name(llm: Any, kwargs: Dict[str, Any]) -> str:
    invocation = _resolve_invocation(kwargs)
    model = (
        invocation.get("model_name")
        or invocation.get("model")
        or kwargs.get("model")
        or getattr(llm, "model_name", "")
        or getattr(llm, "model", "")
        or ""
    )
    return str(model)


def _resolve_tools(kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    invocation = _resolve_invocation(kwargs)
    tools = invocation.get("tools") or invocation.get("functions") or kwargs.get("tools") or []
    if not isinstance(tools, list):
        return []
    return [tool for tool in tools if isinstance(tool, dict)]


def _infer_provider_from_llm(llm: Any) -> str:
    name = llm.__class__.__name__.lower()
    if "openai" in name:
        return "azure_openai" if "azure" in name else "openai"
    if "anthropic" in name:
        return "anthropic"
    if "google" in name:
        return "google"
    if "cohere" in name:
        return "cohere"
    if "mistral" in name:
        return "mistral"
    return "unknown"


def _infer_method_from_provider(provider: str) -> str:
    method_map = {
        "openai": "chat.completions.create",
        "azure_openai": "chat.completions.create",
        "anthropic": "messages.create",
        "google": "generateContent",
        "cohere": "chat",
        "mistral": "chat",
    }
    return method_map.get(provider, "chat")


def _extract_callbacks(kwargs: Dict[str, Any]) -> List[Any]:
    callbacks: List[Any] = []
    direct = kwargs.get("callbacks")
    if isinstance(direct, list):
        callbacks.extend(direct)

    config = kwargs.get("config")
    if isinstance(config, dict):
        cfg_callbacks = config.get("callbacks")
        if isinstance(cfg_callbacks, list):
            callbacks.extend(cfg_callbacks)
    return callbacks


def _has_langgraph_callback(kwargs: Dict[str, Any]) -> bool:
    for cb in _extract_callbacks(kwargs):
        cls_name = cb.__class__.__name__.lower()
        module = cb.__class__.__module__.lower()
        if cls_name == "langgraph_callback":
            return True
        if "agentgit.langgraph_callback" in module:
            return True
    return False


def _extract_usage(ai_message: Any) -> Optional[Dict[str, Any]]:
    if ai_message is None:
        return None
    response_metadata = getattr(ai_message, "response_metadata", None) or {}
    usage = response_metadata.get("usage")
    if isinstance(usage, dict):
        return copy.deepcopy(usage)

    usage_metadata = getattr(ai_message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        return copy.deepcopy(usage_metadata)
    return None


def _build_response_data(ai_message: Any) -> Dict[str, Any]:
    if ai_message is None:
        return {"content": "", "tool_calls": [], "usage": None}

    content = getattr(ai_message, "content", "")
    tool_calls = getattr(ai_message, "tool_calls", None) or []
    usage = _extract_usage(ai_message)
    return {
        "content": str(content or ""),
        "tool_calls": copy.deepcopy(tool_calls) if isinstance(tool_calls, list) else [],
        "usage": usage,
    }


def _emit_runtime_llm_event(
    *,
    llm: Any,
    raw_input: Any,
    kwargs: Dict[str, Any],
    ai_message: Any,
    was_cache_hit: bool,
) -> None:
    record_ctx = _ACTIVE_RECORDING_CONTEXT.get()
    if record_ctx is None:
        return
    if _has_langgraph_callback(kwargs):
        # Callback path already emits this event; avoid duplicate detail rows.
        return

    session = getattr(record_ctx, "session", None)
    if session is None:
        return

    gatekeeper = getattr(get_active_replay_context(), "gatekeeper", None)
    messages = _coerce_messages(raw_input)

    if gatekeeper is not None:
        request_params = gatekeeper._build_request_params(
            messages=messages,
            model=_resolve_model_name(llm, kwargs),
            tools=_resolve_tools(kwargs),
            invocation=_resolve_invocation(kwargs),
        )
        provider = gatekeeper._infer_provider(llm)
        method = gatekeeper._infer_method(provider)
    else:
        # Recording path without gatekeeper.
        request_params = {
            "model": _resolve_model_name(llm, kwargs),
            "messages": [
                {
                    "role": str(getattr(msg, "type", None) or getattr(msg, "role", None) or "unknown"),
                    "content": str(getattr(msg, "content", "")),
                }
                if not isinstance(msg, dict)
                else {
                    "role": str(msg.get("role") or msg.get("type") or "unknown"),
                    "content": str(msg.get("content", "")),
                }
                for msg in messages
            ],
            "tools": _resolve_tools(kwargs),
            "temperature": _resolve_invocation(kwargs).get("temperature"),
            "max_tokens": _resolve_invocation(kwargs).get("max_tokens"),
            "top_p": _resolve_invocation(kwargs).get("top_p"),
            "stream": bool(_resolve_invocation(kwargs).get("stream", False)),
        }
        provider = _infer_provider_from_llm(llm)
        method = _infer_method_from_provider(provider)

    response_data = _build_response_data(ai_message)
    fingerprint = compute_fingerprint(provider, method, request_params)

    session.ag.eventbus.publish(
        EventType.LLM_CALL_END,
        Event(
            type=EventType.LLM_CALL_END,
            user_id=session.user_id,
            session_id=session.session_id,
            run_id=f"runtime-{uuid.uuid4().hex[:10]}",
            model=str(request_params.get("model", "")),
            content=response_data.get("content", ""),
            usage=response_data.get("usage"),
            duration_ms=0,
            metadata={
                "provider": provider,
                "method": method,
                "fingerprint": fingerprint,
                "request_params": request_params,
                "response_data": response_data,
                "is_streaming": False,
                "was_cache_hit": was_cache_hit,
                "source": "runtime",
            },
        ),
    )


def _maybe_intercept(llm: Any, raw_input: Any, kwargs: Dict[str, Any], source: str) -> Any:
    ctx = _ACTIVE_REPLAY_CONTEXT.get()
    if ctx is None:
        return None

    if not _mode_is_interceptable(ctx.mode):
        return None

    gatekeeper = getattr(ctx, "gatekeeper", None)
    if gatekeeper is None:
        return None

    messages = _coerce_messages(raw_input)
    request_params = gatekeeper._build_request_params(
        messages=messages,
        model=_resolve_model_name(llm, kwargs),
        tools=_resolve_tools(kwargs),
        invocation=_resolve_invocation(kwargs),
    )

    provider = gatekeeper._infer_provider(llm)
    method = gatekeeper._infer_method(provider)

    return gatekeeper.check_request(
        provider=provider,
        method=method,
        request_params=request_params,
        source=source,
    )


def _cached_stream_message(ai_message: Any) -> Any:
    if _AI_MESSAGE_CHUNK_CLASS is None:
        return ai_message

    additional_kwargs = getattr(ai_message, "additional_kwargs", {}) or {}
    response_metadata = getattr(ai_message, "response_metadata", {}) or {}

    return _AI_MESSAGE_CHUNK_CLASS(
        content=str(getattr(ai_message, "content", "")),
        additional_kwargs=copy.deepcopy(additional_kwargs),
        response_metadata=copy.deepcopy(response_metadata),
    )


def _make_invoke_wrapper(original: Any):
    @functools.wraps(original)
    def wrapped(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        decision = _maybe_intercept(self, input, kwargs, source="runtime")
        if decision is not None and decision.hit and decision.ai_message is not None:
            return decision.ai_message
        output = original(self, input, *args, **kwargs)
        _emit_runtime_llm_event(
            llm=self,
            raw_input=input,
            kwargs=kwargs,
            ai_message=output,
            was_cache_hit=False,
        )
        return output

    return wrapped


def _make_ainvoke_wrapper(original: Any):
    @functools.wraps(original)
    async def wrapped(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        decision = _maybe_intercept(self, input, kwargs, source="runtime")
        if decision is not None and decision.hit and decision.ai_message is not None:
            return decision.ai_message
        output = await original(self, input, *args, **kwargs)
        _emit_runtime_llm_event(
            llm=self,
            raw_input=input,
            kwargs=kwargs,
            ai_message=output,
            was_cache_hit=False,
        )
        return output

    return wrapped


def _make_stream_wrapper(original: Any):
    @functools.wraps(original)
    def wrapped(self, input: Any, *args: Any, **kwargs: Any):
        decision = _maybe_intercept(self, input, kwargs, source="runtime")
        if decision is not None and decision.hit and decision.ai_message is not None:
            cached = _cached_stream_message(decision.ai_message)

            def _iterator():
                yield cached

            return _iterator()

        return original(self, input, *args, **kwargs)

    return wrapped


def _make_astream_wrapper(original: Any):
    @functools.wraps(original)
    async def wrapped(self, input: Any, *args: Any, **kwargs: Any):
        decision = _maybe_intercept(self, input, kwargs, source="runtime")
        if decision is not None and decision.hit and decision.ai_message is not None:
            cached = _cached_stream_message(decision.ai_message)
            yield cached
            return

        async for chunk in original(self, input, *args, **kwargs):
            yield chunk

    return wrapped


def _make_batch_wrapper(original: Any):
    @functools.wraps(original)
    def wrapped(self, inputs: Any, *args: Any, **kwargs: Any):
        ctx = _ACTIVE_REPLAY_CONTEXT.get()
        if ctx is None or not _mode_is_interceptable(ctx.mode):
            outputs = original(self, inputs, *args, **kwargs)
            if isinstance(inputs, (list, tuple)) and isinstance(outputs, list):
                for idx, item in enumerate(inputs):
                    ai_message = outputs[idx] if idx < len(outputs) else None
                    _emit_runtime_llm_event(
                        llm=self,
                        raw_input=item,
                        kwargs=kwargs,
                        ai_message=ai_message,
                        was_cache_hit=False,
                    )
            return outputs

        if not isinstance(inputs, (list, tuple)):
            output = original(self, inputs, *args, **kwargs)
            _emit_runtime_llm_event(
                llm=self,
                raw_input=inputs,
                kwargs=kwargs,
                ai_message=output,
                was_cache_hit=False,
            )
            return output

        decisions = []
        all_cached = True

        for item in inputs:
            decision = _maybe_intercept(self, item, kwargs, source="runtime")
            decisions.append(decision)
            if decision is None or not decision.hit or decision.ai_message is None:
                all_cached = False

        if all_cached:
            return [decision.ai_message for decision in decisions]

        outputs = original(self, inputs, *args, **kwargs)
        if isinstance(outputs, list):
            for idx, item in enumerate(inputs):
                decision = decisions[idx]
                if decision is not None and decision.hit:
                    continue
                ai_message = outputs[idx] if idx < len(outputs) else None
                _emit_runtime_llm_event(
                    llm=self,
                    raw_input=item,
                    kwargs=kwargs,
                    ai_message=ai_message,
                    was_cache_hit=False,
                )
        return outputs

    return wrapped


def _make_abatch_wrapper(original: Any):
    @functools.wraps(original)
    async def wrapped(self, inputs: Any, *args: Any, **kwargs: Any):
        ctx = _ACTIVE_REPLAY_CONTEXT.get()
        if ctx is None or not _mode_is_interceptable(ctx.mode):
            outputs = await original(self, inputs, *args, **kwargs)
            if isinstance(inputs, (list, tuple)) and isinstance(outputs, list):
                for idx, item in enumerate(inputs):
                    ai_message = outputs[idx] if idx < len(outputs) else None
                    _emit_runtime_llm_event(
                        llm=self,
                        raw_input=item,
                        kwargs=kwargs,
                        ai_message=ai_message,
                        was_cache_hit=False,
                    )
            return outputs

        if not isinstance(inputs, (list, tuple)):
            output = await original(self, inputs, *args, **kwargs)
            _emit_runtime_llm_event(
                llm=self,
                raw_input=inputs,
                kwargs=kwargs,
                ai_message=output,
                was_cache_hit=False,
            )
            return output

        decisions = []
        all_cached = True

        for item in inputs:
            decision = _maybe_intercept(self, item, kwargs, source="runtime")
            decisions.append(decision)
            if decision is None or not decision.hit or decision.ai_message is None:
                all_cached = False

        if all_cached:
            return [decision.ai_message for decision in decisions]

        outputs = await original(self, inputs, *args, **kwargs)
        if isinstance(outputs, list):
            for idx, item in enumerate(inputs):
                decision = decisions[idx]
                if decision is not None and decision.hit:
                    continue
                ai_message = outputs[idx] if idx < len(outputs) else None
                _emit_runtime_llm_event(
                    llm=self,
                    raw_input=item,
                    kwargs=kwargs,
                    ai_message=ai_message,
                    was_cache_hit=False,
                )
        return outputs

    return wrapped


def _extract_host(url: Any) -> str:
    if hasattr(url, "host"):
        host = getattr(url, "host")
        if isinstance(host, bytes):
            host = host.decode("utf-8", errors="ignore")
        if host:
            return str(host).lower()

    raw = str(url)
    try:
        parsed = urlparse(raw)
        if parsed.hostname:
            return parsed.hostname.lower()
    except Exception:
        pass

    return ""


def _is_blocked_llm_host(host: str) -> bool:
    normalized = host.strip().lower().strip(".")
    if not normalized:
        return False

    for blocked in _BLOCKED_LLM_HOSTS:
        if normalized == blocked:
            return True
        if normalized.endswith(f".{blocked}"):
            return True

    return False


def _should_block_url(url: Any) -> bool:
    ctx = _ACTIVE_REPLAY_CONTEXT.get()
    if ctx is None:
        return False

    if not _mode_is_locked(ctx.mode):
        return False

    host = _extract_host(url)
    return _is_blocked_llm_host(host)


def _raise_blocked(url: Any, method: str) -> None:
    host = _extract_host(url) or str(url)
    raise LockedModeNetworkError(
        "LOCKED mode blocked outbound LLM provider request "
        f"({method.upper()} {host}). Ensure AgentTest interception is active."
    )


def _make_httpx_request_wrapper(original: Any):
    @functools.wraps(original)
    def wrapped(self, method: str, url: Any, *args: Any, **kwargs: Any):
        if _should_block_url(url):
            _raise_blocked(url, method)
        return original(self, method, url, *args, **kwargs)

    return wrapped


def _make_httpx_async_request_wrapper(original: Any):
    @functools.wraps(original)
    async def wrapped(self, method: str, url: Any, *args: Any, **kwargs: Any):
        if _should_block_url(url):
            _raise_blocked(url, method)
        return await original(self, method, url, *args, **kwargs)

    return wrapped


def _make_requests_request_wrapper(original: Any):
    @functools.wraps(original)
    def wrapped(self, method: str, url: str, *args: Any, **kwargs: Any):
        if _should_block_url(url):
            _raise_blocked(url, method)
        return original(self, method, url, *args, **kwargs)

    return wrapped


def _make_aiohttp_request_wrapper(original: Any):
    @functools.wraps(original)
    async def wrapped(self, method: str, str_or_url: Any, *args: Any, **kwargs: Any):
        if _should_block_url(str_or_url):
            _raise_blocked(str_or_url, method)
        return await original(self, method, str_or_url, *args, **kwargs)

    return wrapped
