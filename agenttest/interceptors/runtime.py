import contextvars
import copy
import functools
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ReplayRuntimeContext:
    gatekeeper: Any
    mode: str
    baseline_name: str
    replay_name: Optional[str] = None


class LockedModeNetworkError(RuntimeError):
    pass


_ACTIVE_REPLAY_CONTEXT: contextvars.ContextVar[Optional[ReplayRuntimeContext]] = contextvars.ContextVar(
    "agenttest_active_replay_context",
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
        return original(self, input, *args, **kwargs)

    return wrapped


def _make_ainvoke_wrapper(original: Any):
    @functools.wraps(original)
    async def wrapped(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        decision = _maybe_intercept(self, input, kwargs, source="runtime")
        if decision is not None and decision.hit and decision.ai_message is not None:
            return decision.ai_message
        return await original(self, input, *args, **kwargs)

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
            return original(self, inputs, *args, **kwargs)

        if not isinstance(inputs, (list, tuple)):
            return original(self, inputs, *args, **kwargs)

        decisions = []
        all_cached = True

        for item in inputs:
            decision = _maybe_intercept(self, item, kwargs, source="runtime")
            decisions.append(decision)
            if decision is None or not decision.hit or decision.ai_message is None:
                all_cached = False

        if all_cached:
            return [decision.ai_message for decision in decisions]

        return original(self, inputs, *args, **kwargs)

    return wrapped


def _make_abatch_wrapper(original: Any):
    @functools.wraps(original)
    async def wrapped(self, inputs: Any, *args: Any, **kwargs: Any):
        ctx = _ACTIVE_REPLAY_CONTEXT.get()
        if ctx is None or not _mode_is_interceptable(ctx.mode):
            return await original(self, inputs, *args, **kwargs)

        if not isinstance(inputs, (list, tuple)):
            return await original(self, inputs, *args, **kwargs)

        decisions = []
        all_cached = True

        for item in inputs:
            decision = _maybe_intercept(self, item, kwargs, source="runtime")
            decisions.append(decision)
            if decision is None or not decision.hit or decision.ai_message is None:
                all_cached = False

        if all_cached:
            return [decision.ai_message for decision in decisions]

        return await original(self, inputs, *args, **kwargs)

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
