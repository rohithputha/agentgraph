from uuid import UUID
import time
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.callbacks import BaseCallbackHandler

from .eventbus import Eventbus
from .event import Event, EventType

# Import fingerprinting for AgentTest
try:
    from agenttest.fingerprint import compute_fingerprint
    AGENTTEST_AVAILABLE = True
except ImportError:
    AGENTTEST_AVAILABLE = False
    def compute_fingerprint(*args, **kwargs):
        return ""  # Fallback if agenttest not installed
class langgraph_callback(BaseCallbackHandler):
    def __init__(self, eventbus):
        super().__init__()
        self.eventbus = eventbus
        self._runs = {}
        self._tool_runs = {}
        self._context_map = {} # run_id -> (user_id, session_id)
    
    def _get_session_context(self, kwargs: dict, run_id: str = None, parent_run_id: str = None, metadata: dict = None) -> tuple[str, str]:
        """Extract user_id and session_id from config, metadata, or context map."""
        # 1. Try config/configurable
        config = kwargs.get("config") or {}
        configurable = config.get("configurable") or {}
        user_id = configurable.get("user_id")
        session_id = configurable.get("session_id")
        
        # 2. Try metadata
        if not user_id or not session_id:
            meta = metadata or kwargs.get("metadata") or {}
            user_id = meta.get("user_id")
            session_id = meta.get("session_id")

        # 3. Try context map (self or parent)
        if not user_id or not session_id:
            if run_id and run_id in self._context_map:
                user_id, session_id = self._context_map[run_id]
            elif parent_run_id and parent_run_id in self._context_map:
                user_id, session_id = self._context_map[parent_run_id]
                
        return user_id or "default", session_id or "default"

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        # print(f"DEBUG: on_chain_start run_id={run_id} parent={parent_run_id}")
        user_id, session_id = self._get_session_context(kwargs, str(run_id), str(parent_run_id) if parent_run_id else None, metadata)
        self._context_map[str(run_id)] = (user_id, session_id)
    

    def _extract_model(self, serialized: Dict, kwargs: Dict) -> str:
        inv = kwargs.get("invocation_params", {})
        return (
            inv.get("model_name")
            or inv.get("model")
            or (serialized or {}).get("name", "unknown")
        )

    def _extract_provider(self, serialized: Dict[str, Any]) -> str:
        """
        Extract provider name from model class.

        Examples:
            ChatOpenAI → openai
            ChatAnthropic → anthropic
            ChatGoogleGenerativeAI → google
            AzureChatOpenAI → azure_openai
        """
        name = (serialized or {}).get("name", "unknown")
        name_lower = name.lower()

        # Check for known providers
        if "openai" in name_lower:
            return "azure_openai" if "azure" in name_lower else "openai"
        elif "anthropic" in name_lower:
            return "anthropic"
        elif "google" in name_lower:
            return "google"
        elif "cohere" in name_lower:
            return "cohere"
        elif "mistral" in name_lower:
            return "mistral"

        # Fallback: use the class name
        return name_lower

    def _extract_method(self, provider: str) -> str:
        """
        Infer API method from provider.

        This is a reasonable approximation since LangChain abstracts the actual API.
        """
        method_map = {
            "openai": "chat.completions.create",
            "azure_openai": "chat.completions.create",
            "anthropic": "messages.create",
            "google": "generateContent",
            "cohere": "chat",
            "mistral": "chat"
        }
        return method_map.get(provider, "chat")

    def _extract_tools(self, kwargs: Dict[str, Any]) -> List[Dict]:
        """
        Extract tools from invocation_params.

        LangChain passes tools in invocation_params when bind_tools() is used.
        """
        invocation_params = kwargs.get("invocation_params", {})
        tools = invocation_params.get("tools", [])

        # Tools might also be in 'functions' (deprecated OpenAI format)
        if not tools:
            tools = invocation_params.get("functions", [])

        return tools if isinstance(tools, list) else []

    def _build_request_params(
        self,
        model: str,
        flat_messages: List[Dict],
        tools: List[Dict],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build request_params dict for fingerprinting and storage.

        This captures the structural aspects of the request.
        """
        invocation_params = kwargs.get("invocation_params", {})

        return {
            "model": model,
            "messages": flat_messages,
            "tools": tools,
            "temperature": invocation_params.get("temperature"),
            "max_tokens": invocation_params.get("max_tokens"),
            "top_p": invocation_params.get("top_p"),
            "stream": invocation_params.get("stream", False),
        }

    def _extract_tool_calls(self, response) -> List[Dict]:
        """
        Extract tool calls from LLM response.

        Returns list of dicts with structure:
            [{"id": "call_123", "type": "function", "name": "get_weather", "args": {...}}]
        """
        tool_calls = []

        try:
            if response.generations and response.generations[0]:
                gen = response.generations[0][0]
                if hasattr(gen, "message") and hasattr(gen.message, "tool_calls"):
                    raw_calls = gen.message.tool_calls or []
                    for tc in raw_calls:
                        if isinstance(tc, dict):
                            tool_calls.append({
                                "id": tc.get("id", ""),
                                "type": tc.get("type", "function"),
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {})
                            })
        except Exception:
            # Gracefully handle any extraction errors
            pass

        return tool_calls

    def _build_response_data(
        self,
        text: Optional[str],
        tool_calls: List[Dict],
        usage: Optional[Dict],
        response
    ) -> Dict[str, Any]:
        """
        Build response_data dict for storage and comparison.
        """
        response_data = {
            "content": text or "",
            "tool_calls": tool_calls,
            "usage": usage,
        }

        # Add response_metadata if available
        try:
            if response.generations and response.generations[0]:
                gen = response.generations[0][0]
                if hasattr(gen, "message"):
                    meta = getattr(gen.message, "response_metadata", {})
                    if meta:
                        response_data["response_metadata"] = meta
        except Exception:
            pass

        return response_data
    
    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]], *, run_id: UUID, parent_run_id: UUID | None = None, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, **kwargs: Any):
        """ENHANCED: Now captures provider, method, tools, and computes fingerprint for AgentTest"""

        user_id, session_id = self._get_session_context(kwargs, str(run_id), str(parent_run_id) if parent_run_id else None, metadata)
        self._context_map[str(run_id)] = (user_id, session_id)

        # Extract model (existing)
        model = self._extract_model(serialized, kwargs)

        # NEW: Extract provider and method
        provider = self._extract_provider(serialized)
        method = self._extract_method(provider)

        # Flatten messages (existing)
        flat_messages = []
        for batch in messages:
            for msg in batch:
                flat_messages.append({
                    "role": getattr(msg, "type", "unknown"),
                    "content": str(msg.content)
                })

        # NEW: Extract tools
        tools = self._extract_tools(kwargs)

        # NEW: Build request params
        request_params = self._build_request_params(
            model, flat_messages, tools, kwargs
        )

        # NEW: Compute fingerprint
        fingerprint = compute_fingerprint(provider, method, request_params) if AGENTTEST_AVAILABLE else ""

        # Store in run context (ENHANCED with new fields)
        self._runs[run_id] = {
            "model": model,
            "start_time": time.time(),
            "messages": flat_messages,
            "chunks": [],
            # NEW fields for AgentTest
            "provider": provider,
            "method": method,
            "request_params": request_params,
            "fingerprint": fingerprint,
            "tools": tools,
        }

        if self.eventbus:
            self.eventbus.publish(EventType.LLM_CALL_START, Event(
                type=EventType.LLM_CALL_START,
                user_id=user_id,
                session_id=session_id,
                run_id=str(run_id),
                model=model,
                messages=flat_messages,
                timestamp=time.time()
            ))


        
    def on_llm_end(self, response, *, run_id: str, **kwargs):
        """ENHANCED: Now captures tool_calls and full response_data for AgentTest"""

        user_id, session_id = self._context_map.get(str(run_id), ("default", "default"))
        run = self._runs.get(run_id)

        # Handle case where run_id not found (shouldn't happen, but be defensive)
        if not run:
            return

        duration = int((time.time() - run.get("start_time")) * 1000)
        run["duration_ms"] = duration

        # Extract text and usage (existing)
        text = None
        usage = None
        if response.generations and response.generations[0]:
            gen = response.generations[0][0]
            if hasattr(gen, "message"):
                text = str(gen.message.content) if gen.message.content else None
                meta = getattr(gen.message, "response_metadata", {}) or {}
                usage = meta.get("usage", {})
                if usage:
                    usage = {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens")
                    }
            elif hasattr(gen, "text"):
                text = str(gen.text) if gen.text else None

        # NEW: Extract tool calls
        tool_calls = self._extract_tool_calls(response)

        # NEW: Build response data
        response_data = self._build_response_data(text, tool_calls, usage, response)

        # Publish LLM_CALL_END with ENHANCED metadata
        if self.eventbus:
            self.eventbus.publish(EventType.LLM_CALL_END, Event(
                type=EventType.LLM_CALL_END,
                user_id=user_id,
                session_id=session_id,
                run_id=str(run_id),
                model=run.get("model"),
                text=text,
                usage=usage,
                duration_ms=duration,
                timestamp=time.time(),
                # NEW: AgentTest metadata
                metadata={
                    "provider": run.get("provider", "unknown"),
                    "method": run.get("method", "unknown"),
                    "fingerprint": run.get("fingerprint", ""),
                    "request_params": run.get("request_params", {}),
                    "response_data": response_data,
                    "is_streaming": False,
                    "stream_id": None,
                }
            ))

        # Clean up to prevent memory leak
        self._runs.pop(run_id, None)

    
    def on_llm_error(self, error: Exception, *, run_id: str,**kwargs):
        run = self._runs.get(run_id, {})

        if self.eventbus:
            self.eventbus.publish(EventType.LLM_ERROR, Event(
                type=EventType.LLM_ERROR,
                run_id=str(run_id),
                model=run.get("model", "unknown"),
                error=str(error),
                timestamp=time.time()
            ))

        # Clean up to prevent memory leak
        self._runs.pop(run_id, None)


    def on_tool_start( self, serialized: Dict[str, Any], input_str: str, *, run_id: str, inputs: Optional[Dict] = None, **kwargs):
        name = (serialized or {}).get("name", "unknown")
        args = inputs if inputs else {"input": input_str}
        
        self._tool_runs[run_id] = {
            "name": name,
            "args": args,
            "start_time": time.time()
        }
        
        # Publish event
        if self.eventbus:
            self.eventbus.publish(EventType.TOOL_CALL_START, Event(
                type=EventType.TOOL_CALL_START,
                run_id=str(run_id),
                tool_name=name,
                tool_args=args,
                timestamp=time.time()
            ))
    
    def on_tool_end( self, output: str, *, run_id: str, **kwargs):
        run = self._tool_runs.pop(run_id, {})
        duration_ms = int((time.time() - run.get("start_time", time.time())) * 1000)
        
        # Publish event
        if self.eventbus:
            user_id, session_id = self._context_map.get(str(run_id), ("default", "default"))
            self.eventbus.publish(EventType.TOOL_CALL_END, Event(
                type=EventType.TOOL_CALL_END,
                user_id=user_id,
                session_id=session_id,
                run_id=str(run_id),
                tool_name=run.get("name", "unknown"),
                content=str(output),
                duration_ms=duration_ms,
                timestamp=time.time()
            ))

    def on_tool_error(self, error: Exception, *, run_id: str, **kwargs):
        run = self._tool_runs.pop(run_id, {})

        if self.eventbus:
            self.eventbus.publish(EventType.TOOL_ERROR, Event(
                type=EventType.TOOL_ERROR,
                run_id=str(run_id),
                tool_name=run.get("name", "unknown"),
                error=str(error),
                timestamp=time.time()
            ))

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: str, **kwargs):
        """Called when a chain completes - publishes chain_end event."""

        # Don't restrict to only LLM runs - chains can complete without being in self._runs
        # This happens in LangGraph where chains orchestrate multiple steps

        if self.eventbus:
            # Handle different output formats
            event_data = {
                "run_id": str(run_id),
                "timestamp": time.time()
            }

            # If outputs contains messages, include them
            if isinstance(outputs, dict) and "messages" in outputs:
                messages = outputs["messages"]
                if isinstance(messages, list):
                    # Convert messages to serializable format
                    event_data["messages"] = [
                        {
                            "type": getattr(msg, "type", "unknown"),
                            "content": str(getattr(msg, "content", ""))
                        } for msg in messages
                    ]
            else:
                # Include raw outputs for other chain types
                event_data["outputs"] = str(outputs)

            user_id, session_id = self._context_map.get(str(run_id), ("default", "default"))
            self.eventbus.publish(EventType.AGENT_TURN_END, Event(
                type=EventType.AGENT_TURN_END,
                user_id=user_id,
                session_id=session_id,
                run_id=str(run_id),
                outputs=event_data,
                timestamp=time.time()
            ))

        # Clean up context map to prevent memory leak
        # Remove this run_id from context map after chain completes
        self._context_map.pop(str(run_id), None)
    


    

        
    