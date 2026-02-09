from uuid import UUID
import time
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.callbacks import BaseCallbackHandler

from .eventbus import Eventbus
from .event import Event, EventType
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
    
    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]], *, run_id: UUID, parent_run_id: UUID | None = None, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, **kwargs: Any):
        user_id, session_id = self._get_session_context(kwargs, str(run_id), str(parent_run_id) if parent_run_id else None, metadata)
        self._context_map[str(run_id)] = (user_id, session_id)
        model = self._extract_model(serialized, kwargs)
        
        flat_messages = []
        for batch in messages:
            for msg in batch:
                flat_messages.append({
                    "role": getattr(msg, "type", "unknown"),
                    "content": str(msg.content)
                })
        
        self._runs[run_id] = {
            "model": model,
            "start_time": time.time(),
            "messages": flat_messages,
            "chunks" : []
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
        user_id, session_id = self._context_map.get(str(run_id), ("default", "default"))
        run = self._runs.get(run_id)

        # Handle case where run_id not found (shouldn't happen, but be defensive)
        if not run:
            return

        duration = int((time.time() - run.get("start_time")) * 1000)
        run["duration_ms"] = duration

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

            if self.eventbus:
                self.eventbus.publish(EventType.LLM_CALL_END, Event(
                    type=EventType.LLM_CALL_END,
                    user_id=user_id,
                    session_id=session_id,
                    run_id=str(run_id),
                    text=text,
                    usage=usage,
                    duration_ms=duration,
                    timestamp=time.time()
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
    


    

        
    