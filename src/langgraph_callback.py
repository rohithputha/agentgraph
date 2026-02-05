
import time
from typing import Dict, Any, Optional, List
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from eventbus import eventbus

class langgraph_callback(BaseCallbackHandler):
    def __init__(self, eventbus):
        super().__init__()
        self.eventbus = eventbus
        self._runs = {}
        self._tool_runs = {}
    

    def _extract_model(self, serialized: Dict, kwargs: Dict) -> str:
        inv = kwargs.get("invocation_params", {})
        return (
            inv.get("model_name")
            or inv.get("model")
            or (serialized or {}).get("name", "unknown")
        )

    def _track_message(self, message: BaseMessage):
        self.tracer.track_message(
            role=getattr(message, "type", "unknown"),
            content=str(message.content),
            metadata={
                "id": getattr(message, "id", None),
                "name": getattr(message, "name", None),
            }
        )
    
    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]], *, run_id: UUID, parent_run_id: UUID | None = None, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, **kwargs: Any):
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
            self.eventbus.publish("llm_start", {
                "run_id": str(run_id),
                "model": model,
                "messages": flat_messages,
                "timestamp": time.time()
            })


        
    def on_llm_end(self, response, *, run_id: str, **kwargs):
        run = self._runs.get(run_id)
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
                self.eventbus.publish("llm_end", {
                    "run_id": str(run_id),
                    "text": text,
                    "usage": usage,
                    "duration_ms": duration,
                    "timestamp": time.time()
                })

    
    def on_llm_error(self, error: Exception, *, run_id: str,**kwargs):
        run = self._runs.get(run_id)
        
        if self.eventbus:
            self.eventbus.publish("llm_error", {
                "run_id": str(run_id),
                "model": run.get("model", "unknown"),
                "error": str(error),
                "timestamp": time.time()
            })


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
            self.eventbus.publish("tool_start", {
                "run_id": str(run_id),
                "tool_name": name,
                "tool_args": args,
                "timestamp": time.time()
            })
    
    def on_tool_end( self, output: str, *, run_id: str, **kwargs):
        run = self._tool_runs.pop(run_id, {})
        duration_ms = int((time.time() - run.get("start_time", time.time())) * 1000)
        
        # Publish event
        if self.eventbus:
            self.eventbus.publish("tool_end", {
                "run_id": str(run_id),
                "tool_name": run.get("name", "unknown"),
                "content": str(output),
                "duration_ms": duration_ms,
                "timestamp": time.time()
            })

    def on_tool_error(self, error: Exception, *, run_id: str, **kwargs):
        run = self._tool_runs.pop(run_id, {})
        
        if self.eventbus:
            self.eventbus.publish("tool_error", {
                "run_id": str(run_id),
                "tool_name": run.get("name", "unknown"),
                "error": str(error),
                "timestamp": time.time()
            })

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
            
            self.eventbus.publish("chain_end", event_data)
    


    

        
    