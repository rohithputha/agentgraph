from typing import Dict, Any, List
from langchain_core.messages import AIMessage


class ResponseBuilder:
    def build_message(self, response_data: Dict[str, Any]) -> AIMessage:
        content = response_data.get("content", "")
        tool_calls_data = response_data.get("tool_calls", [])
        usage = response_data.get("usage", {})
        response_metadata = response_data.get("response_metadata", {})

        tool_calls = []
        if tool_calls_data:
            tool_calls = self._build_tool_calls(tool_calls_data)

        usage_metadata = None
        if usage:
            usage_metadata = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }

        ai_message = AIMessage(
            content=content,
            additional_kwargs={
                "tool_calls": tool_calls if tool_calls else None,
            },
            response_metadata=response_metadata,
            usage_metadata=usage_metadata
        )

        return ai_message

    def _build_tool_calls(self, tool_calls_data: List[Dict]) -> List[Dict]:
        tool_calls = []

        for tc in tool_calls_data:
            tool_calls.append({
                "id": tc.get("id", ""),
                "type": tc.get("type", "function"),
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": tc.get("args", {})
                }
            })

        return tool_calls
