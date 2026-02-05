# AgentGraph

LangGraph callback framework for tracking agent execution events in real-time.

## Features

- 🔔 **Event Bus System** - Pub/sub architecture for decoupled event handling
- 📊 **LangGraph Integration** - Seamless callbacks for LangGraph agents
- ⚡ **Real-time Events** - Track LLM and tool completions as they happen
- 🎓 **Educational Examples** - Complete examples for teaching new developers
- 🤖 **Gemini 2.0 Compatible** - Works with Google's latest models

## Quick Start

```bash
# Install dependencies
pip install -r examples/requirements.txt

# Set your API key
export GOOGLE_API_KEY="your-api-key"

# Run the example
python examples/langgraph_example.py
```

## What Gets Tracked

The callback system captures:

| Event | Fires When | Use Case |
|-------|------------|----------|
| `llm_start` | LLM call begins | Track when agent starts thinking |
| `llm_end` | **Message completes** | Get complete responses in real-time |
| `tool_start` | Tool execution begins | Monitor tool invocations |
| `tool_end` | **Tool completes** | Get tool results immediately |
| `llm_error` | LLM fails | Handle errors gracefully |
| `tool_error` | Tool fails | Debug tool failures |

> **Note**: Events fire in **real-time** after each step completes, not token-by-token.

## Project Structure

```
agentgraph/
├── src/
│   ├── eventbus.py              # Pub/sub event system
│   ├── langgraph_callback.py    # LangChain callback handler
│   └── core.py
├── examples/
│   ├── README.md                      # Comprehensive guide
│   ├── QUICKSTART.md                  # Fast setup
│   ├── langgraph_example.py          # Main teaching example
│   ├── simple_test.py                # Quick verification
│   ├── advanced_storage.py           # Event persistence
│   └── test_streaming_callbacks.py   # Streaming tests
└── requirements.txt
```

## Examples

See the [`examples/`](examples/) directory for:

1. **`langgraph_example.py`** - Full agent with tools and callbacks
2. **`simple_test.py`** - Verify your setup works
3. **`advanced_storage.py`** - Save events to JSON
4. **`test_streaming_callbacks.py`** - Test event firing

Each example is heavily commented and ready to use for teaching.

## Usage

### Basic Setup

```python
from eventbus import eventbus
from langgraph_callback import langgraph_callback
from langchain_google_genai import ChatGoogleGenerativeAI

# Create event bus
bus = eventbus()

# Subscribe to events
def on_message_complete(event):
    print(f"Message: {event['text']}")

bus.subscribe("llm_end", on_message_complete)

# Create callback handler
callback = langgraph_callback(bus)

# Attach to LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    callbacks=[callback]
)
```

### In LangGraph Agents

```python
from langgraph.graph import StateGraph

# Your callback handler tracks all LLM/tool calls automatically
graph = StateGraph(AgentState)
graph.add_node("agent", call_agent)
graph = graph.compile()

# Events fire in real-time during execution!
result = graph.invoke({"messages": [...]})
```

## Documentation

- **[Full Documentation](examples/README.md)** - Complete guide with examples
- **[Quick Start](examples/QUICKSTART.md)** - Get running in 2 minutes
- **[Chain End Investigation](examples/CHAIN_END_INVESTIGATION.md)** - Event behavior details

## Requirements

- Python 3.8+
- LangChain Core
- LangGraph
- Google Generative AI

See [`examples/requirements.txt`](examples/requirements.txt) for full dependencies.

## License

MIT

## Contributing

Contributions welcome! This framework is designed to be educational and extensible.
