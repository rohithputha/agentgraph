from agenttest.fingerprint import compute_fingerprint


def test_fingerprint_ignores_message_content_when_structure_is_same():
    params_a = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "human", "content": "What is 2+2?"},
        ],
        "tools": [{"type": "function", "function": {"name": "calculator"}}],
    }
    params_b = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "Behave concisely"},
            {"role": "human", "content": "What is 200+200?"},
        ],
        "tools": [{"type": "function", "function": {"name": "calculator"}}],
    }

    fp_a = compute_fingerprint("openai", "chat.completions.create", params_a)
    fp_b = compute_fingerprint("openai", "chat.completions.create", params_b)

    assert fp_a == fp_b


def test_fingerprint_changes_on_role_sequence_change():
    baseline = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "A"},
            {"role": "human", "content": "B"},
        ],
        "tools": [],
    }
    replay = {
        "model": "gpt-4",
        "messages": [
            {"role": "human", "content": "B"},
            {"role": "system", "content": "A"},
        ],
        "tools": [],
    }

    fp_a = compute_fingerprint("openai", "chat.completions.create", baseline)
    fp_b = compute_fingerprint("openai", "chat.completions.create", replay)

    assert fp_a != fp_b


def test_fingerprint_changes_on_tool_set_change():
    baseline = {
        "model": "gpt-4",
        "messages": [{"role": "human", "content": "check"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }
    replay = {
        "model": "gpt-4",
        "messages": [{"role": "human", "content": "check"}],
        "tools": [{"type": "function", "function": {"name": "get_news"}}],
    }

    fp_a = compute_fingerprint("openai", "chat.completions.create", baseline)
    fp_b = compute_fingerprint("openai", "chat.completions.create", replay)

    assert fp_a != fp_b


def test_fingerprint_changes_on_tool_order_change():
    baseline = {
        "model": "gpt-4",
        "messages": [{"role": "human", "content": "tools"}],
        "tools": [
            {"type": "function", "function": {"name": "tool_a"}},
            {"type": "function", "function": {"name": "tool_b"}},
        ],
    }
    replay = {
        "model": "gpt-4",
        "messages": [{"role": "human", "content": "tools"}],
        "tools": [
            {"type": "function", "function": {"name": "tool_b"}},
            {"type": "function", "function": {"name": "tool_a"}},
        ],
    }

    fp_a = compute_fingerprint("openai", "chat.completions.create", baseline)
    fp_b = compute_fingerprint("openai", "chat.completions.create", replay)

    assert fp_a != fp_b


def test_fingerprint_changes_on_model_change():
    params_a = {
        "model": "gpt-4",
        "messages": [{"role": "human", "content": "hello"}],
        "tools": [],
    }
    params_b = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "human", "content": "hello"}],
        "tools": [],
    }

    fp_a = compute_fingerprint("openai", "chat.completions.create", params_a)
    fp_b = compute_fingerprint("openai", "chat.completions.create", params_b)

    assert fp_a != fp_b
