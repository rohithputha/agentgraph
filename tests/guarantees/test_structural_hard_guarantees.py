from agenttest.comparator import Comparison

from tests.helpers import make_detail


def test_g11_addition_of_llm_node_is_detected():
    baseline = [
        make_detail(recording_id="b", step_index=0, node_id=1, roles=("human",), response_text="a"),
        make_detail(recording_id="b", step_index=1, node_id=2, roles=("human", "ai"), response_text="b"),
    ]
    replay = [
        make_detail(recording_id="r", step_index=0, node_id=11, roles=("human",), response_text="a"),
        make_detail(recording_id="r", step_index=1, node_id=12, roles=("system", "human"), response_text="new"),
        make_detail(recording_id="r", step_index=2, node_id=13, roles=("human", "ai"), response_text="b"),
    ]

    result = Comparison().compare_recordings(baseline, replay)

    assert result.added_steps == 1
    assert result.overall_pass is False
    assert result.root_cause_index == 1


def test_g12_removal_of_llm_node_is_detected():
    baseline = [
        make_detail(recording_id="b", step_index=0, node_id=1, roles=("human",), response_text="a"),
        make_detail(recording_id="b", step_index=1, node_id=2, roles=("system", "human"), response_text="x"),
        make_detail(recording_id="b", step_index=2, node_id=3, roles=("human", "ai"), response_text="b"),
    ]
    replay = [
        make_detail(recording_id="r", step_index=0, node_id=11, roles=("human",), response_text="a"),
        make_detail(recording_id="r", step_index=1, node_id=12, roles=("human", "ai"), response_text="b"),
    ]

    result = Comparison().compare_recordings(baseline, replay)

    assert result.removed_steps == 1
    assert result.overall_pass is False
    assert result.root_cause_index == 1


def test_g13_execution_order_change_is_detected():
    baseline = [
        make_detail(recording_id="b", step_index=0, node_id=1, roles=("human",), response_text="a"),
        make_detail(recording_id="b", step_index=1, node_id=2, roles=("system", "human"), response_text="b"),
        make_detail(recording_id="b", step_index=2, node_id=3, roles=("human", "ai"), response_text="c"),
    ]
    replay = [
        make_detail(recording_id="r", step_index=0, node_id=11, roles=("system", "human"), response_text="b"),
        make_detail(recording_id="r", step_index=1, node_id=12, roles=("human",), response_text="a"),
        make_detail(recording_id="r", step_index=2, node_id=13, roles=("human", "ai"), response_text="c"),
    ]

    result = Comparison().compare_recordings(baseline, replay)

    assert result.overall_pass is False
    assert result.added_steps >= 1
    assert result.removed_steps >= 1


def test_g14_routing_change_is_detected():
    baseline = [
        make_detail(recording_id="b", step_index=0, node_id=1, roles=("human",), response_text="a"),
        make_detail(recording_id="b", step_index=1, node_id=2, roles=("system", "human"), response_text="route-a"),
        make_detail(recording_id="b", step_index=2, node_id=3, roles=("human", "ai"), response_text="end"),
    ]
    replay = [
        make_detail(recording_id="r", step_index=0, node_id=11, roles=("human",), response_text="a"),
        make_detail(recording_id="r", step_index=1, node_id=12, roles=("tool", "human"), response_text="route-b"),
        make_detail(recording_id="r", step_index=2, node_id=13, roles=("human", "ai"), response_text="end"),
    ]

    result = Comparison().compare_recordings(baseline, replay)

    assert result.overall_pass is False
    assert result.root_cause_index == 1


def test_g15_tool_set_change_is_detected():
    baseline = [
        make_detail(
            recording_id="b",
            step_index=0,
            node_id=1,
            roles=("human",),
            tool_names=("get_weather",),
            response_text="weather",
        )
    ]
    replay = [
        make_detail(
            recording_id="r",
            step_index=0,
            node_id=10,
            roles=("human",),
            tool_names=("get_news",),
            response_text="weather",
        )
    ]

    result = Comparison().compare_recordings(baseline, replay)

    assert result.overall_pass is False
    assert result.added_steps == 1
    assert result.removed_steps == 1
