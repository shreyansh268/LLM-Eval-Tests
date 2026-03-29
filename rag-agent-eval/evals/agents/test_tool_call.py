#Deepeval+Pytest

import pytest
from deepeval import assert_test
from deepeval.test_case import  LLMTestCase, ToolCallTestCase
from deepeval.metrics import ToolCorrectnessMetric, TaskCompletionMetric

@pytest.mark.parametrize("test_case", [
    LLMTestCase(
        input="",
        actual_output = agent.run(input),
        tools_called=[
            ToolCall(name="lookup_order", input_params={"order_id": "4821"}),
            ToolCall(name="send_email", input_params={"to": "a@gmail.com"})
        ],
        expected_tools=[
            ToolCall(name="lookup_order", input_parameters={"order_id": "4821"}),
            ToolCall(name="send_email",   input_parameters={"to": "customer@email.com"})
        ])
    ])

def test_tool_calls_sequence(test_case):
    metric=ToolCorrectnessMetric(threshold=0.9)
    assert_test(test_case, metric)