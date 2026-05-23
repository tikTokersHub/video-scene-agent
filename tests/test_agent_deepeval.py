import pytest

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from video_agent.agent import ask


@pytest.mark.parametrize(
    "question, expected_classification",
    [
        ("Was anyone riding a bicycle in the video?", "anomalous"),
        ("What happened around 7 seconds into the video?", "anomalous"),
        ("Were two people sitting near the pathway at the start of the video?", "normal"),
    ],
)
def test_agent_answer_quality_with_deepeval(question, expected_classification):
    answer = ask(question)

    retrieval_context = [e.caption for e in answer.evidence]

    test_case = LLMTestCase(
        input=question,
        actual_output=answer.answer,
        retrieval_context=retrieval_context,
    )

    assert_test(
        test_case,
        [
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
        ],
    )

    assert answer.classification == expected_classification