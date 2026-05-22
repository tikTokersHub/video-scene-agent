import pytest

from video_agent.agent import ask


GOLDEN_QUESTIONS = [
    {
        "question": "Was anyone riding a bicycle in the video?",
        "expected_classification": "anomalous",
        "must_include_frames": [170, 175, 200, 205],
    },
    {
        "question": "What happened around 7 seconds into the video?",
        "expected_classification": "anomalous",
        "must_include_frames": [170, 175],
    },
    {
        "question": "Were two people sitting near the pathway at the start of the video?",
        "expected_classification": "normal",
        "must_include_frames": [0, 5, 10, 15, 20],
    },
    {
        "question": "Was anyone fighting or pushing another person?",
        "expected_classification": "normal",
        "must_include_frames": [],
    },
]


def has_matching_frame(
    predicted_frames: list[int],
    expected_frames: list[int],
    tolerance: int = 5,
) -> bool:
    """
    Return True if at least one expected frame is matched by prediction.

    We use at least one match for CI because the agent may retrieve only
    a subset of valid evidence frames.
    """
    if not expected_frames:
        return True

    for expected in expected_frames:
        for predicted in predicted_frames:
            if abs(expected - predicted) <= tolerance:
                return True

    return False


@pytest.mark.parametrize("item", GOLDEN_QUESTIONS)
def test_golden_question_classification_and_evidence(item):
    answer = ask(item["question"])

    predicted_frames = [e.frame_idx for e in answer.evidence]

    assert answer.classification == item["expected_classification"]

    assert has_matching_frame(
        predicted_frames=predicted_frames,
        expected_frames=item["must_include_frames"],
        tolerance=5,
    )