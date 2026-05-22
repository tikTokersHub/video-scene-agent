import json
from collections import defaultdict
from pathlib import Path

from video_agent.agent import ask

def load_jsonl(path:Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"File not found on {path}")
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return items

def frame_recall(
        expected_frames: list[int],
        predicted_frames: list[int],
        tolerance: int = 5,
) -> float:
    """
    Compute evidence recall with frame tolerance.

    Example:
    expected frame = 170
    predicted frame = 175
    tolerance = 5

    This counts as a match because abs(170 - 175) <= 5.
    """
    if not expected_frames:
        return 1.0 if not predicted_frames else 0.0
    
    matched = 0

    for expected in expected_frames:
        found_match = any(
             abs(expected - predicted) <= tolerance
             for predicted in predicted_frames
        )

        if found_match:
            matched += 1
    
    return matched/len(expected_frames)


def evaluate_all(
    test_set_path: str = "data/eval/qa_test_set.jsonl",
    output_path: str = "reports/eval_predictions.json",
    tolerance: int = 5,
) -> dict:
    """
    Run all QA test questions through the agent and compute:

    1. Overall classification accuracy
    2. Evidence recall
    3. Category-level classification accuracy
    4. Per-question prediction records
    """

    test_set = load_jsonl(Path(test_set_path))

    rows = []
    
    for item in test_set:
        question = item["question"]
        expected_classification = item["expected_classification"]
        expected_frames = item.get("evidence_frames", [])
        category = item.get("category", "unknown")

        print(f"\nQuestion: {question}")

        answer = ask(question)
        predicted_classification = answer.classification
        predicted_frames = [e.frame_idx for e in answer.evidence]

        classification_correct = (
            predicted_classification == expected_classification
        )
        recall = frame_recall(
            expected_frames=expected_frames,
            predicted_frames=predicted_frames,
            tolerance=tolerance,
        )

        row = {
            "question": question,
            "category": category,
            "expected_classification": expected_classification,
            "predicted_classification": predicted_classification,
            "classification_correct": classification_correct,
            "expected_frames": expected_frames,
            "predicted_frames": predicted_frames,
            "evidence_recall": recall,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "rules_consulted": answer.rules_consulted,
        }


        rows.append(row)

        print(f"Expected class: {expected_classification}")
        print(f"Predicted class: {predicted_classification}")
        print(f"Correct: {classification_correct}")
        print(f"Expected frames: {expected_frames}")
        print(f"Predicted frames: {predicted_frames}")
        print(f"Evidence recall: {recall:.2f}")

    total = len(rows)
    
    classification_accuracy = sum(
        row["classification_correct"] for row in rows
    ) / total

    mean_evidence_recall = sum(
        row["evidence_recall"] for row in rows
    ) / total

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in rows:
        category = row["category"]
        category_stats[category]["total"] += 1

        if row["classification_correct"]:
            category_stats[category]["correct"] += 1

    category_accuracy = {
        category: values["correct"] / values["total"]
        for category, values in category_stats.items()
    }

    metrics = {
        "n_questions": total,
        "classification_accuracy": classification_accuracy,
        "mean_evidence_recall": mean_evidence_recall,
        "category_accuracy": category_accuracy,
        "frame_tolerance": tolerance,
    }

    output = {
        "metrics": metrics,
        "rows": rows,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(f"Questions: {total}")
    print(f"Classification accuracy: {classification_accuracy:.2%}")
    print(f"Mean evidence recall: {mean_evidence_recall:.2%}")

    print("\nCategory accuracy:")
    for category, acc in category_accuracy.items():
        print(f"- {category}: {acc:.2%}")

    print(f"\nSaved predictions to: {output_file}")

    return output


if __name__ == "__main__":
    evaluate_all()