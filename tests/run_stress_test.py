import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.rag import answer_question

TEST_FILE = Path("tests/stress_questions.json")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

REFUSAL_TEXT = "I don't have enough information in the knowledge base to answer that."

def evaluate_one(case: dict, result: dict) -> dict:
    answer = result["answer"]
    source_names = {s["source"] for s in result["sources"]}

    expected_behavior = case["expected_behavior"]
    expected_sources = set(case.get("expected_sources", []))

    if expected_behavior == "refuse":
        behavior_pass = REFUSAL_TEXT.lower() in answer.lower()
    elif expected_behavior == "answer":
        behavior_pass = REFUSAL_TEXT.lower() not in answer.lower()
    else:
        # Ambiguous questions should either ask for clarification or give a qualified answer.
        lowered = answer.lower()
        behavior_pass = (
            "clarif" in lowered
            or "do you mean" in lowered
            or "if you mean" in lowered
            or "depends" in lowered
            or "don't have enough information" in lowered
        )

    source_pass = (
        True if not expected_sources
        else expected_sources.issubset(source_names)
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_behavior": expected_behavior,
        "answer": answer,
        "retrieval_sufficient": result["retrieval_sufficient"],
        "max_relevance_score": round(result["max_relevance_score"], 4),
        "retrieved_sources": ", ".join(sorted(source_names)),
        "expected_sources": ", ".join(sorted(expected_sources)),
        "behavior_pass": behavior_pass,
        "source_pass": source_pass,
        "overall_pass": behavior_pass and source_pass,
    }

def main():
    cases = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    rows = []

    for case in cases:
        print(f"\n[{case['id']:02d}] {case['category']}: {case['question']}")
        result = answer_question(case["question"]).to_dict()
        row = evaluate_one(case, result)
        rows.append(row)
        print(row["answer"])
        print(
            f"PASS={row['overall_pass']} | "
            f"score={row['max_relevance_score']} | "
            f"sources={row['retrieved_sources']}"
        )

    passed = sum(1 for r in rows if r["overall_pass"])
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "total": len(rows),
        "pass_rate": passed / len(rows) if rows else 0,
        "results": rows,
    }

    json_path = REPORT_DIR / "stress_test_report.json"
    csv_path = REPORT_DIR / "stress_test_report.csv"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPassed {passed}/{len(rows)} ({summary['pass_rate']:.1%})")
    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")

if __name__ == "__main__":
    main()
