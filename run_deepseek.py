"""
SCRIPT 2 (FIXED): Run DeepSeek-R1 locally via Ollama
=====================================================
Uses the ollama Python library with think=True to properly
capture the reasoning content.

Install:
    pip install ollama

Usage:
    python step2_run_deepseek_local.py

No API key needed. No rate limits. Runs all 1000 prompts in one go.
"""

import csv
import os
import time
import ollama

# --- Config ---
MODEL  = "deepseek-r1:8b"
INPUT  = "xstest_age_conditioned.csv"
OUTPUT = "results.csv"
SLEEP  = 0.5


def call_deepseek(prompt: str) -> dict:
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=True,   # key flag that captures thinking content separately
            options={"temperature": 0},
        )
        think         = response.message.thinking or ""
        response_text = response.message.content or ""
        return {"think": think, "response": response_text, "error": ""}

    except Exception as e:
        return {"think": "", "response": "", "error": str(e)}


def get_processed_indices(output_file: str) -> set:
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                processed.add(int(row["row_index"]))
    return processed


def main():
    # Load all prompts
    with open(INPUT, "r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    print(f"Total prompts to run: {len(all_rows)}")

    # Check which are already done (allows resuming if interrupted)
    processed   = get_processed_indices(OUTPUT)
    remaining   = len(all_rows) - len(processed)
    print(f"Already done: {len(processed)} | Remaining: {remaining}")

    file_exists = os.path.exists(OUTPUT)
    fieldnames  = ["row_index", "age_condition", "category",
                   "original_prompt", "conditioned_prompt",
                   "think_content", "final_response", "error"]

    with open(OUTPUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, row in enumerate(all_rows):
            if i in processed:
                continue

            print(f"[{i+1}/{len(all_rows)}] {row['age_condition']:8s} | {row['category']}")

            result = call_deepseek(row["conditioned_prompt"])

            writer.writerow({
                "row_index":          i,
                "age_condition":      row["age_condition"],
                "category":           row["category"],
                "original_prompt":    row["original_prompt"],
                "conditioned_prompt": row["conditioned_prompt"],
                "think_content":      result["think"],
                "final_response":     result["response"],
                "error":              result["error"],
            })
            f.flush()  # save after every row so nothing is lost if interrupted

            if result["error"]:
                print(f"  ERROR: {result['error']}")
            else:
                think_len    = len(result["think"])
                response_len = len(result["response"])
                print(f"  think={think_len} chars | response={response_len} chars")

            time.sleep(SLEEP)

    print(f"\nAll done! Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()