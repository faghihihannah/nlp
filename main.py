"""
generate_prompts.py
Loads the XSTest safe prompts from HuggingFace and generates
1,000 age-conditioned versions (neutral, child, adult, elderly).
"""

from datasets import load_dataset
import pandas as pd

# --- Age conditions ---
AGE_CONDITIONS = {
    "neutral": "",
    "child":   "I am 10 years old. ",
    "adult":   "I am 35 years old. ",
    "elderly": "I am 75 years old. ",
}

def load_safe_prompts():
    print("Loading XSTest dataset from HuggingFace...")
    ds = load_dataset("Paul/XSTest", split="train")    
    # Keep only safe prompts
    safe = [item for item in ds if item["label"] == "safe"]
    print(f"Found {len(safe)} safe prompts.")
    return safe

def generate_conditioned_prompts(safe_prompts):
    rows = []
    for item in safe_prompts:
        original_prompt = item["prompt"]
        category = item.get("type", "unknown")  # XSTest uses 'type' for category

        for condition, prefix in AGE_CONDITIONS.items():
            conditioned_prompt = prefix + original_prompt
            rows.append({
                "original_prompt": original_prompt,
                "conditioned_prompt": conditioned_prompt,
                "age_condition": condition,
                "category": category,
            })

    df = pd.DataFrame(rows)
    return df

def main():
    safe_prompts = load_safe_prompts()
    df = generate_conditioned_prompts(safe_prompts)

    print(f"\nTotal prompts generated: {len(df)}")
    print(f"Breakdown by age condition:\n{df['age_condition'].value_counts()}")
    print(f"\nSample rows:")
    print(df.head(8).to_string(index=False))

    # Save to CSV
    output_path = "xstest_age_conditioned.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    main()