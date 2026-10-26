
import ollama
import pandas as pd
from pathlib import Path

# Set up file paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

prompts_file = DATASET_DIR / "prompts.csv"
gemma_file = DATASET_DIR / "llm_responses.csv"
comparison_file = DATASET_DIR / "response_comparison.csv"

# Read prompts and previous Gemma responses
prompts_df = pd.read_csv(prompts_file)
gemma_df = pd.read_csv(gemma_file)

comparison_results = []

print("Starting LLM Model Comparison...")
print("Gemma 3 vs Llama 3.2")

for _, row in prompts_df.iterrows():

    prompt_id = row["Prompt_ID"]
    prompt = row["Prompt"]

    print(f"\nComparing Prompt {prompt_id}/10...")

    # Retrieve previously generated Gemma response
    gemma_match = gemma_df[
        gemma_df["Prompt_ID"] == prompt_id
    ]

    if gemma_match.empty:
        print("Gemma response not found. Skipping.")
        continue

    gemma_response = str(
        gemma_match.iloc[0]["Response"]
    )

    # Generate Llama response
    try:
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        llama_response = response["message"]["content"]

    except Exception as e:
        print("Error:", e)
        llama_response = f"ERROR: {e}"

    # Calculate response lengths
    gemma_length = len(gemma_response.split())
    llama_length = len(llama_response.split())

    # Store comparison
    comparison_results.append({
        "Prompt_ID": prompt_id,
        "Domain": row["Domain"],
        "Complexity": row["Complexity"],
        "Prompt": prompt,
        "Gemma_Response": gemma_response,
        "Llama_Response": llama_response,
        "Gemma_Word_Count": gemma_length,
        "Llama_Word_Count": llama_length,
        "Longer_Response": (
            "Gemma" if gemma_length > llama_length
            else "Llama" if llama_length > gemma_length
            else "Equal"
        )
    })

    print("Comparison completed!")

# Save results
comparison_df = pd.DataFrame(comparison_results)

comparison_df.to_csv(
    comparison_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nAll comparisons completed!")
print("File saved:", comparison_file)

print("\nComparison Summary:")
print(comparison_df[[
    "Prompt_ID",
    "Domain",
    "Gemma_Word_Count",
    "Llama_Word_Count",
    "Longer_Response"
]].to_string(index=False))
