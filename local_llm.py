


import ollama
import pandas as pd
from pathlib import Path

# Step 1: Set up file paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

input_file = DATASET_DIR / "prompts.csv"
output_file = DATASET_DIR / "llm_responses.csv"
history_file = DATASET_DIR / "conversation_history.txt"

# Step 2: Read the 10 prompts
df = pd.read_csv(input_file)

# Step 3: Store AI responses
results = []

print("Starting Local LLM Inference...")
print("Model: Gemma 3 (1B)")

# Step 4: Generate responses
for _, row in df.iterrows():
    prompt_id = row["Prompt_ID"]
    domain = row["Domain"]
    complexity = row["Complexity"]
    prompt = row["Prompt"]

    print(f"\nProcessing Prompt {prompt_id}...")

    try:
        response = ollama.chat(
            model="gemma3:1b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response["message"]["content"]

    except Exception as e:
        print(f"Error: {e}")
        answer = f"ERROR: {e}"

    results.append({
        "Prompt_ID": prompt_id,
        "Domain": domain,
        "Complexity": complexity,
        "Prompt": prompt,
        "Model": "gemma3:1b",
        "Response": answer,
        "Response_Length": len(answer.split())
    })

    print("Response generated!")

# Step 5: Save responses to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

# Step 6: Save conversation history
with open(history_file, "w", encoding="utf-8") as file:
    for result in results:
        file.write(f"Prompt {result['Prompt_ID']}\n")
        file.write(f"Domain: {result['Domain']}\n")
        file.write(f"Question: {result['Prompt']}\n")
        file.write(f"Answer: {result['Response']}\n")
        file.write("-" * 60 + "\n")

print("\nAll prompts processed!")
print("Responses saved to:", output_file)
print("Conversation history saved to:", history_file)

# Step 7: Display results
print("\nResponse Summary:")
print(results_df[[
    "Prompt_ID", "Domain", "Model", "Response_Length"
]].to_string(index=False))
