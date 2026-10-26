
import ollama
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "dataset" / "prompt_experiment.csv"

prompts = [
    {
        "Style": "Vague",
        "Prompt": "Explain Python."
    },
    {
        "Style": "Improved",
        "Prompt": (
            "Explain Python programming to a beginner "
            "in 100 words. Include its main features "
            "and one simple real-world example."
        )
    }
]

results = []

for model in ["gemma3:1b", "llama3.2:1b"]:
    for item in prompts:

        print(f"\nTesting {model}: {item['Style']}")

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": item["Prompt"]
                }
            ]
        )

        answer = response["message"]["content"]

        results.append({
            "Model": model,
            "Prompt_Style": item["Style"],
            "Prompt": item["Prompt"],
            "Response": answer,
            "Word_Count": len(answer.split())
        })

        print(answer)
        print("-" * 50)

pd.DataFrame(results).to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nPrompt experiment saved successfully!")
