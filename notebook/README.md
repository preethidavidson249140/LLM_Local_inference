
# Week 8 – Local LLM Inference Using Ollama

## Project Overview
This project demonstrates running open-source
Large Language Models locally using Ollama
and Python.

## Objectives
- Install and configure Ollama.
- Run Gemma 3 and Llama 3.2 locally.
- Generate responses for 10 prompts.
- Compare response length, quality, and accuracy.
- Study the effect of prompt wording.
- Save results for future chatbot development.

## Technologies Used
- Python 3.12
- Visual Studio Code
- Ollama
- Gemma 3 (1B)
- Llama 3.2 (1B)
- Pandas
- Matplotlib
- Jupyter Notebook

## Project Files
- local_llm.py: Generates Gemma responses.
- compare_models.py: Compares both models.
- evaluate_models.py: Creates evaluation data.
- prompt_experiment.py: Tests prompt wording.
- dataset/prompts.csv: Contains 10 prompts.
- dataset/llm_responses.csv: Gemma responses.
- dataset/response_comparison.csv: Model comparison.
- dataset/model_evaluation.csv: Manual ratings.
- dataset/prompt_experiment.csv: Prompt experiments.
- dataset/conversation_history.txt: Saved conversations.
- notebook/Week8_Local_LLM_Inference.ipynb:
  Displays results, comparisons, and graphs.

## Installation

Install Python dependencies:

    python -m pip install -r requirements.txt

Download the local models:

    ollama pull gemma3:1b
    ollama pull llama3.2:1b

Ensure Ollama is running.

## Execution

Run the following programs in order:

    python local_llm.py
    python compare_models.py
    python evaluate_models.py
    python prompt_experiment.py

Open the Jupyter Notebook in VS Code
and run all cells.

## Results
Ten prompts across five domains were tested
using Gemma 3 and Llama 3.2.

Responses were saved in CSV format.
Response lengths were compared automatically.
Quality and accuracy were evaluated manually.

## Conclusion
Local LLM inference provides an opportunity
to explore language models without relying
on paid cloud APIs.

This project forms the foundation for
developing a local AI chatbot in Week 9.
