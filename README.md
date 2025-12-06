# AgentSociety Challenge: Enhanced Recommendation Agent

This repository contains our enhanced Recommendation Agent developed for the [AgentSociety Challenge](https://agentsocietychallenge.github.io/). We significantly improved the baseline agent by implementing Context Retrieval (CR), Chain of Thought (CoT) reasoning, Reflection (Self-Correction), and Robustness Safety Nets.

## 📂 Repository Structure

* **`my_rec_agent.py`**: The main execution script containing our custom `MyRecommendationAgent` class and the full evaluation pipeline.

* **`websocietysimulator/`**: The core challenge library. 
  * *Note:* We include this folder in our submission because we modified `llm.py` to support additional APIs (DeepSeek, Gemini, etc.) that were not in the original codebase.
  
* **`log/`**: Evaluation results and JSON metrics are saved here after running the agent.

* **`db/`** (Not included): Contains the processed database (Yelp/Amazon/Goodreads). The simulator loads data from here.

* **`track2/`** (Not included): Contains the evaluation tasks and ground truth files for the recommendation track.

* **`key.py`** (Not included): You must create this file to store your API keys (see Setup below).

## 🚀 Setup & Configuration

### 1. API Keys
For security, API keys are imported from a separate file named `key.py` in the root directory. You can create this file or add your keys directly to the `main` block of `my_rec_agent.py`.

**Create `key.py`:**
```python
# key.py
DEEPSEEK_API_KEY = "sk-..."
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "..."
```

### 2. Model Selection & Workers
We primarily evaluated our agent using **OpenAI** and **DeepSeek**. You can switch models by commenting/uncommenting the relevant lines in the `if __name__ == "__main__":` block of `my_rec_agent.py`.

* **OpenAI (GPT-4o-mini):** 
    * Recommended `max_workers`: **3** (to avoid rate limits).
  ```python
  simulator.set_llm(OpenAILLM(api_key=OPENAI_API_KEY, model="gpt-4o-mini"))
  agent_outputs = simulator.run_simulation(..., max_workers=3)
  ```

* **DeepSeek (DeepSeek-Chat):**
  * Recommended `max_workers`: **10**.
  ```python
  simulator.set_llm(DeepseekLLM(api_key=DEEPSEEK_API_KEY, model="deepseek-chat"))
  agent_outputs = simulator.run_simulation(..., max_workers=10)
  ```

* **Note on Gemini:** While we implemented the `GeminiLLM` class, we excluded it from our final evaluation. The Yelp review dataset frequently triggers Google's safety filters (e.g., discussions of alcohol in bars), which caused incomplete simulation runs.

### 3. Data Directories
If your data is stored in different locations, update these paths in `my_rec_agent.py`:

```python
# 1. Database Directory
simulator = Simulator(data_dir="./db", device="auto", cache=True)

# 2. Tasks & Ground Truth Directory
simulator.set_task_and_groundtruth(
    task_dir=f"./track2/{task_set}/tasks", 
    groundtruth_dir=f"./track2/{task_set}/groundtruth"
)
```

## 🏃‍♂️ Running the Agent

To run the full evaluation simulation using Poetry:

```bash
poetry run python my_rec_agent.py
```

**Outputs:**
* Console output will show the progress of the simulation.
* Final evaluation metrics (HR@1, HR@3, HR@5) are printed at the end.
* Detailed logs and the JSON result file are saved to the **`/log`** directory.

## 📊 Key Features Implemented

1.  **Critical Bug Fixes:** Corrected the baseline aggregation loop that failed to process all 20 candidate items.
2.  **Context Retrieval (Soft Filtering):** Implemented a keyword-based filtering system to prioritize user reviews relevant to the current task (e.g., matching "Sushi" reviews to "Japanese Restaurant" tasks) without discarding general context.
3.  **Chain of Thought (CoT):** Forced the model to generate an "Analysis" section before outputting the final list, improving ranking logic by allowing "thinking time."
4.  **Reflection Loop:** Added an automated retry mechanism. If the model outputs invalid format or hallucinates IDs, the error is fed back to the model for self-correction (up to 2 retries).
5.  **Safety Net:** If all retries fail, the agent returns the default candidate list instead of an empty list, ensuring a non-zero score for syntax errors.