# DeepEval Tutorial

Hands-on examples for evaluating LLM outputs with [DeepEval](https://github.com/confident-ai/deepeval), using **Google Gemini** as the judge model.

The project follows this video tutorial: <https://www.youtube.com/watch?v=HAoKJT3af7Y>

## What's covered

| File | Topic | Metric(s) |
| --- | --- | --- |
| [test_1_simple.py](test_1_simple.py) | Custom LLM-as-a-judge metrics for single-turn and multi-turn (chat) test cases | `GEval`, `ConversationalGEval` |
| [test_rag_metrics.py](test_rag_metrics.py) | RAG evaluation: is the answer grounded in the retrieved context? | `FaithfulnessMetric`, `AnswerRelevancyMetric` |
| [test_evaluating_dataset.py](test_evaluating_dataset.py) | Evaluating a whole dataset built from "goldens" | `GEval` + `EvaluationDataset` / `Golden` |

### 1. Simple and conversational evaluation (`test_1_simple.py`)

- **Correctness (single-turn, commented out):** a `GEval` metric compares `actual_output` against `expected_output` (e.g. "What is 5 divided by 2?" → "The result is 2.5") and passes if the meaning matches, regardless of formatting. Run it with `assert_test`.
- **Professionalism (multi-turn):** a `ConversationalGEval` metric scores two chat transcripts, one rude assistant and one polite one, to check whether the assistant answered professionally. The rude conversation should fail and the polite one should pass.

### 2. RAG metrics (`test_rag_metrics.py`)

- **`FaithfulnessMetric`** checks whether `actual_output` is supported by `retrieval_context`. The example deliberately uses a wrong context ("The capital of France is Medrid") with the answer "Paris", so the output is *not* faithful to the context and the metric should fail.
- **`AnswerRelevancyMetric`** (commented out) checks whether the answer is relevant to the input question.
- `include_reason=True` makes the judge explain its score.

### 3. Evaluating a dataset (`test_evaluating_dataset.py`)

1. Define `Golden` items (input + expected output).
2. Wrap them in an `EvaluationDataset`.
3. Generate an `actual_output` for each golden. Here `simulate_LLM_answer` stands in for a real LLM call.
4. Add each one as an `LLMTestCase` and run `evaluate()` with a `GEval` correctness metric.

## Key concepts

- **`LLMTestCase`**: one interaction: `input`, `actual_output`, plus optional `expected_output` and `retrieval_context`.
- **`ConversationalTestCase`**: a multi-turn chat made of `Turn(role, content)` objects.
- **Metric**: scores a test case from 0 to 1. It passes when the score is at or above its `threshold` (0.5 in these examples).
- **`GEval`**: a custom metric defined in plain English through `criteria`, with an LLM acting as judge.
- **Judge model**: every metric here uses `GeminiModel` instead of the default OpenAI model, so only a Google API key is needed.
- **`evaluate()` vs `assert_test()`**: `evaluate()` runs metrics and prints a report; `assert_test()` raises on failure, so it fits inside pytest tests.

## Setup

Requirements: Python 3.13 (other recent 3.x versions should work) and a Google Gemini API key.

```bash
# 1. Create and activate a virtual environment
python3 -m venv eval_env
source eval_env/bin/activate

# 2. Install dependencies
pip install deepeval google-genai python-dotenv pytest

# 3. Add your Gemini API key
echo "GOOGLE_API_KEY=your-key-here" > .env
```

The examples use the `gemini-3.1-flash-lite` model. Change the `model=` argument in `GeminiModel(...)` to use a different one.

## Running the examples

Run a test file through DeepEval's pytest wrapper:

```bash
deepeval test run test_1_simple.py
```

The RAG and dataset scripts call `evaluate()` at module level, so you can also run them directly:

```bash
python test_rag_metrics.py
python test_evaluating_dataset.py
```

Plain `pytest test_1_simple.py` works as well. DeepEval prints each metric's score, pass/fail status, and the judge's reasoning. Run results are cached in `.deepeval/`.

## Project structure

```
deep_eval_tutorial/
├── test_1_simple.py            # GEval + ConversationalGEval
├── test_rag_metrics.py         # Faithfulness / Answer Relevancy
├── test_evaluating_dataset.py  # EvaluationDataset + Goldens
├── .env                        # GOOGLE_API_KEY (keep this out of git)
├── .deepeval/                  # DeepEval run cache (generated)
└── eval_env/                   # virtual environment (ignored)
```

## Resources

- Video tutorial: <https://www.youtube.com/watch?v=HAoKJT3af7Y>
- DeepEval docs: <https://deepeval.com/docs/getting-started>
- DeepEval GitHub: <https://github.com/confident-ai/deepeval>
