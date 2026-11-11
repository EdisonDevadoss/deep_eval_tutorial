from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.models import GeminiModel

import os
from dotenv import load_dotenv

load_dotenv()

gemini_model = GeminiModel(
    model="gemini-3.1-flash-lite", api_key=os.getenv("GOOGLE_API_KEY")
)

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine of the 'actual output' is correct based on the 'expected output'. It has to be correct in it's essence. Ignore format issues.",
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.5,
    model=gemini_model,
)

golden = [
    Golden(input="What is the capital of France?", expected_output="Paris"),
    Golden(input="What is 12 * 3?", expected_output="36"),
]

dataset = EvaluationDataset(goldens=golden)


def simulate_LLM_answer(prompt):
    return {
        "What is the capital of France?": "The capital is Paris.",
        "What is 12 * 3?": "thirty-six",
    }[prompt]


for golden in dataset.goldens:
    dataset.add_test_case(
        LLMTestCase(
            input=golden.input,
            expected_output=golden.expected_output,
            actual_output=simulate_LLM_answer(golden.input),
        )
    )

evaluate(test_cases=dataset.test_cases, metrics=[correctness_metric])
