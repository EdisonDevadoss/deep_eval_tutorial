from deepeval import evaluate
from deepeval.test_case import (
    ConversationalTestCase,
    Turn,
    MultiTurnParams,
    LLMTestCase,
)
from deepeval.models import GeminiModel
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

import os
from dotenv import load_dotenv

load_dotenv()


gemini_model = GeminiModel(
    model="gemini-3.1-flash-lite", api_key=os.getenv("GOOGLE_API_KEY")
)

# metric = AnswerRelevancyMetric(threshold=0.5, include_reason=True, model=gemini_model)

metric = FaithfulnessMetric(threshold=0.5, include_reason=True, model=gemini_model)


test_case = LLMTestCase(
    input="What is the capital of france?",
    actual_output="Paris",
    retrieval_context=["The capital of France is Medrid"],
)

evaluate(test_cases=[test_case], metrics=[metric])
