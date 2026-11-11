from deepeval import assert_test, evaluate
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
    ConversationalTestCase,
    Turn,
    MultiTurnParams,
)
from deepeval.metrics import GEval, ConversationalGEval
from deepeval.models import GeminiModel
import os
from dotenv import load_dotenv

load_dotenv()


gemini_model = GeminiModel(
    model="gemini-3.1-flash-lite", api_key=os.getenv("GOOGLE_API_KEY")
)


# def test_correctness():
# correctness_metric = GEval(
# name="Correctness",
# criteria="Check if the actual output is the same as the expected output. Do not focus too much on the formatting. If the meaning is the same that is fine. It just has to be a true answer.",
# evaluation_params=[
# SingleTurnParams.ACTUAL_OUTPUT,
# SingleTurnParams.EXPECTED_OUTPUT,
# ],
# threshold=0.5,
# model=gemini_model,
# )

# test_case = LLMTestCase(
# input="What is 5 divided by 2?",
# expected_output="2.5",
# actual_output="The result is 2.5",
# )

# assert_test(test_case, [correctness_metric])


def test_correctness():
    professionalism_metric = ConversationalGEval(
        name="Professionalism",
        criteria="Determine whether the assistant answered the questions of the user in a professional and polite manner.",
        model=gemini_model,
        evaluation_params=[MultiTurnParams.CONTENT],
    )

    conversation_example1 = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Is python a interpreted language?"),
            Turn(role="assistant", content="Of Course. How could you not know that?"),
            Turn(role="user", content="What about C++?"),
            Turn(
                role="assistant",
                content="Damn... you really know nothing. C++ is compiled man.",
            ),
        ]
    )

    conversation_example2 = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Is python a interpreted language?"),
            Turn(role="assistant", content="Yes, python is the interpreted language."),
            Turn(role="user", content="What about C++?"),
            Turn(role="assistant", content="C++ is compiled language."),
        ]
    )

    evaluate(
        test_cases=[conversation_example1, conversation_example2],
        metrics=[professionalism_metric],
    )
