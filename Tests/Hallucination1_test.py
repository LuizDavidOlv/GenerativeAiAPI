
import pytest
from Evaluations.ModelGradedEvaluation import ModelGradedEvaluation
from Mocks.HallucinationTestDataset import HallucinationTestDataset
from Prompts.QuizBankPrompt import QUIZ_BANK
import pandas as pd
from IPython.display import display, HTML
import os
from dotenv import load_dotenv, find_dotenv
from Prompts.ModelGradedPrompts import MODEL_GRADED_SYSTEM_MESSAGE, MODEL_GRADED_USER_MESSAGE

load_dotenv(find_dotenv())

@pytest.fixture
def Mge():
    return ModelGradedEvaluation()

def test_model_graded_eval_hallucination(Mge):
    assistant = Mge.assistant_chain()
    quiz_request = "Write me a quiz about books."
    result = assistant.invoke({"question": quiz_request})
    eval_prompt = Mge.create_eval_prompt(system_message=MODEL_GRADED_SYSTEM_MESSAGE, human_message=MODEL_GRADED_USER_MESSAGE)
    eval_chain = Mge.create_eval_chain(eval_prompt)
    
    eval_response = eval_chain.invoke({"context": QUIZ_BANK, "agent_response": result})

    assert "Decision: No" in eval_response
    


def test_report_evals(Mge):
    assistant_chain = Mge.assistant_chain()
    eval_prompt = Mge.create_eval_prompt(system_message=MODEL_GRADED_SYSTEM_MESSAGE, human_message=MODEL_GRADED_USER_MESSAGE)
    model_graded_evaluator = Mge.create_eval_chain(eval_prompt)
    test_dataset = HallucinationTestDataset.get_dataset()

    eval_results = Mge.evaluate_dataset(
        dataset = test_dataset, 
        quiz_bank = QUIZ_BANK, 
        assistant = assistant_chain, 
        evaluator = model_graded_evaluator
    )

    # df = pd.DataFrame(eval_results)
    # ## clean up new lines to be html breaks
    # df_html = df.to_html().replace("\\n","<br>")

    # if "hallucination_test_results.html" in os.listdir():
    #     os.remove("hallucination_test_results.html")

    # with open("hallucination_test_results.html","w") as f:
    #     f.write(df_html)
    #TODO: figure out how to upload the html file to gitlab
    
    for result in eval_results:
        assert "Decision: No" in result["grader_response"]
    

if __name__ == "__main__":
   pytest.main([__file__])