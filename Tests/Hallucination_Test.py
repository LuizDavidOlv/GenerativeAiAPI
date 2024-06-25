
from Evaluations.ModelGradedEvaluation import ModelGradedEvaluation
from Mocks.HallucinationTestDataset import HallucinationTestDataset
from Prompts.QuizBankPrompt import QUIZ_BANK
import pandas as pd
from IPython.display import display, HTML
import os
from dotenv import load_dotenv, find_dotenv
from Prompts.ModelGradedPrompts import MODEL_GRADED_SYSTEM_MESSAGE, MODEL_GRADED_USER_MESSAGE

load_dotenv(find_dotenv())

class HallucinationTest:
    def __init__ (self):
        self.Mge = ModelGradedEvaluation()

    def test_model_graded_eval_hallucination(self,quiz_bank):
        assistant = self.Mge.assistant_chain()
        quiz_request = "Write me a quiz about books."
        result = assistant.invoke({"question": quiz_request})
        eval_prompt = self.Mge.create_eval_prompt(system_message=MODEL_GRADED_SYSTEM_MESSAGE, human_message=MODEL_GRADED_USER_MESSAGE)
        eval_chain = self.Mge.create_eval_chain(eval_prompt)
        
        eval_response = eval_chain.invoke({"context": quiz_bank, "agent_response": result})

        assert "Decision: No" in eval_response
       


    def report_evals(self,display_to_notebook=False):
        assistant_chain = self.Mge.assistant_chain()
        eval_prompt = self.Mge.create_eval_prompt(system_message=MODEL_GRADED_SYSTEM_MESSAGE, human_message=MODEL_GRADED_USER_MESSAGE)
        model_graded_evaluator = self.Mge.create_eval_chain(eval_prompt)
        test_dataset = HallucinationTestDataset.get_dataset()

        eval_results = self.evaluate_dataset(
            dataset = test_dataset, 
            quiz_bank = QUIZ_BANK, 
            assistant = assistant_chain, 
            evaluator = model_graded_evaluator
        )

        df = pd.DataFrame(eval_results)
        ## clean up new lines to be html breaks
        df_html = df.to_html().replace("\\n","<br>")

        if "hallucination_test_results.html" in os.listdir():
            os.remove("hallucination_test_results.html")

        with open("hallucination_test_results.html","w") as f:
            f.write(df_html)
        



    def evaluate_dataset(self,dataset, quiz_bank, assistant, evaluator):
        eval_results = []
        for row in dataset:
            eval_result = {}
            user_input = row["input"]
            answer = assistant.invoke({"question": user_input})
            eval_response = evaluator.invoke({"context": quiz_bank, "agent_response": answer})

            eval_result["input"] = user_input
            eval_result["output"] = answer
            eval_result["grader_response"] = eval_response

            eval_results.append(eval_result)
        
        return eval_results



if __name__ == "__main__":
    hallucination_test = HallucinationTest()
    hallucination_test.report_evals()