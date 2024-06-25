from dotenv import load_dotenv, find_dotenv
from Prompts.SystemMessagePrompt import SYSTEM_MESSAGE
from Evaluations.RuleBasedEvaluation import RuleBasedEvaluation


load_dotenv(find_dotenv())

class GenerateQuiz_Test:
    def __init__ (self):
        self.Rbl = RuleBasedEvaluation()

    def test_science_quiz(self):
        question = "Generate a quiz about science."
        expected_subjects = ["davinci","telescope","physics","curie"]
        self.Rbl.eval_expected_words(
            system_message=SYSTEM_MESSAGE, 
            question=question, 
            expected_words=expected_subjects
            )

    def test_geography_quiz(self):
        question = "Generate a quiz about geography."
        expected_subjects = ["paris","frace","louvre"]
        self.Rbl.eval_expected_words(
            system_message=SYSTEM_MESSAGE, 
            question=question, 
            expected_words=expected_subjects
            )
        
    def test_refusal_rome(self):
        question = "Help me create a quiz about Rome."
        decline_response = "I'm sorry"
        self.Rbl.evaluate_refusal(
            system_message=SYSTEM_MESSAGE,
            question=question,
            decline_response=decline_response
        )               

if __name__ == "__main__":
    test = GenerateQuiz_Test()
    test.test_science_quiz()
    test.test_geography_quiz()
    test.test_refusal_rome()
    print("All tests passed!")