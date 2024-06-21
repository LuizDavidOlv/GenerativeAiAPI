from Prompts.SystemMessagePrompt import SYSTEM_MESSAGE
from Evaluations.RuleBasedEvaluation import Evaluation


def test_science_quiz():
    question = "Generate a quiz about science."
    expected_subjects = ["davinci","telescope","physics","curie"]
    Evaluation.eval_expected_words(
        system_message=SYSTEM_MESSAGE, 
        question=question, 
        expected_words=expected_subjects
        )

def test_geography_quiz():
    question = "Generate a quiz about geography."
    expected_subjects = ["paris","frace","louvre"]
    Evaluation.eval_expected_words(
        system_message=SYSTEM_MESSAGE, 
        question=question, 
        expected_words=expected_subjects
        )
    
def test_refusal_rome():
    question = "Help me create a quiz about Rome."
    decline_response = "I'm sorry"
    Evaluation.evaluate_refusal(
        system_message=SYSTEM_MESSAGE,
        question=question,
        decline_response=decline_response
    )

if __name__ == "__main__":
    test_science_quiz()
    test_geography_quiz()
    test_refusal_rome()
    print("All tests passed!")