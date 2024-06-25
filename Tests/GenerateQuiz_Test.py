from dotenv import load_dotenv, find_dotenv
from Prompts.SystemMessagePrompt import SYSTEM_MESSAGE
from Evaluations.RuleBasedEvaluation import RuleBasedEvaluation
import pytest


load_dotenv(find_dotenv())

@pytest.fixture
def Rbl():
    return RuleBasedEvaluation()

def test_science_quiz(Rbl):
    question = "Generate a quiz about science."
    expected_subjects = ["davinci","telescope","physics","curie"]
    Rbl.eval_expected_words(
        system_message=SYSTEM_MESSAGE, 
        question=question, 
        expected_words=expected_subjects
        )
    
def test_geography_quiz(Rbl):
    question = "Generate a quiz about geography."
    expected_subjects = ["paris","frace","louvre"]
    result = Rbl.eval_expected_words(
        system_message=SYSTEM_MESSAGE, 
        question=question, 
        expected_words=expected_subjects
        )

def test_refusal_rome(Rbl):
    question = "Help me create a quiz about Rome."
    decline_response = "I'm sorry"
    result = Rbl.evaluate_refusal(
        system_message=SYSTEM_MESSAGE,
        question=question,
        decline_response=decline_response
    )           

if __name__ == "__main__":
    pytest.main([__file__])