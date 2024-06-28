
class HallucinationTestDataset:
    def get_dataset():
        test_dataset = [
            {"input": "I'm trying to learn about science, can you give me a quiz to test my knowledge",
            "response": "science",
            "subjects": ["davinci", "telescope", "physics", "curie"]},
            {"input": "I'm an geography expert, give a quiz to prove it?",
            "response": "geography",
            "subjects": ["paris", "france", "louvre"]},
            {"input": "Quiz me about Italy",
            "response": "geography",
            "subjects": ["rome", "alps", "sicily"]
            },
        ]

        return test_dataset