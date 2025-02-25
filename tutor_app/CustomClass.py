class VirtualPatient:
    def __init__(self, clinical_case):
        self.name = clinical_case["name"]
        self.age = clinical_case["age"]
        self.gender = clinical_case["gender"]
        self.symptoms = clinical_case["symptoms"]
        self.history = clinical_case["history"]
        self.diagnosis = clinical_case["diagnosis"]
        self.emotions = clinical_case["emotions"]
        self.pain_level = clinical_case["pain_level"]
        self.conversation_history = []

    def generate_response_prompt(self, question):
        prompt = f"The patient {self.name} is a {self.age}-year-old {self.gender} who is feeling {self.emotions}. "
        if "symptoms" in question.lower():
            prompt += f"They report experiencing {', '.join(self.symptoms)}."
        elif "history" in question.lower():
            prompt += f"They have a history of {self.history}."
        elif "pain" in question.lower():
            prompt += f"Their pain level is {self.pain_level}."
        else:
            prompt += f"They seem to be struggling with answering the question."
        
        return prompt
