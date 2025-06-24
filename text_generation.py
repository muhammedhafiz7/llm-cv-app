from transformers import pipeline

class Textgenerator:
    def __init__(self, model_name='distilgpt2'):
        self.generator = pipeline('text-generation', model=model_name)

    def generate(self, prompt, max_length=100):
        try:
            result = self.generator(prompt, max_length=max_length, num_return_sequences=1)
            return result[0]['generated_text']
        except Exception as e:
            raise ValueError(f"Text generation failed: {e}")
