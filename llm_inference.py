from openai import OpenAI

class LLMInference:
    def __init__(self, model_id = 'mistral:instruct', url = 'http://localhost:11434/v1', api_key = 'ollama'):
        self.model_id = model_id
        self.url = url
        self.api_key = api_key
        self.client = OpenAI(
            base_url = self.url,
            api_key=self.api_key,
        )

    def query_model(self, text):
        response = self.client.chat.completions.create(
            model = self.model_id,
            messages = [
                {"role": "system", "content": "You are a helpful assistant.\
                You provide single sentence and accurate answers to the user's question"},
                {"role": "user", "content": text}
                ]
            )        
        
        return response.choices[0].message.content
