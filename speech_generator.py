from transformers import pipeline
from datasets import load_dataset
import sounddevice
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class SpeechGenerator:
    def __init__(self, device, model_id = "microsoft/speecht5_tts", speaker = 7306):
        self.synthesiser = pipeline("text-to-speech", model = model_id, device=device)
        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embedding = torch.tensor(self.embeddings_dataset[speaker]["xvector"]).unsqueeze(0)
    
    def synthesize_audio(self, text):
        speech = self.synthesiser(text, 
                             forward_params={"speaker_embeddings": self.speaker_embedding})
        return speech
