from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import torch

app = FastAPI()

# Root endpoint for status
@app.get("/")
async def read_root():
    return {"message": "API is running! Use POST /predict with JSON {\"reply\": \"...\"}"}

# Load the saved model
model_path = 'saved_model'
try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    raise Exception(f"Failed to load model from {model_path}: {str(e)}")

label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

class InputText(BaseModel):
    reply: str

@app.post('/predict')
def predict(input: InputText):
    result = classifier(input.reply)[0]
    label_id = int(result['label'].split('_')[-1])  # LABEL_0 -> 0
    level = label_map[label_id]
    confidence = result['score']
    return {'level': level, 'confidence': confidence}