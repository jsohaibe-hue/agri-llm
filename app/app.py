import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE = "google/flan-t5-base"
ADAPTER = "models/lora-agri-adapter"  # dossier extrait du zip

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, ADAPTER).to(device).eval()

def answer(question):
    prompt = f"Answer the following agricultural question.\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=64, no_repeat_ngram_size=3)
    return tokenizer.decode(out[0], skip_special_tokens=True)

gr.Interface(
    fn=answer, inputs="text", outputs="text",
    title="Agri-QA : FLAN-T5 + LoRA",
    description="Démo de recherche. Les réponses peuvent être fausses : ne pas utiliser pour de vrais conseils agricoles.",
).launch()