import os
from dotenv import load_dotenv
load_dotenv("api.env")
print("Pinecone Key loaded:", os.getenv("PINECONE_API_KEY") is not None)

import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec

MODEL_NAME = "/Users/noopurnishikantzambare/Downloads/medical-diagnosis-chatbot/MedEmbed-base-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=False)
model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()

def embed_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

    return mean_pooled.squeeze().tolist()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "medbot-custom"

if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

def upsert_report(report_id, text, patient_id):
    vector = embed_text(text)

    index.upsert(vectors=[{
        "id": report_id,
        "values": vector,
        "metadata": {
            "patient_id": patient_id,
            "report_text": text
        }
    }])

def query_similar(text, patient_id=None, top_k=3):
    vector = embed_text(text)

    query_params = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True
    }

    if patient_id:
        query_params["filter"] = {"patient_id": {"$eq": patient_id}}

    results = index.query(**query_params)

    return results
