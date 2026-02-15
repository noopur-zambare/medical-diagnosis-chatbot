from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "abhinand/MedEmbed-base-v0.1"
LOCAL_PATH = "./MedEmbed-base-v0.1"

snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=LOCAL_PATH,
    local_dir_use_symlinks=False
)

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH, use_fast=True) 
model = AutoModel.from_pretrained(LOCAL_PATH)
