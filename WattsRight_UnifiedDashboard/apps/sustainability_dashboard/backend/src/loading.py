import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)


def load_huggingface_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return model, tokenizer


def load_huggingface_generative_model(model_name: str):
    """Load a causal LM from HuggingFace for generative perplexity evaluation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, device


def load_local_model(model_path: str):
    if model_path.endswith('.h5') or model_path.endswith('.keras'):
        from keras.models import load_model  # lazy import avoids TF/numpy conflicts
        model = load_model(model_path)
    else:
        raise ValueError("Unsupported model file format. Only .h5 and .keras are supported.")

    return model