from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from keras.models import load_model

def load_huggingface_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return model, tokenizer

def load_local_model(model_path: str):
    if model_path.endswith('.h5') or model_path.endswith('.keras'):
        model = load_model(model_path)
    else:
        raise ValueError("Unsupported model file format. Only .h5 and .keras are supported.")

    return model