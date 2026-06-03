# ===============================
# 🔹 1. IMPORTS
# ===============================
import torch
import torch.nn as nn
import pandas as pd
import re
from transformers import BertTokenizer, BertModel
import sys
sys.stdout.reconfigure(encoding='utf-8')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 🔹 2. STRONG PREPROCESSING (REMOVE EVERYTHING)
# ===============================
def normalize_text(text: str):
    text = str(text)

    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+", "", text)

    # remove ALL punctuation & symbols
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def shorten_text(text: str, max_words=1000):
    return " ".join(text.split()[:max_words])


def preprocess(text):
    cleaned = normalize_text(text)
    cleaned = shorten_text(cleaned)
    return cleaned


# ===============================
# 🔹 3. MODEL (BERT + CAPSULE)
# ===============================
class CapsuleLayer(nn.Module):
    def __init__(self, input_dim=768, num_capsules=8, dim_capsule=16):
        super().__init__()
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.fc = nn.Linear(input_dim, num_capsules * dim_capsule)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.num_capsules, self.dim_capsule)

        norm = torch.norm(x, dim=-1, keepdim=True)
        x = (x / (1 + norm**2)) * (norm / (1 + norm))

        return x


class BertCapsuleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            "D:/FakeNewsDetection/Model/bert_model"
        )

        self.capsule = CapsuleLayer(768, 8, 16)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(8 * 16, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls = outputs.last_hidden_state[:, 0, :]

        x = self.capsule(cls)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        return self.classifier(x)


# ===============================
# 🔹 4. LOAD MODEL + TOKENIZER
# ===============================
model_path = "D:/FakeNewsDetection/Model/Bert_and_capsule/final_model.pth"

tokenizer = BertTokenizer.from_pretrained(
    "D:/FakeNewsDetection/Model/tokenizer"
)

model = BertCapsuleModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model Loaded Successfully")


# ===============================
# 🔹 5. PREDICTION FUNCTION
# ===============================
def predict(text):
    cleaned_text = preprocess(text)

    encoding = tokenizer(
        cleaned_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        confidence = probs.max().item()

    return cleaned_text, pred, confidence


def predict_label(text):
    cleaned_text, pred, conf = predict(text)

    label = "Real News" if pred == 1 else "Fake News"

    return cleaned_text, label, conf


# ===============================
# 🔹 6. TEST INPUT (WITH BEFORE/AFTER)
# ===============================
text = """Iran war disrupts the circuit board supply chain, raises costs for tech firms"""

cleaned, label, conf = predict_label(text)

print("\n========== RESULT ==========")
print("Original Text:\n", text)
print("\nCleaned Text:\n", cleaned)
print("\nPrediction:", label)
print("Confidence:", round(conf, 3))