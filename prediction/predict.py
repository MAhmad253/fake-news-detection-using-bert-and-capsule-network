from transformers import BertForSequenceClassification, BertTokenizer

# Load model & tokenizer
model = BertForSequenceClassification.from_pretrained(r"D:\Trustcheck\model\bert_model")
tokenizer = BertTokenizer.from_pretrained(r"D:\Trustcheck\model\tokenizer")

# Sample text
text = """He refuted Trump's accusation that Iran was violating the truce by not opening this strategic waterway. Under the terms of the ceasefire announced by Pakistan on 8 April, Iran had agreed to open this strait. Its closure is causing economic shocks worldwide.

Baghaei recalled what happened after Iran's foreign minister posted on social media on Friday that the maritime corridor would "completely open" on routes designated by Iran.

"President Trump immediately said 'thank you Iran', and then an hour later he said that he would keep his blockade."

But the spokesman would not be drawn on what conditions Iran required to return to the negotiating table."""
# Tokenize with truncation
inputs = tokenizer(
    text,
    return_tensors="pt",
    max_length=512,
    truncation=True
)

import string

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

# Model prediction
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# Print result
if predictions.item() == 0:
    print("Predicted class: Fake")
else:
    print("Predicted class: True")
