import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from training.dataset_loader import FakeNewsDataset
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(r"D:\Trustcheck\model\bert_model")
tokenizer = BertTokenizer.from_pretrained(r"D:\Trustcheck\model\tokenizer")
model.to(device)
model.eval()

# Load your test dataset
test_df = pd.read_csv(r"D:\Trustcheck\data\test.csv")
test_texts = [str(t).lower().strip() for t in test_df['text'].tolist()]
test_labels = test_df['label'].tolist()
test_dataset = FakeNewsDataset(test_texts, test_labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Run inference
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute F1 and Confusion Matrix
f1 = f1_score(all_labels, all_preds)
print(f"Test F1 Score: {f1:.4f}")

cm = confusion_matrix(all_labels, all_preds)
print(f"Confusion Matrix:\n{cm}")

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Test Confusion Matrix')
plt.show()
