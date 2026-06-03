import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from training.dataset_loader import FakeNewsDataset
from sklearn.metrics import f1_score

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create save directories
os.makedirs('../model/bert_model/', exist_ok=True)
os.makedirs('../tokenizer/', exist_ok=True)

#  Load CSVs

train_df = pd.read_csv(r"D:\Trustcheck\data\train.csv")
test_df = pd.read_csv(r"D:\Trustcheck\data\test.csv")

# Preprocessing function
def preprocess(text):
    text = str(text).lower().strip()
    return text

train_texts = [preprocess(t) for t in train_df['text'].tolist()]
train_labels = train_df['Label'].tolist()

test_texts = [preprocess(t) for t in test_df['text'].tolist()]
test_labels = test_df['Label'].tolist()

#  Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets

full_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer)
test_dataset = FakeNewsDataset(test_texts, test_labels, tokenizer)

# Train/Validation split
val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)

# Load pre-trained BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# Optimizer + scheduler

optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Training loop with F1 tracking

for epoch in range(epochs):
    model.train()
    total_loss = 0
    batch_count = 0

    print(f"\nStarting Epoch {epoch + 1}/{epochs}...\n")

    for batch_idx, batch in enumerate(train_loader, start=1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        batch_count += 1

        # Print progress every 100 batches
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / batch_count
    print(f"\nEpoch {epoch+1} Completed — Average Loss: {avg_loss:.4f}")

    # Validation F1
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    print(f"Validation F1 after Epoch {epoch+1}: {f1:.4f}")

# 9️⃣ Save model & tokenizer

model.save_pretrained('../model/bert_model/')
tokenizer.save_pretrained('../tokenizer/')
print("\nModel and Tokenizer Saved Successfully!")
