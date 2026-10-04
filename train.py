"""
Training Script — Product NER
==============================
Trains a DistilBERT token classifier on labeled product title data.

Usage:
    python train.py
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset_builder import LABELED_SENTENCES, get_label_maps
from models.ner_model import ProductNERModel

CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 64,
    "batch_size": 8,
    "epochs": 20,
    "learning_rate": 3e-5,
    "dropout_rate": 0.1,
    "validation_size": 0.2,
    "random_state": 42,
}

IGNORE_INDEX = -100  # PyTorch ignores this label in loss computation


class NERDataset(Dataset):
    """
    PyTorch Dataset for NER.
    Handles the tricky subword tokenization alignment.
    """

    def __init__(self, sentences, label_to_id, tokenizer, max_length):
        self.encodings = []
        self.label_ids = []

        for sentence in sentences:
            words = [w for w, _ in sentence]
            labels = [l for _, l in sentence]

            # Tokenize with word IDs
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Align labels to subword tokens
            word_ids = encoding.word_ids()
            aligned_labels = []
            prev_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(IGNORE_INDEX)  # CLS, SEP, PAD
                elif word_id != prev_word_id:
                    # First subword → use real label
                    aligned_labels.append(label_to_id[labels[word_id]])
                else:
                    # Continuation subword → ignore in loss
                    aligned_labels.append(IGNORE_INDEX)
                prev_word_id = word_id

            self.encodings.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
            })
            self.label_ids.append(torch.tensor(aligned_labels, dtype=torch.long))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings[idx]["input_ids"],
            "attention_mask": self.encodings[idx]["attention_mask"],
            "labels": self.label_ids[idx],
        }


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        # Reshape for loss: (batch * seq_len, num_labels) vs (batch * seq_len)
        loss = criterion(
            logits.view(-1, model.num_labels),
            labels.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, id_to_label, device):
    """
    Evaluates using seqeval — the standard NER evaluation library.
    seqeval handles BIO scheme correctly (partial matches don't count).
    """
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=2)

            for pred_seq, true_seq in zip(predictions, labels):
                true_labels, pred_labels = [], []
                for p, t in zip(pred_seq.tolist(), true_seq.tolist()):
                    if t == IGNORE_INDEX:
                        continue
                    true_labels.append(id_to_label[t])
                    pred_labels.append(id_to_label[p])
                all_true.append(true_labels)
                all_pred.append(pred_labels)

    f1 = f1_score(all_true, all_pred)
    return f1, all_true, all_pred


def main():
    print("=" * 60)
    print("PRODUCT NER — TRAINING")
    print("=" * 60)

    random.seed(CONFIG["random_state"])
    np.random.seed(CONFIG["random_state"])
    torch.manual_seed(CONFIG["random_state"])
    device = torch.device("cpu")

    # 1. Labels
    label_to_id, id_to_label = get_label_maps(LABELED_SENTENCES)
    print(f"\nEntity labels ({len(label_to_id)}): {list(label_to_id.keys())}")

    # 2. Train/validation split
    train_sents, validation_sents = train_test_split(
        LABELED_SENTENCES,
        test_size=CONFIG["validation_size"],
        random_state=CONFIG["random_state"]
    )
    print(f"Train: {len(train_sents)} | Validation: {len(validation_sents)}")

    # 3. Tokenizer
    print(f"\nLoading tokenizer: {CONFIG['model_name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    # 4. Datasets
    train_dataset = NERDataset(train_sents, label_to_id, tokenizer, CONFIG["max_length"])
    validation_dataset = NERDataset(validation_sents, label_to_id, tokenizer, CONFIG["max_length"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=CONFIG["batch_size"])

    # 5. Model
    print("\nBuilding model...")
    model = ProductNERModel(
        num_labels=len(label_to_id),
        model_name=CONFIG["model_name"],
        dropout_rate=CONFIG["dropout_rate"]
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # 7. Training loop
    print(f"\nTraining for {CONFIG['epochs']} epochs...")
    print("-" * 60)

    best_f1 = float("-inf")
    for epoch in range(1, CONFIG["epochs"] + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        f1, _, _ = evaluate(model, validation_loader, id_to_label, device)
        print(f"Epoch {epoch:>2}/{CONFIG['epochs']} | Loss: {loss:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs("models/saved", exist_ok=True)
            torch.save(model.state_dict(), "models/saved/best_model.pt")

    print(f"\nBest F1: {best_f1:.4f}")

    model.load_state_dict(torch.load("models/saved/best_model.pt", map_location=device, weights_only=True))

    # 8. Validation report for the saved checkpoint
    print("\nBest-checkpoint evaluation on validation data:")
    checkpoint_f1, all_true, all_pred = evaluate(model, validation_loader, id_to_label, device)
    print(classification_report(all_true, all_pred, zero_division=0))
    with open("models/saved/validation_report.json", "w") as f:
        json.dump({"evaluation_role": "checkpoint_selection_validation",
                   "train_count": len(train_sents), "validation_count": len(validation_sents),
                   "seed": CONFIG["random_state"], "span_f1": checkpoint_f1,
                   "classification_report": classification_report(all_true, all_pred, output_dict=True, zero_division=0),
                   "true_labels": all_true, "predicted_labels": all_pred}, f, indent=2)

    # 9. Save artifacts
    tokenizer.save_pretrained("models/saved/tokenizer")
    with open("models/saved/label_maps.json", "w") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2)
    with open("models/saved/config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("\n✅ Done! Run: python demo.py")


if __name__ == "__main__":
    main()
