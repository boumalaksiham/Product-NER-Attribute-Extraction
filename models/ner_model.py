"""
Product NER Model — Fine-tuned BERT for Token Classification
=============================================================

Architecture:
    Input: "Apple iPhone 14 Pro 256GB Deep Purple Unlocked"
        ↓
    [BERT Tokenizer]  → splits into subword tokens
    ["Apple", "i", "##Phone", "14", "Pro", "256", "##GB", ...]
        ↓
    [BERT Encoder]  → contextual embedding for EACH token (768-dim)
        ↓
    [Linear Classification Head]  → label logits for EACH token
        ↓
    [Argmax]  → predicted label for each token
    [B-BRD,  B-MDL,  I-MDL, I-MDL, B-STR,  B-CLR,   I-CLR,   B-NET]

Key challenge — subword tokenization:
    BERT splits words into subword pieces:
    "iPhone" → ["i", "##Phone"]
    "256GB"  → ["256", "##GB"]
    
    We only predict labels for the FIRST subword of each word.
    The rest get a special -100 label that is ignored in loss computation.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np


class ProductNERModel(nn.Module):
    """
    BERT-based token classifier for product attribute extraction.
    
    For each input token, predicts one of:
        B-BRD, I-BRD  → Brand
        B-MDL, I-MDL  → Model
        B-CLR, I-CLR  → Color
        B-STR, I-STR  → Storage
        B-SIZ, I-SIZ  → Size
        B-MAT, I-MAT  → Material
        B-NET, I-NET  → Network/Connectivity
        B-GEN, I-GEN  → Generation
        B-CNT, I-CNT  → Count/Quantity
        O             → Not an entity
    """

    def __init__(self, num_labels: int,
                 model_name: str = "distilbert-base-uncased",
                 dropout_rate: float = 0.1):
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # Pretrained BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        self.dropout = nn.Dropout(dropout_rate)

        # Token-level classification head
        # Unlike sentence classification (which uses CLS token),
        # NER uses ALL token embeddings
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      Token IDs, shape (batch, seq_len)
            attention_mask: 1 for real tokens, 0 for padding, shape (batch, seq_len)

        Returns:
            logits: shape (batch, seq_len, num_labels)
                    One set of label scores per token
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use ALL token embeddings (not just CLS)
        # sequence_output shape: (batch, seq_len, 768)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # Apply classifier to every token position
        # logits shape: (batch, seq_len, num_labels)
        logits = self.classifier(sequence_output)

        return logits


class ProductNERPredictor:
    """
    Wraps trained model for easy inference.
    Handles the subword → word alignment automatically.
    """

    def __init__(self, model: ProductNERModel,
                 tokenizer, id_to_label: dict,
                 max_length: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.id_to_label = id_to_label
        self.max_length = max_length
        self.model.eval()

    def predict(self, title: str) -> dict:
        """
        Extracts named entities from a product title.

        Returns:
            {
                "title": "Apple iPhone 14 Pro 256GB Deep Purple Unlocked",
                "tokens": ["Apple", "iPhone", "14", "Pro", "256GB", ...],
                "labels": ["B-BRD", "B-MDL", "I-MDL", "I-MDL", "B-STR", ...],
                "entities": {
                    "BRAND":   ["Apple"],
                    "MODEL":   ["iPhone 14 Pro"],
                    "STORAGE": ["256GB"],
                    "COLOR":   ["Deep Purple"],
                    "NETWORK": ["Unlocked"]
                }
            }
        """
        # Step 1: Split title into words
        words = title.split()

        # Step 2: Tokenize with word IDs so we know which subword
        #         belongs to which original word
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,   # tells tokenizer words are pre-split
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=False
        )

        # Get word_ids: maps each subword token back to its original word index
        word_ids = encoding.word_ids()  # e.g. [None, 0, 1, 1, 2, 3, None]
        #                                        ^CLS  ^Apple ^i ^##Phone ^14 ^Pro ^SEP

        with torch.no_grad():
            logits = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"]
            )

        # Step 3: Get predicted label for each subword token
        predictions = torch.argmax(logits, dim=2)[0].tolist()

        # Step 4: Align subword predictions back to original words
        # Only keep the prediction for the FIRST subword of each word
        word_labels = []
        prev_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # skip [CLS] and [SEP]
            if word_id != prev_word_id:
                # First subword of this word — keep prediction
                word_labels.append(self.id_to_label[predictions[idx]])
            prev_word_id = word_id

        # Trim to actual number of words (in case of truncation)
        word_labels = word_labels[:len(words)]

        # Step 5: Group consecutive B/I tags into entity spans
        entities = self._extract_entities(words, word_labels)

        return {
            "title": title,
            "tokens": words,
            "labels": word_labels,
            "entities": entities
        }

    def _extract_entities(self, words: list, labels: list) -> dict:
        """
        Groups consecutive B/I tags into complete entity strings.

        Example:
            words:  ["Deep",  "Purple", "Unlocked"]
            labels: ["B-CLR", "I-CLR",  "B-NET"]
            →  COLOR: ["Deep Purple"], NETWORK: ["Unlocked"]
        """
        # Map short codes to readable names
        label_names = {
            "BRD": "BRAND", "MDL": "MODEL", "CLR": "COLOR",
            "STR": "STORAGE", "SIZ": "SIZE", "MAT": "MATERIAL",
            "NET": "NETWORK", "GEN": "GENERATION", "CNT": "COUNT"
        }

        entities = {name: [] for name in label_names.values()}
        current_entity = []
        current_type = None

        for word, label in zip(words, labels):
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity and current_type:
                    readable = label_names.get(current_type, current_type)
                    entities[readable].append(" ".join(current_entity))
                # Start new entity
                current_type = label[2:]
                current_entity = [word]

            elif label.startswith("I-") and current_type == label[2:]:
                # Continue current entity
                current_entity.append(word)

            else:
                # O label or mismatched I- tag — save and reset
                if current_entity and current_type:
                    readable = label_names.get(current_type, current_type)
                    entities[readable].append(" ".join(current_entity))
                current_entity = []
                current_type = None

        # Don't forget the last entity
        if current_entity and current_type:
            readable = label_names.get(current_type, current_type)
            entities[readable].append(" ".join(current_entity))

        # Remove empty entity types
        return {k: v for k, v in entities.items() if v}
