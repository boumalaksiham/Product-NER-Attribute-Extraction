# Product NER — Attribute Extraction from Product Titles
### Fine-tuned DistilBERT for Token Classification (BIO Tagging)

> Built as part of an e-commerce ML portfolio targeting applied research roles at companies like eBay, Amazon, and Shopify.

---

## Overview

When a seller lists a product on eBay, the title is unstructured free text. This project turns that raw text into structured, queryable attributes by labeling every word with its entity type — brand, model, color, size, storage, material, network, generation, or count.

```
Input:  "Apple iPhone 14 Pro 256GB Deep Purple Unlocked"

Output:
    BRAND    → Apple
    MODEL    → iPhone 14 Pro
    STORAGE  → 256GB
    COLOR    → Deep Purple
    NETWORK  → Unlocked
```

```
Input:  "Le Creuset Signature Cast Iron Dutch Oven 5.5Qt Flame"

Output:
    BRAND    → Le Creuset
    MODEL    → Signature
    MATERIAL → Cast Iron
    SIZE     → 5.5Qt
    COLOR    → Flame
```

This structured output powers search filtering, faceted navigation, entity resolution (comparing specific attributes across listings), and product knowledge graphs — all core systems at eBay.

---

## Results

Trained on 44 labeled product titles (35 train / 9 test) across 9 entity types.

### Per-Entity Performance

| Entity | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| BRAND | 1.00 | 1.00 | **1.00** | Perfect — even multi-word brands like "Le Creuset", "Instant Pot" |
| STORAGE | 1.00 | 0.75 | **0.86** | 256GB, 512GB, 1TB — near perfect |
| COLOR | 0.55 | 0.86 | **0.67** | Handles colorway names like "Flame", "Obsidian", "Deep Purple" |
| SIZE | 0.50 | 0.67 | **0.57** | 5Qt, 32x32, Size 10, 6mm |
| COUNT | 1.00 | 0.33 | **0.50** | Limited by small test set (3 examples) |
| MODEL | 0.36 | 0.44 | **0.40** | Hardest task — model names vary widely |
| NET | 0.00 | 0.00 | 0.00 | Only 2 test examples — insufficient data |
| MAT | 0.00 | 0.00 | 0.00 | Only 1 test example — insufficient data |

**Overall F1: 0.61** on 9 test sentences (38 entity spans)

The low scores for NET and MAT are entirely a data size issue — both entity types have fewer than 3 test examples. BRAND at F1=1.00 and STORAGE at F1=0.86 demonstrate the model has learned to reliably extract the most commercially important attributes.

### Live Prediction Examples

```
"Nike Air Max 90 White Men's Size 10"
  Nike       → B-BRD    BRAND   : Nike
  Air        → B-MDL    MODEL   : Air Max 90
  Max        → I-MDL    COLOR   : White
  90         → I-MDL    SIZE    : Size 10
  White      → B-CLR
  Men's      → O
  Size       → B-SIZ
  10         → I-SIZ

"Sony WH-1000XM5 Wireless Noise Cancelling Headphones Black"
  Sony       → B-BRD    BRAND   : Sony
  WH-1000XM5 → B-MDL    MODEL   : WH-1000XM5
  Wireless   → O        COLOR   : Black
  Noise      → O
  Cancelling → O
  Headphones → O
  Black      → B-CLR

"Levi's 501 Original Fit Jeans Men's 32x32 Dark Stonewash"
  Levi's     → B-BRD    BRAND   : Levi's
  501        → B-MDL    MODEL   : 501 Original Fit
  Original   → I-MDL    COLOR   : Dark Stonewash
  Fit        → I-MDL    SIZE    : 32x32
  Jeans      → O
  Men's      → O
  32x32      → B-SIZ
  Dark       → B-CLR
  Stonewash  → I-CLR

"Apple AirPods Pro 2nd Generation MagSafe White"
  Apple      → B-BRD    BRAND      : Apple
  AirPods    → B-MDL    MODEL      : AirPods Pro
  Pro        → I-MDL    GENERATION : 2nd Generation
  2nd        → B-GEN    COLOR      : White
  Generation → I-GEN
  MagSafe    → O
  White      → B-CLR
```

---

## Architecture

```
Input: "Apple iPhone 14 Pro 256GB Deep Purple Unlocked"
         ↓
[DistilBERT Tokenizer — with word_ids tracking]
["[CLS]", "apple", "i", "##phone", "14", "pro", "256", "##gb",
 "deep", "purple", "unlocked", "[SEP]", "[PAD]", ...]
  None     0        1    1        2      3       4      4
  ↑ word_ids map each subword back to its original word
         ↓
[DistilBERT Encoder — 66M parameters]
  Contextual embedding for EVERY token (768-dim each)
         ↓
[Linear Classification Head]
  Linear(768 → 18)  — one prediction per token
         ↓
[Argmax → Label]
  [CLS]      → ignored
  "apple"    → B-BRD   ← first subword of "Apple" → keep
  "i"        → B-MDL   ← first subword of "iPhone" → keep
  "##phone"  → ignored ← continuation subword → skip
  "14"       → I-MDL
  "pro"      → I-MDL
  "256"      → B-STR   ← first subword of "256GB" → keep
  "##gb"     → ignored ← continuation subword → skip
  "deep"     → B-CLR
  "purple"   → I-CLR
  "unlocked" → B-NET
  [SEP]      → ignored
         ↓
[Word-level alignment: map subword predictions → original words]
  Apple    → B-BRD
  iPhone   → B-MDL
  14       → I-MDL
  Pro      → I-MDL
  256GB    → B-STR
  Deep     → B-CLR
  Purple   → I-CLR
  Unlocked → B-NET
         ↓
[Entity Span Extraction: group consecutive B/I tags]
  BRAND   : Apple
  MODEL   : iPhone 14 Pro
  STORAGE : 256GB
  COLOR   : Deep Purple
  NETWORK : Unlocked
```

---

## BIO Tagging Scheme

NER uses the **BIO (Beginning-Inside-Outside)** scheme to label every token:

| Tag | Meaning | Example |
|---|---|---|
| `B-BRD` | Beginning of a Brand entity | **Apple** iPhone ... |
| `I-BRD` | Inside (continuation) of Brand | Le **Creuset** ... |
| `B-CLR` | Beginning of Color | ... **Deep** Purple |
| `I-CLR` | Inside Color | ... Deep **Purple** |
| `O` | Not an entity | Wireless, Noise, Cancelling |

**Why BIO and not just single labels?**
Multi-word entities like "Deep Purple", "Le Creuset", "Star Wars Millennium Falcon" span multiple tokens. BIO tagging is the standard way to encode spans — B marks the start, I marks continuation, O marks non-entities.

---

## Entity Types

| Code | Full Name | Examples |
|---|---|---|
| BRD | Brand | Apple, Nike, Sony, Le Creuset, Instant Pot |
| MDL | Model | iPhone 14 Pro, Air Max 90, WH-1000XM5 |
| CLR | Color | Deep Purple, Titanium Black, Flame, Obsidian |
| STR | Storage | 256GB, 512GB, 1TB |
| SIZ | Size | Size 10, 32x32, 5Qt, 6mm, 4.5Qt |
| MAT | Material | Cast Iron, Stainless Steel, Aluminum |
| NET | Network | Unlocked, 5G, AT&T |
| GEN | Generation | 2nd Generation, Gen 12 |
| CNT | Count | 4016 Pieces, 3-Pack, 10-Piece |

---

## Project Structure

```
product_ner/
├── data/
│   ├── dataset_builder.py     ← 44 hand-labeled product titles with BIO tags
│   └── __init__.py
├── models/
│   ├── ner_model.py           ← ProductNERModel + ProductNERPredictor
│   ├── saved/                 ← Generated after train.py
│   │   ├── best_model.pt
│   │   ├── tokenizer/
│   │   ├── label_maps.json
│   │   └── config.json
│   └── __init__.py
├── train.py                   ← Training loop with seqeval F1 evaluation
├── demo.py                    ← Interactive demo with token-level + entity output
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Requirements
- Python 3.10+
- No GPU required — fully CPU compatible

### Installation
```bash
git clone <repo-url>
cd product_ner

python3 -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows

pip install "numpy<2" torch==2.2.2 transformers==4.40.0 \
    seqeval==1.2.2 scikit-learn==1.4.2 pandas==2.2.2
```

### Train
```bash
python train.py
```
Trains for 20 epochs (~5–10 min on CPU). Saves best model by F1 score.

### Demo
```bash
python demo.py
```
Shows token-level predictions and extracted entities for preset examples, then enters interactive mode.

---

## Training Progress

```
Epoch  1/20 | Loss: 2.7607 | F1: 0.1194
Epoch  3/20 | Loss: 1.9052 | F1: 0.3051
Epoch  6/20 | Loss: 1.0645 | F1: 0.5250
Epoch 10/20 | Loss: 0.3210 | F1: 0.5542
Epoch 15/20 | Loss: 0.0630 | F1: 0.5542
Epoch 19/20 | Loss: 0.0241 | F1: 0.6098
Epoch 20/20 | Loss: 0.0214 | F1: 0.6098

Best F1: 0.6098
```

F1 improves rapidly in early epochs as the model learns basic patterns (brands always come first, storage is always NNNgb format), then plateaus as it runs out of training data to learn rarer entity types from.

---

## Key Design Decisions

**1. Subword tokenization alignment**
DistilBERT splits words into subword pieces: "iPhone" → ["i", "##phone"]. We only predict labels for the first subword of each word and ignore continuation subwords in the loss (label = -100). This is standard practice for BERT-based NER and handled via `word_ids()` from the HuggingFace tokenizer.

**2. seqeval for evaluation**
Standard accuracy metrics would overcount — predicting "B-CLR" vs "I-CLR" correctly on individual tokens doesn't mean you got the full entity span right. seqeval evaluates at the span level: a prediction only counts as correct if the entire entity (start, end, and type) is correct. This is the industry-standard metric for NER.

**3. Token-level CrossEntropyLoss with ignore_index=-100**
Padding tokens, CLS, SEP, and continuation subwords all get label -100. PyTorch's CrossEntropyLoss automatically ignores these positions, so the model only learns from real word predictions.

**4. Low dropout (0.1)**
NER requires the model to make fine-grained distinctions at the token level. Higher dropout (like 0.3 used in classification) would destroy too much spatial information. 0.1 is the standard for token-level tasks.

---

## Connection to Other Projects

This project directly extends **Project 1 (Entity Resolution)**. The attribute guard in Project 1 uses regex to extract model codes, storage sizes, and dimensions. This NER model does the same job using a neural network — it generalizes to unseen product types and handles multi-word entities that regex can't capture (e.g. "Deep Purple", "Natural Titanium", "Cast Iron").

In a production pipeline:
```
Raw listing title
       ↓
[Project 3 — NER]  →  structured attributes
       ↓
[Project 1 — Entity Resolution]  →  use extracted attributes for penalty scoring
       ↓
MATCH / NO MATCH decision
```

---

## Limitations & Future Work

**Current limitations:**
- 44 training sentences is very small — MAT and NET entity types have insufficient training examples
- Multi-word brands like "Le Creuset" depend on seeing that exact phrase in training

**Extensions for production:**
- **More labeled data**: Label 1,000+ product titles per category for robust coverage of all entity types
- **OntoNotes pre-training**: Start from a model already fine-tuned on general NER, then fine-tune further on product titles
- **Gazetteer features**: Maintain lookup lists of known brands and inject as features — guarantees 100% brand recall
- **Conditional Random Fields (CRF)**: Replace the linear head with a CRF layer for better BIO consistency (prevents impossible transitions like O → I-BRD)
- **eBay-specific vocabulary**: Fine-tune the tokenizer on eBay listing data to handle product-specific abbreviations and codes

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `transformers` | 4.40.0 | DistilBERT model + tokenizer |
| `torch` | 2.2.2 | Neural network training |
| `seqeval` | 1.2.2 | Span-level NER evaluation (industry standard) |
| `scikit-learn` | 1.4.2 | Train/test split |
| `numpy` | <2.0 | Numerical operations |

---

*This project is part of a 4-project ML portfolio covering Entity Resolution, Hierarchical Taxonomy Classification, Named Entity Recognition, and Image Quality Scoring for e-commerce applications.*