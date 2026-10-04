# Product Attribute Extraction with DistilBERT

A named-entity recognition prototype for extracting structured attributes from product titles using BIO token labels and DistilBERT.

## From a title to structured attributes

The curated dataset includes `Apple iPhone 14 Pro 256GB Deep Purple Unlocked`, annotated as:

| Attribute | Annotated span |
|---|---|
| Brand | Apple |
| Model | iPhone 14 Pro |
| Storage | 256GB |
| Color | Deep Purple |
| Network | Unlocked |

This is a **labeling example**, not a claimed model prediction. It illustrates why token-level accuracy is insufficient: extracting only `Purple` misses part of the annotated color, even if most other tokens are correct.

**Design choice:** use contextual token classification for variable-length spans and evaluate complete entity spans with seqeval. The small dataset makes annotation consistency, rare entity types, and subword alignment central to interpreting the result.

## Data and labels

[data/dataset_builder.py](data/dataset_builder.py) contains **44 manually labeled titles**. BIO labels mark an entity's beginning (`B-`), continuation (`I-`), or non-entity tokens (`O`).

| Code | Attribute |
|---|---|
| BRD | Brand |
| MDL | Model |
| CLR | Color |
| STR | Storage |
| SIZ | Size |
| MAT | Material |
| NET | Network |
| GEN | Generation |
| CNT | Count |

The tokenizer aligns word-level annotations with subword tokens. Only the first subword receives the word's label; continuation, special and padding tokens are ignored in the loss using `-100`. Evaluation uses seqeval span-level F1, which requires the full entity span and type to match.

## Training configuration

`distilbert-base-uncased`; 64-token maximum; batch size 8; 20 epochs; learning rate 0.00003; dropout 0.1. A seed-42 split places **35 titles in training and 9 in validation**. Python, NumPy, and PyTorch are seeded before model initialization and training. Determinism across different platforms and library versions is not guaranteed.

This is a validation split used for checkpoint selection. The trainer now saves a checkpoint even when the first F1 is zero, reloads the best checkpoint before reporting, and writes `models/saved/validation_report.json` with counts, span metrics, and label sequences. There is still no independent final test set.

## Setup

Use Python 3.10 or 3.11 for the repository's pinned PyTorch 2.2.2 environment. Run commands from the repository root. The first model load downloads pretrained weights; an internet connection is needed unless the model cache is populated. A GPU is not required by the current scripts. Runtime depends on hardware and is not benchmarked here.

```bash
git clone https://github.com/boumalaksiham/Product-NER-Attribute-Extraction.git
cd Product-NER-Attribute-Extraction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. If installation reports an unsupported wheel, check Python version and architecture before changing the pinned environment.

## Train and demonstrate

```bash
python train.py
python demo.py
```

Training writes a checkpoint, label maps, configuration and tokenizer under `models/saved/`. Train before running the demo; these artifacts are not included in the checkout. Keep checkpoint, tokenizer and label maps together.

The demo prints token predictions and extracted entities for example titles and supports interactive input. Example predictions are demonstrations, not an independent benchmark.

## Repository map

| File | Purpose |
|---|---|
| [data/dataset_builder.py](data/dataset_builder.py) | Labeled tokens and label mappings |
| [models/ner_model.py](models/ner_model.py) | DistilBERT token classifier and extraction helpers |
| [train.py](train.py) | Token alignment, training and span-level evaluation |
| [demo.py](demo.py) | Inference examples and interactive predictions |

## Results and limitations

Historical entity-specific F1 values are not retained as independent test evidence. There is no untouched final test set or representative real-listing benchmark. Rare entities, unseen brands, punctuation, subword alignment, and labels absent from a split can materially affect results.

Before publishing a final result, separate train/validation/test data by product, reload the best checkpoint, evaluate once on the test set, and save per-entity precision/recall/F1 plus support counts and representative errors. Also evaluate BIO consistency and long-title truncation.

The modified training script passes Python syntax compilation. Training has not been rerun; existing artifacts predate this repair. Rerun training to generate the new selected-checkpoint validation report.

Checkpoint-selection regression checks: `python -m unittest discover -s tests -v`. These execute the trainer’s selection/reload statements with controlled epoch scores, including zero-score ties and a best epoch before the final epoch. They passed without model downloads; they do not test learned-model quality.
