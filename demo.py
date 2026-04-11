"""
Demo — Product NER
===================
Loads trained model and extracts attributes from any product title.

Usage:
    python demo.py
"""

import os, sys, json
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.ner_model import ProductNERModel, ProductNERPredictor

SAVE_DIR = "models/saved"


def load_model():
    if not os.path.exists(f"{SAVE_DIR}/best_model.pt"):
        print("❌ No saved model. Run train.py first.")
        sys.exit(1)

    with open(f"{SAVE_DIR}/label_maps.json") as f:
        maps = json.load(f)
    with open(f"{SAVE_DIR}/config.json") as f:
        config = json.load(f)

    label_to_id = maps["label_to_id"]
    id_to_label = {int(k): v for k, v in maps["id_to_label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(f"{SAVE_DIR}/tokenizer")
    model = ProductNERModel(
        num_labels=len(label_to_id),
        model_name=config["model_name"]
    )
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_model.pt", map_location="cpu"))

    return ProductNERPredictor(model, tokenizer, id_to_label, config["max_length"])


def print_result(result: dict):
    print("\n" + "─" * 62)
    print(f"  Title: {result['title']}")
    print()

    # Token-level labels
    print("  Token-level predictions:")
    for token, label in zip(result["tokens"], result["labels"]):
        if label != "O":
            print(f"    {token:<20} → {label}")
        else:
            print(f"    {token:<20}   (O)")

    print()
    print("  Extracted Entities:")
    if result["entities"]:
        for entity_type, values in result["entities"].items():
            for val in values:
                print(f"    {entity_type:<12} : {val}")
    else:
        print("    (none found)")
    print("─" * 62)


def run_presets(predictor):
    print("\n" + "=" * 62)
    print("  PRESET EXAMPLES")
    print("=" * 62)

    titles = [
        "Apple iPhone 14 Pro 256GB Deep Purple Unlocked",
        "Samsung Galaxy S24 Ultra 512GB Titanium Black 5G Unlocked",
        "Nike Air Max 90 White Men's Size 10",
        "Sony WH-1000XM5 Wireless Noise Cancelling Headphones Black",
        "Levi's 501 Original Fit Jeans Men's 32x32 Dark Stonewash",
        "KitchenAid Artisan Stand Mixer 5Qt Empire Red",
        "Apple AirPods Pro 2nd Generation MagSafe White",
        "Adidas Ultraboost 22 Running Shoes White Men's Size 11",
        "Instant Pot Duo 7-in-1 Electric Pressure Cooker 6Qt",
        "LEGO Star Wars Millennium Falcon 75192 4016 Pieces",
    ]

    for title in titles:
        result = predictor.predict(title)
        print_result(result)


def interactive_mode(predictor):
    print("\n" + "=" * 62)
    print("  INTERACTIVE MODE")
    print("  (Press Ctrl+C to quit)")
    print("=" * 62)

    while True:
        try:
            print()
            title = input("Product Title: ").strip()
            if not title:
                continue
            result = predictor.predict(title)
            print_result(result)
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def main():
    print("=" * 62)
    print("  PRODUCT NER DEMO")
    print("  Fine-tuned DistilBERT — Attribute Extraction")
    print("=" * 62)
    print("\nLoading model...")
    predictor = load_model()
    print("Ready!")

    run_presets(predictor)
    interactive_mode(predictor)


if __name__ == "__main__":
    main()
