"""
NER Dataset Builder — Product Title Attribute Extraction
=========================================================

NER (Named Entity Recognition) works by labeling every single word (token)
in a sentence with a tag that says what kind of entity it is.

We use the BIO tagging scheme:
    B- = Beginning of an entity
    I- = Inside (continuation) of an entity
    O  = Outside (not an entity)

Example:
    "Apple  iPhone  14   Pro  256GB  Deep    Purple  Unlocked"
     B-BRD  B-MDL   I-MDL I-MDL B-STR B-CLR   I-CLR   B-NET

Entity types we extract:
    BRD = Brand        (Nike, Apple, Sony, Samsung...)
    MDL = Model        (iPhone 14 Pro, Air Max 90, WH-1000XM5...)
    CLR = Color        (Black, White, Deep Purple, Space Gray...)
    STR = Storage      (256GB, 512GB, 1TB...)
    SIZ = Size         (Size 10, 32x32, 5Qt, 6mm...)
    MAT = Material     (Leather, Stainless Steel, Aluminum...)
    NET = Network      (Unlocked, 5G, AT&T...)
    GEN = Generation   (2nd Generation, Gen 12...)
    CNT = Count/Qty    (3-Pack, 4-Piece, 10-Pack...)
"""

# ─────────────────────────────────────────────────────────────
# LABELED DATASET
# Each entry is a list of (word, label) tuples
# ─────────────────────────────────────────────────────────────

LABELED_SENTENCES = [

    # ── Apple iPhone ──────────────────────────────────────────
    [("Apple", "B-BRD"), ("iPhone", "B-MDL"), ("14", "I-MDL"), ("Pro", "I-MDL"),
     ("256GB", "B-STR"), ("Deep", "B-CLR"), ("Purple", "I-CLR"), ("Unlocked", "B-NET")],

    [("Apple", "B-BRD"), ("iPhone", "B-MDL"), ("15", "I-MDL"), ("Pro", "I-MDL"), ("Max", "I-MDL"),
     ("512GB", "B-STR"), ("Natural", "B-CLR"), ("Titanium", "I-CLR"), ("Unlocked", "B-NET")],

    [("Apple", "B-BRD"), ("iPhone", "B-MDL"), ("14", "I-MDL"),
     ("128GB", "B-STR"), ("Midnight", "B-CLR"), ("Black", "I-CLR"), ("Unlocked", "B-NET")],

    [("Apple", "B-BRD"), ("iPhone", "B-MDL"), ("13", "I-MDL"), ("mini", "I-MDL"),
     ("256GB", "B-STR"), ("Starlight", "B-CLR"), ("Unlocked", "B-NET")],

    # ── Samsung Galaxy ────────────────────────────────────────
    [("Samsung", "B-BRD"), ("Galaxy", "B-MDL"), ("S24", "I-MDL"), ("Ultra", "I-MDL"),
     ("512GB", "B-STR"), ("Titanium", "B-CLR"), ("Black", "I-CLR"), ("5G", "B-NET"), ("Unlocked", "I-NET")],

    [("Samsung", "B-BRD"), ("Galaxy", "B-MDL"), ("S23", "I-MDL"), ("Plus", "I-MDL"),
     ("256GB", "B-STR"), ("Phantom", "B-CLR"), ("Black", "I-CLR"), ("5G", "B-NET")],

    [("Samsung", "B-BRD"), ("Galaxy", "B-MDL"), ("A54", "I-MDL"),
     ("128GB", "B-STR"), ("Awesome", "B-CLR"), ("Graphite", "I-CLR"), ("5G", "B-NET")],

    # ── Google Pixel ──────────────────────────────────────────
    [("Google", "B-BRD"), ("Pixel", "B-MDL"), ("8", "I-MDL"), ("Pro", "I-MDL"),
     ("128GB", "B-STR"), ("Obsidian", "B-CLR"), ("Unlocked", "B-NET")],

    [("Google", "B-BRD"), ("Pixel", "B-MDL"), ("7a", "I-MDL"),
     ("128GB", "B-STR"), ("Charcoal", "B-CLR"), ("5G", "B-NET"), ("Unlocked", "I-NET")],

    # ── Apple MacBook ─────────────────────────────────────────
    [("Apple", "B-BRD"), ("MacBook", "B-MDL"), ("Pro", "I-MDL"), ("14-inch", "I-MDL"),
     ("M3", "I-MDL"), ("Pro", "I-MDL"), ("512GB", "B-STR"), ("Space", "B-CLR"), ("Black", "I-CLR")],

    [("Apple", "B-BRD"), ("MacBook", "B-MDL"), ("Air", "I-MDL"), ("15-inch", "I-MDL"),
     ("M3", "I-MDL"), ("256GB", "B-STR"), ("Midnight", "B-CLR")],

    # ── Sony Headphones ───────────────────────────────────────
    [("Sony", "B-BRD"), ("WH-1000XM5", "B-MDL"), ("Wireless", "O"),
     ("Noise", "O"), ("Cancelling", "O"), ("Headphones", "O"), ("Black", "B-CLR")],

    [("Sony", "B-BRD"), ("WH-1000XM4", "B-MDL"), ("Wireless", "O"),
     ("Headphones", "O"), ("Silver", "B-CLR")],

    [("Sony", "B-BRD"), ("WF-1000XM5", "B-MDL"), ("True", "O"),
     ("Wireless", "O"), ("Earbuds", "O"), ("Black", "B-CLR")],

    # ── Bose ──────────────────────────────────────────────────
    [("Bose", "B-BRD"), ("QuietComfort", "B-MDL"), ("45", "I-MDL"),
     ("Bluetooth", "O"), ("Headphones", "O"), ("White", "B-CLR"), ("Smoke", "I-CLR")],

    [("Bose", "B-BRD"), ("SoundLink", "B-MDL"), ("Flex", "I-MDL"),
     ("Bluetooth", "O"), ("Speaker", "O"), ("Stone", "B-CLR"), ("Blue", "I-CLR")],

    # ── Nike Shoes ────────────────────────────────────────────
    [("Nike", "B-BRD"), ("Air", "B-MDL"), ("Max", "I-MDL"), ("90", "I-MDL"),
     ("White", "B-CLR"), ("Men's", "O"), ("Size", "B-SIZ"), ("10", "I-SIZ")],

    [("Nike", "B-BRD"), ("Air", "B-MDL"), ("Force", "I-MDL"), ("1", "I-MDL"), ("Low", "I-MDL"),
     ("White", "B-CLR"), ("Men's", "O"), ("Size", "B-SIZ"), ("9", "I-SIZ")],

    [("Nike", "B-BRD"), ("Air", "B-MDL"), ("Jordan", "I-MDL"), ("1", "I-MDL"),
     ("Retro", "I-MDL"), ("High", "I-MDL"), ("OG", "I-MDL"),
     ("Black", "B-CLR"), ("White", "I-CLR"), ("Size", "B-SIZ"), ("11", "I-SIZ")],

    [("Nike", "B-BRD"), ("Dri-FIT", "B-MDL"), ("Training", "I-MDL"),
     ("T-Shirt", "O"), ("Black", "B-CLR"), ("Large", "B-SIZ")],

    # ── Adidas ────────────────────────────────────────────────
    [("Adidas", "B-BRD"), ("Ultraboost", "B-MDL"), ("22", "I-MDL"),
     ("Running", "O"), ("Shoes", "O"), ("White", "B-CLR"),
     ("Men's", "O"), ("Size", "B-SIZ"), ("11", "I-SIZ")],

    [("Adidas", "B-BRD"), ("Stan", "B-MDL"), ("Smith", "I-MDL"),
     ("White", "B-CLR"), ("Green", "I-CLR"), ("Size", "B-SIZ"), ("10", "I-SIZ")],

    # ── Levi's Jeans ──────────────────────────────────────────
    [("Levi's", "B-BRD"), ("501", "B-MDL"), ("Original", "I-MDL"), ("Fit", "I-MDL"),
     ("Jeans", "O"), ("Men's", "O"), ("32x32", "B-SIZ"), ("Dark", "B-CLR"), ("Stonewash", "I-CLR")],

    [("Levi's", "B-BRD"), ("511", "B-MDL"), ("Slim", "I-MDL"), ("Fit", "I-MDL"),
     ("Jeans", "O"), ("Men's", "O"), ("30x30", "B-SIZ"), ("Medium", "B-CLR"), ("Indigo", "I-CLR")],

    # ── KitchenAid ────────────────────────────────────────────
    [("KitchenAid", "B-BRD"), ("Artisan", "B-MDL"), ("Stand", "I-MDL"), ("Mixer", "I-MDL"),
     ("5Qt", "B-SIZ"), ("Empire", "B-CLR"), ("Red", "I-CLR")],

    [("KitchenAid", "B-BRD"), ("Classic", "B-MDL"), ("Stand", "I-MDL"), ("Mixer", "I-MDL"),
     ("4.5Qt", "B-SIZ"), ("Aqua", "B-CLR"), ("Sky", "I-CLR")],

    # ── Instant Pot ───────────────────────────────────────────
    [("Instant", "B-BRD"), ("Pot", "I-BRD"), ("Duo", "B-MDL"), ("7-in-1", "I-MDL"),
     ("Electric", "O"), ("Pressure", "O"), ("Cooker", "O"), ("6Qt", "B-SIZ")],

    [("Instant", "B-BRD"), ("Pot", "I-BRD"), ("Pro", "B-MDL"), ("Plus", "I-MDL"),
     ("9-in-1", "I-MDL"), ("8Qt", "B-SIZ"), ("Stainless", "B-MAT"), ("Steel", "I-MAT")],

    # ── Le Creuset ────────────────────────────────────────────
    [("Le", "B-BRD"), ("Creuset", "I-BRD"), ("Signature", "B-MDL"),
     ("Cast", "B-MAT"), ("Iron", "I-MAT"), ("Dutch", "O"), ("Oven", "O"),
     ("5.5Qt", "B-SIZ"), ("Flame", "B-CLR")],

    # ── Apple AirPods ─────────────────────────────────────────
    [("Apple", "B-BRD"), ("AirPods", "B-MDL"), ("Pro", "I-MDL"),
     ("2nd", "B-GEN"), ("Generation", "I-GEN"), ("MagSafe", "O"), ("White", "B-CLR")],

    [("Apple", "B-BRD"), ("AirPods", "B-MDL"), ("3rd", "B-GEN"), ("Generation", "I-GEN"),
     ("Lightning", "O"), ("Case", "O"), ("White", "B-CLR")],

    # ── Dyson ─────────────────────────────────────────────────
    [("Dyson", "B-BRD"), ("V15", "B-MDL"), ("Detect", "I-MDL"),
     ("Absolute", "I-MDL"), ("Cordless", "O"), ("Vacuum", "O"), ("Yellow", "B-CLR")],

    [("Dyson", "B-BRD"), ("V11", "B-MDL"), ("Torque", "I-MDL"), ("Drive", "I-MDL"),
     ("Cordless", "O"), ("Vacuum", "O"), ("Blue", "B-CLR")],

    # ── LEGO ──────────────────────────────────────────────────
    [("LEGO", "B-BRD"), ("Star", "B-MDL"), ("Wars", "I-MDL"),
     ("Millennium", "I-MDL"), ("Falcon", "I-MDL"), ("75192", "I-MDL"),
     ("4016", "B-CNT"), ("Pieces", "I-CNT")],

    [("LEGO", "B-BRD"), ("Technic", "B-MDL"), ("Bugatti", "I-MDL"),
     ("Chiron", "I-MDL"), ("42083", "I-MDL"),
     ("3599", "B-CNT"), ("Pieces", "I-CNT")],

    # ── Vitamix ───────────────────────────────────────────────
    [("Vitamix", "B-BRD"), ("5200", "B-MDL"), ("Professional", "I-MDL"),
     ("Blender", "O"), ("64oz", "B-SIZ"), ("Black", "B-CLR")],

    # ── All-Clad ──────────────────────────────────────────────
    [("All-Clad", "B-BRD"), ("D3", "B-MDL"), ("Stainless", "B-MAT"),
     ("Steel", "I-MAT"), ("Cookware", "O"), ("Set", "O"),
     ("10-Piece", "B-CNT")],

    # ── Hanes ─────────────────────────────────────────────────
    [("Hanes", "B-BRD"), ("ComfortSoft", "B-MDL"), ("Crewneck", "I-MDL"),
     ("T-Shirt", "O"), ("White", "B-CLR"), ("3-Pack", "B-CNT"), ("Large", "B-SIZ")],

    # ── Canon ─────────────────────────────────────────────────
    [("Canon", "B-BRD"), ("EOS", "B-MDL"), ("R5", "I-MDL"),
     ("Mirrorless", "O"), ("Camera", "O"), ("Body", "O"), ("Only", "O")],

    [("Canon", "B-BRD"), ("EOS", "B-MDL"), ("R6", "I-MDL"), ("Mark", "I-MDL"), ("II", "I-MDL"),
     ("Mirrorless", "O"), ("Camera", "O"), ("Body", "O")],

    # ── Dell ──────────────────────────────────────────────────
    [("Dell", "B-BRD"), ("XPS", "B-MDL"), ("15", "I-MDL"),
     ("Intel", "O"), ("Core", "O"), ("i9", "O"),
     ("32GB", "B-STR"), ("RAM", "O"), ("1TB", "B-STR"), ("SSD", "O"),
     ("Silver", "B-CLR")],

    # ── Manduka ───────────────────────────────────────────────
    [("Manduka", "B-BRD"), ("PRO", "B-MDL"), ("Yoga", "O"), ("Mat", "O"),
     ("6mm", "B-SIZ"), ("Long", "I-SIZ"), ("Black", "B-CLR")],

    # ── Bowflex ───────────────────────────────────────────────
    [("Bowflex", "B-BRD"), ("SelectTech", "B-MDL"), ("552", "I-MDL"),
     ("Adjustable", "O"), ("Dumbbell", "O"), ("Pair", "B-CNT")],

    # ── REI ───────────────────────────────────────────────────
    [("REI", "B-BRD"), ("Co-op", "I-BRD"), ("Passage", "B-MDL"), ("2", "I-MDL"),
     ("Person", "O"), ("Camping", "O"), ("Tent", "O"), ("Green", "B-CLR")],

]


def get_unique_labels(sentences):
    """Returns sorted list of all unique labels in the dataset."""
    labels = set()
    for sentence in sentences:
        for _, label in sentence:
            labels.add(label)
    return sorted(labels)


def get_label_maps(sentences):
    """Creates label → integer and integer → label mappings."""
    unique_labels = get_unique_labels(sentences)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return label_to_id, id_to_label


def print_dataset_stats(sentences):
    """Prints dataset statistics."""
    label_counts = {}
    total_tokens = 0
    for sentence in sentences:
        for _, label in sentence:
            label_counts[label] = label_counts.get(label, 0) + 1
            total_tokens += 1

    print(f"Total sentences : {len(sentences)}")
    print(f"Total tokens    : {total_tokens}")
    print(f"Avg tokens/sent : {total_tokens / len(sentences):.1f}")
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        bar = "█" * (count // 2)
        print(f"  {label:<8} {count:>3}  {bar}")


if __name__ == "__main__":
    print_dataset_stats(LABELED_SENTENCES)
