# train_layoutlm.py
"""
Training Script for LayoutLMv3 Token Classification
---------------------------------------------------

Run:

python train_layoutlm.py \
  --dataset_json dataset_prepared/dataset.jsonl \
  --meta_json dataset_prepared/meta.json \
  --output_dir outputs/layoutlmv3 \
  --per_device_train_batch_size 2 \
  --num_train_epochs 6
"""

import argparse
import json
import os
from PIL import Image
import torch
from datasets import Dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    default_data_collator
)

# For image loading (safe on Windows)
Image.MAX_IMAGE_PIXELS = None

###############################
# TESSERACT
###############################
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


########################################################
# 1. LOAD JSONL
########################################################
def load_jsonl_dataset(jsonl_path):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows


########################################################
# 2. HF DATASET CONVERSION (lightweight)
########################################################
def convert_to_hf_dataset(entries):
    return Dataset.from_list(entries)


########################################################
# 3. TOKENIZE + ALIGN LABELS
########################################################
def tokenize_and_align_labels(example, processor, label_to_id):
    """
    FIXED: normalize bbox to 0–1000 range
    """
    image = Image.open(example["image_path"]).convert("RGB")
    width, height = image.size

    # Word-level labels
    words = example["tokens"]
    boxes = example["bboxes"]
    word_labels = example["labels"]

    # --- Normalize bounding boxes ---
    norm_bboxes = []
    for (x0, y0, x1, y1) in boxes:
        nx0 = int((x0 / width) * 1000)
        ny0 = int((y0 / height) * 1000)
        nx1 = int((x1 / width) * 1000)
        ny1 = int((y1 / height) * 1000)

        nx0 = max(0, min(1000, nx0))
        ny0 = max(0, min(1000, ny0))
        nx1 = max(0, min(1000, nx1))
        ny1 = max(0, min(1000, ny1))

        norm_bboxes.append([nx0, ny0, nx1, ny1])

    encoding = processor(
        images=image,
        text=words,
        boxes=norm_bboxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"][0]
    offset_mapping = encoding["offset_mapping"][0]

    labels = []
    word_idx = 0
    for (start, end) in offset_mapping.tolist():
        if start == end:
            labels.append(-100)
        else:
            if word_idx < len(word_labels):
                labels.append(label_to_id[word_labels[word_idx]])
                word_idx += 1
            else:
                labels.append(label_to_id["O"])

    encoding["labels"] = torch.tensor(labels, dtype=torch.long)
    encoding.pop("offset_mapping")

    encoding["pixel_values"] = encoding["pixel_values"][0].numpy()
    encoding["input_ids"] = encoding["input_ids"][0].tolist()
    encoding["attention_mask"] = encoding["attention_mask"][0].tolist()
    encoding["bbox"] = encoding["bbox"][0].tolist()

    return encoding


########################################################
# 4. MAIN
########################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--meta_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()

    ###############################
    # Load meta
    ###############################
    with open(args.meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label_list = meta["label_list"]
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for i, l in enumerate(label_list)}

    ###############################
    # Load dataset entries
    ###############################
    entries = load_jsonl_dataset(args.dataset_json)
    print("Loaded entries:", len(entries))

    hf_dataset = convert_to_hf_dataset(entries)

    ###############################
    # Load processor
    ###############################
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False
    )

    ###############################
    # Apply tokenizer mapping
    ###############################
    def map_fn(example):
        return tokenize_and_align_labels(example, processor, label_to_id)

    dataset_tokenized = hf_dataset.map(
        map_fn,
        remove_columns=hf_dataset.column_names
    )

    ###############################
    # Torch formatting
    ###############################
    dataset_tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "bbox", "pixel_values", "labels"]
    )

    ###############################
    # Load model
    ###############################
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id
    )

    ###############################
    # Training Arguments (safe version)
    ###############################
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2
    )

    ###############################
    # Trainer
    ###############################
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tokenized,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator
    )

    ###############################
    # TRAIN
    ###############################
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nTraining completed successfully!")
    print("Model saved to:", args.output_dir)


########################################################
if __name__ == "__main__":
    main()
