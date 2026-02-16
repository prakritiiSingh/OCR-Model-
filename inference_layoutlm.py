# inference_layoutlm.py

import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import pytesseract
import json
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------
# LOAD TRAINED MODEL
# --------------------------
MODEL_DIR = "outputs/model"

processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

# --------------------------
# OCR + BBOX EXTRACTION
# --------------------------
def extract_tokens_and_bboxes(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    tokens = []
    bboxes = []

    n = len(ocr_data["text"])
    for i in range(n):
        word = ocr_data["text"][i].strip()
        if word == "":
            continue

        x = ocr_data["left"][i]
        y = ocr_data["top"][i]
        w = ocr_data["width"][i]
        h = ocr_data["height"][i]

        x0, y0, x1, y1 = x, y, x + w, y + h

        # Normalize to 0–1000
        nx0 = int((x0 / width) * 1000)
        ny0 = int((y0 / height) * 1000)
        nx1 = int((x1 / width) * 1000)
        ny1 = int((y1 / height) * 1000)

        tokens.append(word)
        bboxes.append([nx0, ny0, nx1, ny1])

    return image, tokens, bboxes


# --------------------------
# PREDICT LABELS
# --------------------------
def run_inference(image_path):
    image, tokens, bboxes = extract_tokens_and_bboxes(image_path)

    encoding = processor(
        images=image,
        text=tokens,
        boxes=bboxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            bbox=encoding["bbox"],
            pixel_values=encoding["pixel_values"]
        )

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    id2label = model.config.id2label

    # Build final word-label mapping
    result = []
    for token, bbox, pred_id in zip(tokens, bboxes, predictions[:len(tokens)]):
        label = id2label[pred_id]
        if label != "O":
            result.append({
                "text": token,
                "bbox": bbox,
                "label": label
            })

    return result


# --------------------------
# TEST
# --------------------------
if __name__ == "__main__":
    test_image = "data/sample_doc/images/Calibration_Certificate_-_PUN_-_01_page1.png"

    output = run_inference(test_image)

    print("\n===== MODEL OUTPUT =====")
    print(json.dumps(output, indent=2))
