import pytesseract
import json
import os
import re
from pathlib import Path
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGES_DIR = Path("data/sample_doc/images")
LABELS_DIR = Path("data/sample_doc/labels")
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def detect_document_type(text):
    t = text.lower()
    if "apfc" in t or "capacitor panel" in t:
        return "calibration_certificate"
    if "calibration certificate" in t:
        return "calibration_certificate"
    if "raw matl" in t or "test report" in t and "raw" in t:
        return "raw_material_test_report"
    if "invoice" in t:
        return "invoice"
    if "transformer" in t or "bdv" in t or "oil test" in t:
        return "transformer_test_report"
    return "unknown"


def extract_tables(text):
    rows = []
    for line in text.split("\n"):
        columns = [c.strip() for c in line.split() if c.strip()]
        if len(columns) > 2:
            rows.append(columns)
    return rows


def make_json(image_name, ocr_text, doc_type):
    base = {
        "source_image": image_name,
        "document_type": doc_type,
        "ocr_text": ocr_text,
        "entities": {},
        "tables": [],
        "remarks": ""
    }

    # Extract common fields
    lines = ocr_text.split("\n")

    # certificate number
    cert_no = ""
    for l in lines:
        if "cert" in l.lower() or "certificate" in l.lower():
            m = re.search(r"[A-Z0-9\-\/]{6,}", l)
            if m:
                cert_no = m.group()
                break
    if cert_no:
        base["entities"]["certificate_no"] = cert_no

    # customer name
    for l in lines:
        if "sharada" in l.lower():
            base["entities"]["customer_name"] = "SHARADA INDUSTRIES"

    # dates
    date_list = re.findall(r"\d{1,2}[-\/\.]\d{1,2}[-\/\.]\d{2,4}", ocr_text)
    if date_list:
        base["entities"]["dates"] = date_list

    # table extraction
    base["tables"] = extract_tables(ocr_text)

    # remarks extraction
    for l in lines:
        if "remark" in l.lower():
            base["remarks"] = l

    return base


def main():
    images = sorted([p for p in IMAGES_DIR.glob("*.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    print(f"Found {len(images)} images.")
    print("Generating JSON labels...\n")

    for img_path in images:
        print(f"Processing: {img_path.name}")

        img = Image.open(img_path)
        ocr_text = pytesseract.image_to_string(img)

        doc_type = detect_document_type(ocr_text)

        json_obj = make_json(img_path.name, ocr_text, doc_type)

        out_path = LABELS_DIR / (img_path.stem + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2, ensure_ascii=False)

    print("\n--------------------------------------")
    print("JSON LABEL GENERATION COMPLETED")
    print("--------------------------------------")


if __name__ == "__main__":
    main()
