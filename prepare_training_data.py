#!/usr/bin/env python3
"""
prepare_training_data.py

Converts your structured JSON labels (no bboxes) + images -> token-level BIO labels
suitable for LayoutLMv3 token-classification training.

Usage (single line, PowerShell or cmd):
python prepare_training_data.py --images_dir data/sample_doc/images --labels_dir data/sample_doc/labels --out_dir dataset_prepared

Requirements:
- pytesseract must be installed and tesseract binary must be in PATH (you already set it).
- Pillow, tqdm, numpy installed.
"""
import os
import json
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image
import pytesseract
import re
from utils import ensure_dir, bbox_from_xywh, bbox_iou, load_json

# >>> Ensure tesseract path (windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- Utilities ----------
def flatten_json(prefix, obj, out):
    """
    Flatten nested JSON into key -> string value.
    For lists (tables), flatten each item field to key like `table.field` (no index)
    but include index as suffix when needed to disambiguate: table[0].field -> table.field (we will keep all rows under same key)
    We'll emit multiple values for same flattened key (store list).
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_pref = f"{prefix}.{k}" if prefix else k
            flatten_json(new_pref, v, out)
    elif isinstance(obj, list):
        # For lists, flatten each element. We'll store multiple values under same key.
        for item in obj:
            flatten_json(prefix, item, out)
    else:
        # primitive value - store as string (strip)
        s = "" if obj is None else str(obj)
        s = s.strip()
        if s == "":
            return
        out.setdefault(prefix, []).append(s)

def parse_annotation_json_to_fields(js):
    """
    Accepts your structured JSON and flattens into:
    { 'certificate_no': ['val'], 'instrument.nomenclature': ['APFC PANEL'], 'measurement_results.std_value': ['R-Y Phase IR ...', ...], ... }
    """
    out = {}
    flatten_json("", js, out)
    # remove empty keys
    return out

# OCR -> tokens + bboxes
def ocr_tokens_with_boxes(image_path):
    img = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    tokens = []
    for i, txt in enumerate(data['text']):
        txt = (txt or "").strip()
        if txt == "":
            continue
        x = int(data['left'][i])
        y = int(data['top'][i])
        w = int(data['width'][i])
        h = int(data['height'][i])
        bbox = [x, y, x + w, y + h]
        tokens.append({"text": txt, "bbox": bbox})
    return tokens, img.size

def match_by_bbox(tokens, value_bbox):
    """
    Return list of token indices whose IoU with value_bbox > threshold
    """
    matches = []
    for i, t in enumerate(tokens):
        if bbox_iou(t["bbox"], value_bbox) > 0.35:
            matches.append(i)
    if not matches:
        for i, t in enumerate(tokens):
            if bbox_iou(t["bbox"], value_bbox) > 0.05:
                matches.append(i)
    return matches

def find_subsequence_indices(tokens_texts, value_text):
    """
    Find value_text (string) inside tokens_texts sequence (concatenated)
    Return start_index and length (number of tokens matched) or None.
    Uses whitespace-normalized substring match; fallback to greedy token-match by word start.
    """
    if not value_text:
        return None
    cleaned_val = re.sub(r"\s+", " ", value_text.strip()).lower()
    joined = " ".join(tokens_texts).lower()
    idx = joined.find(cleaned_val)
    if idx != -1:
        # count words before
        before = joined[:idx].strip()
        start_token = len(before.split()) if before else 0
        nwords = len(cleaned_val.split())
        return start_token, nwords
    # fallback: token-level fuzzy match (match first word start)
    val_words = cleaned_val.split()
    if not val_words:
        return None
    first = val_words[0][:4]
    for i, tok in enumerate(tokens_texts):
        if tok.lower().startswith(first):
            # try to match sequentially
            matches = 0
            for k, w in enumerate(val_words):
                if i + k < len(tokens_texts) and tokens_texts[i + k].lower().startswith(w[:4]):
                    matches += 1
                else:
                    break
            if matches > 0:
                return i, matches
    return None

def assign_bio_labels(tokens, fields_dict):
    """
    tokens: list of {"text","bbox"}
    fields_dict: flattened dict key -> list of string values (multiple rows map to same key)
    returns labels list same length as tokens: O or B-<key> or I-<key>
    Strategy:
      1) If any field value is short and appears multiple times (tables), we still mark all occurrences.
      2) Try bbox match if bbox present in JSON (we detect bbox-looking strings? not applicable here).
      3) Fallback substring alignment.
    """
    n = len(tokens)
    labels = ["O"] * n
    token_texts = [t["text"] for t in tokens]

    # order fields by length of value descending (longer first to avoid partial matches)
    items = []
    for k, vals in fields_dict.items():
        for v in vals:
            items.append((k, v))
    items.sort(key=lambda x: len(x[1]), reverse=True)

    for key, val in items:
        if not val:
            continue
        # try to find as exact bbox text in tokens (unlikely since JSON had no bboxes)
        # try substring match
        found = find_subsequence_indices(token_texts, val)
        if found:
            start, length = found
            # assign BIO avoiding overwriting existing B labels (skip if already labeled)
            # but if overlapping existing labels, skip that match
            conflict = False
            for idx in range(start, min(start+length, n)):
                if labels[idx] != "O":
                    conflict = True
                    break
            if conflict:
                # try to continue (skip this occurrence)
                continue
            labels[start] = f"B-{key}"
            for j in range(start+1, min(start+length, n)):
                labels[j] = f"I-{key}"
            continue
        # else, try token-level approximate matching of each word in val sequentially
        # already handled by fuzzy in find_subsequence_indices; if didn't match, attempt per-token containment
        val_parts = val.split()
        # greedy search for sequence where each token contains val_parts[k] as substring (small)
        for i in range(n):
            ok = True
            for k_word, w in enumerate(val_parts):
                if i + k_word >= n:
                    ok = False; break
                if w.lower()[:3] not in token_texts[i + k_word].lower()[:len(w[:3])]:
                    ok = False; break
            if ok:
                # assign labels if zone free
                conflict = any(labels[i + kk] != "O" for kk in range(len(val_parts)))
                if conflict:
                    continue
                labels[i] = f"B-{key}"
                for kk in range(1, len(val_parts)):
                    labels[i+kk] = f"I-{key}"
                break
    return labels

def collect_label_list(all_label_lists):
    s = set()
    for lab in all_label_lists:
        for t in lab:
            s.add(t)
    # ensure 'O' exists and is first
    s.discard("O")
    ordered = ["O"] + sorted(list(s))
    return ordered

# ---------- Main conversion ----------
def convert_dir(images_dir, labels_dir, out_dir, max_words=4096):
    ensure_dir(out_dir)
    image_files = sorted([p for p in glob(os.path.join(images_dir, "*")) if p.lower().endswith((".png",".jpg",".jpeg"))])
    examples = []
    all_label_lists = []
    for img_path in tqdm(image_files, desc="Processing images"):
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base + ".json")
        fields = {}
        if os.path.exists(label_path):
            try:
                js = load_json(label_path)
                fields = parse_annotation_json_to_fields(js)
            except Exception as e:
                print("Failed parse json:", label_path, e)
                fields = {}
        else:
            # no JSON for this image: skip or create empty
            fields = {}
        # OCR tokens
        tokens, img_size = ocr_tokens_with_boxes(img_path)
        if not tokens:
            print("No OCR tokens for", img_path)
            continue
        # truncate tokens if needed (keeps beginning)
        if len(tokens) > max_words:
            tokens = tokens[:max_words]
        labels = assign_bio_labels(tokens, fields)
        # collect
        token_texts = [t["text"] for t in tokens]
        bboxes = [t["bbox"] for t in tokens]
        examples.append({
            "image_path": os.path.abspath(img_path),
            "width": img_size[0],
            "height": img_size[1],
            "tokens": token_texts,
            "bboxes": bboxes,
            "labels": labels
        })
        all_label_lists.append(labels)
    # create label list meta
    label_list = collect_label_list(all_label_lists)
    meta = {"label_list": label_list, "n_examples": len(examples)}
    # write dataset.jsonl
    out_data = os.path.join(out_dir, "dataset.jsonl")
    with open(out_data, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Wrote", out_data)
    print("Meta:", meta)
    return out_data, os.path.join(out_dir, "meta.json")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_words", type=int, default=4096)
    args = parser.parse_args()
    convert_dir(args.images_dir, args.labels_dir, args.out_dir, args.max_words)
