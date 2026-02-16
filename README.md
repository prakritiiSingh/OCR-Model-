# OCR-Model-
This project takes scanned documents or images as input, runs OCR to extract text and layout, and feeds the extracted content into a document-understanding model (for example, a LayoutLM-based or transformer-based model) to perform tasks such as key-value extraction, classification, or table parsing.


# OCR + Document Understanding — README

> This README documents the OCR + model pipeline you built. It contains a high-level flowchart, component descriptions, installation & usage instructions, pointers for evaluation, and tips for improving accuracy.

---

## Project overview

This project takes scanned documents or images as input, runs OCR to extract text and layout, and feeds the extracted content into a document-understanding model (for example, a LayoutLM-based or transformer-based model) to perform tasks such as key-value extraction, classification, or table parsing.

## High-level flowchart

```mermaid
flowchart TD
  A[Input: Scanned image / PDF] --> B[Preprocessing]
  B --> B1[Image normalization: resize / deskew / denoise]
  B --> B2[Page segmentation: detect regions, columns]
  B2 --> C[OCR Engine]
  C --> C1[Text + bounding boxes]
  C1 --> D[Post-OCR processing]
  D --> D1[Tokenization & alignment with boxes]
  D --> D2[Spellcheck / regex cleaning]
  D --> E[Layout-aware model]
  E --> E1[Feature encoding (text + layout + visual)]
  E1 --> E2[Task heads: K-V extraction / Table extraction / Classification]
  E2 --> F[Postprocessing]
  F --> F1[Confidence filtering / normalization]
  F --> G[Output: JSON / CSV / UI (Streamlit)]
  subgraph Optional
    H[Human-in-the-loop / Validation UI]
    G --> H
    H --> I[Label updates -> Retrain]
  end
```

## Components & responsibilities

* **Input**: Single-page images, multi-page PDFs, or photos taken with phones.
* **Preprocessing**: Image cleanup (denoise, binarize), deskew, resolution adjustment, and segmentation (detect paragraphs, tables, forms). Use OpenCV or PIL for most ops.
* **OCR engine**: Tesseract, Google Vision OCR, or a layout-aware OCR pipeline (e.g., LayoutLM inference step that extracts token boxes). The OCR must return text with bounding-box coordinates.
* **Post-OCR processing**: Tokenize OCR output, apply normalization (unicode normalizing), simple heuristics (remove headers/footers), and regex-based extraction for structured fields (dates, amounts, IDs).
* **Layout-aware model**: A transformer-based model that consumes (text tokens + bounding boxes + optional visual embeddings). Typical choices: LayoutLM / LayoutLMv2 / LayoutLMv3, Donut, or your custom model backed by Mistral/OpenAI embeddings for text + layout.
* **Task heads**: Heads for key-value pair detection, table recognition (cell detection + structure parsing), and document classification.
* **Postprocessing & Output**: Merge overlapping predictions, normalize formats (ISO dates), write final JSON/CSV for downstream consumption.

## Folder structure (suggested)

```
ocr-model-project/
├─ data/                 # raw and preprocessed images & PDFs
├─ notebooks/            # experiments and EDA
├─ src/
│  ├─ preprocessing.py
│  ├─ ocr_engine.py
│  ├─ postprocess.py
│  ├─ model/
│  │  ├─ inference_layoutlm.py
│  │  └─ train.py
│  └─ app.py             # Streamlit / FastAPI UI
├─ models/               # trained weights, tokenizer
├─ tests/
├─ requirements.txt
└─ README.md
```

## Installation

```bash
# create env
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Typical requirements** (example):

```
opencv-python
pillow
pytesseract
transformers
torch
numpy
pandas
streamlit
pdf2image
```

## Running locally

Below are clear **How to Run** instructions for both OCR-only & full Model pipeline.

### 🔹 0) Start the virtual environment (Windows)

```bash
.venv\Scripts\activate
```

### 🔹 0) Start the virtual environment (Mac/Linux)

```bash
source .venv/bin/activate
```

### 1) Run OCR-only pipeline

```bash
python src/ocr_engine.py --input samples/invoice_001.png --output outputs/ocr_invoice_001.json
```

### 2) Run Full OCR + Model inference

```bash
python src/app.py --config config.yaml  # Streamlit or CLI wrapper
# or for a single-run inference
python src/model/inference_layoutlm.py --image samples/invoice_001.png --model models/layoutlm_best.pt
```

## Example output schema (JSON)

```json
{
  "document_id": "invoice_001",
  "pages": [
    {
      "page_num": 1,
      "lines": [
        {"text": "Invoice Date: 2025-10-29", "bbox": [10, 20, 300, 40]},
        {"text": "Total: INR 12345", "bbox": [500, 1200, 750, 1240]}
      ],
      "key_values": [
        {"key": "Invoice Date", "value": "2025-10-29", "confidence": 0.98},
        {"key": "Total", "value": "12345", "confidence": 0.95}
      ]
    }
  ]
}
```

## Training & fine-tuning notes

1. Prepare labeled data with token-level labels and bounding boxes (e.g., CoNLL format extended with boxes or JSON label files).
2. Use a layout-aware model that supports (tokens + boxes). For fine-tuning: freeze some transformer layers initially, use a low learning rate (e.g., 1e-5 to 5e-5), and monitor entity-level F1.
3. Augment data with synthetic noise: rotate small angles, vary DPI, add blur, and vary lighting.
4. If table extraction is critical, add table-structured labels (cell-level) and use a model that outputs cell coordinates + structure.

## Evaluation metrics

* **OCR accuracy**: Character Error Rate (CER) and Word Error Rate (WER).
* **Entity extraction**: Precision, Recall, F1 at the field level.
* **Table parsing**: Cell-level F1 and structural accuracy (correct adjacency & order).

## Tips to improve performance

* Improve preprocessing — correct skew and remove background noise.
* Use confidence thresholds to drop low-quality OCR tokens before model input.
* Merge token-level predictions with simple heuristics (e.g., nearest neighbor) to form multi-word field values.
* Active learning: present low-confidence examples in the validation UI for human labeling and incremental retraining.

## Troubleshooting

* If OCR output is garbled, check image DPI (prefer 300+), ensure proper binarization and lighting.
* If model misses fields, review label consistency (labels in training must match expected output exactly).
* For long documents, process page-by-page to avoid memory blowups.

## Quick reference commands

### 🔥 Run the full Streamlit App (Main UI)

```bash
streamlit run src/app.py
```

* Run Streamlit demo: `streamlit run src/app.py`
* Convert PDF to images: `python -m pdf2image.samples --input doc.pdf --output-dir data/images`
* Run tests: `pytest tests/`

## Next steps / Extensions

* Add a named-entity recognition (NER) head for domain-specific fields.
* Add a QA head to answer queries about document content.
* Deploy with a FastAPI wrapper + Dockerfile for production use.

## Contact / Author

* Maintainer: Prakriti
* Email: prakriti14a14j@gmail.com

---

*End of README.*
