import fitz
import os
from pathlib import Path

RAW_PDF_DIR = Path("data/sample_doc/raw_pdfs")
IMG_DIR = Path("data/sample_doc/images")

print("RAW PDF DIR:", RAW_PDF_DIR.resolve())
print("IMG DIR:", IMG_DIR.resolve())

if not RAW_PDF_DIR.exists():
    print("❌ ERROR: raw_pdfs folder NOT FOUND at:", RAW_PDF_DIR.resolve())
else:
    print("✔ raw_pdfs folder FOUND")

if not IMG_DIR.exists():
    print("Creating images folder...")
    IMG_DIR.mkdir(parents=True, exist_ok=True)

def pdf_to_images():
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
    print("Found PDFs:", pdf_files)

    if len(pdf_files) == 0:
        print("❌ No PDF files found inside:")
        print(RAW_PDF_DIR.resolve())
        return

    for pdf_file in pdf_files:
        print("Processing:", pdf_file)
        pdf = fitz.open(pdf_file)

        base = pdf_file.stem.replace(" ", "_")

        for i, page in enumerate(pdf):
            pix = page.get_pixmap(dpi=200)
            out_path = IMG_DIR / f"{base}_page{i+1}.png"
            pix.save(out_path)
            print("Saved:", out_path)

if __name__ == "__main__":
    pdf_to_images()
