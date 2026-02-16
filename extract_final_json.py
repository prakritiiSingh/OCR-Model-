# extract_final_json.py

from mistralai.client import MistralClient
from inference_layoutlm import run_inference

# ------------------------------------
# ✅ USE YOUR VERIFIED WORKING API KEY
# ------------------------------------
client = MistralClient(api_key="69jVvQFLfQPS6eeAO8IP26c5IPOGWLKx")


def build_text_from_tokens(tokens):
    """Convert token list into plain readable text."""
    return " ".join([item["text"] for item in tokens]).strip()


def ask_llm_for_json(extracted_text):
    """Send OCR text to Mistral LLM and receive structured JSON."""

    # Use triple single-quotes to avoid format() escaping issues
    json_template = '''
{
  "document_type": "",
  "certificate_no": "",
  "customer_name": "",
  "customer_address": "",
  "date_of_calibration": "",
  "issue_date": "",
  "next_due_date": "",
  "instrument_details": "",
  "test_values": [],
  "remarks": ""
}
'''

    prompt = (
        "Extract structured JSON from the following document text.\n"
        "Return ONLY the JSON — no explanation, no extra text.\n\n"
        f"Document Text:\n{extracted_text}\n\n"
        "JSON Format:\n"
        f"{json_template}"
    )

    response = client.chat(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": "You output ONLY clean JSON. No markdown."},
            {"role": "user", "content": prompt}
        ]
    )

    # message is an object -> .content
    return response.choices[0].message.content


def extract_json(image_path):
    """Run OCR (LayoutLMv3) and extract structured JSON from certificate."""

    print("Running LayoutLMv3 inference...")
    tokens = run_inference(image_path)

    text = build_text_from_tokens(tokens)

    print("\n===== EXTRACTED TEXT (RAW) =====\n")
    print(text[:800] + "...\n")

    print("Sending to Mistral for JSON extraction...")
    extracted_json = ask_llm_for_json(text)

    return extracted_json


if __name__ == "__main__":
    img = "data/sample_doc/images/Calibration_Certificate_-_PUN_-_01_page1.png"
    result = extract_json(img)

    print("\n===== FINAL STRUCTURED JSON =====\n")
    print(result)
