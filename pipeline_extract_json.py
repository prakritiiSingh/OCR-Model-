# pipeline_extract_json.py

import json
from inference_layoutlm import run_inference
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

###########################################################
# STEP 1: GROUP TOKENS INTO LINES
###########################################################
def group_tokens_by_line(tokens, y_threshold=10):
    """
    Group tokens into lines based on Y coordinate closeness.
    """
    lines = []
    current_line = []
    
    # Sort tokens by top y coordinate
    tokens_sorted = sorted(tokens, key=lambda x: x["bbox"][1])

    last_y = None

    for token in tokens_sorted:
        y = token["bbox"][1]

        if last_y is None:
            current_line.append(token)
            last_y = y
            continue

        # If token is close to previous line -> same line
        if abs(y - last_y) <= y_threshold:
            current_line.append(token)
        else:
            # Start new line
            lines.append(current_line)
            current_line = [token]

        last_y = y

    if current_line:
        lines.append(current_line)

    return lines


###########################################################
# STEP 2: CONVERT GROUPED LINES TO READABLE TEXT
###########################################################
def lines_to_text(lines):
    final_text = ""
    for line in lines:
        sorted_line = sorted(line, key=lambda x: x["bbox"][0])
        line_text = " ".join([t["text"] for t in sorted_line])
        final_text += line_text + "\n"
    return final_text.strip()


###########################################################
# STEP 3: ASK GPT TO EXTRACT STRUCTURED JSON
###########################################################
from openai import OpenAI
client = OpenAI()

def ask_gpt_for_json(document_text):
    prompt = f"""
You are an industrial document expert.

Extract a structured JSON from the following document text:

-----------------------
{document_text}
-----------------------

Recognize the type automatically. Use this schema:

{{
  "document_type": "",
  "certificate_no": "",
  "customer_name": "",
  "customer_address": "",
  "instrument_details": "",
  "date_of_calibration": "",
  "issue_date": "",
  "next_due_date": "",
  "equipment_used": [],
  "test_results": [],
  "remarks": ""
}}

If information is missing, leave empty string or empty list.

Return ONLY pure JSON.
"""

    response = client.chat.completions.create(

        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

json_text = response.choices[0].message["content"]

###########################################################
# STEP 4: FULL PIPELINE — IMAGE → TOKENS → LINES → GPT JSON
###########################################################
def extract_json_from_image(image_path):
    print("Running LayoutLMv3 inference...")
    tokens = run_inference(image_path)

    print("Grouping tokens...")
    lines = group_tokens_by_line(tokens)

    print("Building text...")
    doc_text = lines_to_text(lines)
    print("\n===== EXTRACTED TEXT =====\n")
    print(doc_text)

    print("\nSending to GPT for JSON extraction...")
    final_json = ask_gpt_for_json(doc_text)

    return final_json


###########################################################
# TEST
###########################################################
if __name__ == "__main__":
    test_img = "data/sample_doc/images/Calibration_Certificate_-_PUN_-_01_page1.png"

    result = extract_json_from_image(test_img)

    print("\n===== FINAL STRUCTURED JSON =====\n")
    print(result)
