import streamlit as st
import pytesseract
from PIL import Image
import fitz
import io
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---- FIXED: Correct import for Mistral v0.4.2 ----
from mistralai.client import MistralClient

from openai import OpenAI
import os


# -----------------------------
# TESSERACT CONFIG
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -----------------------------
# API KEY TEST FUNCTIONS
# -----------------------------
def test_mistral_key(key: str):
    """Test Mistral key directly (fixed bug)"""
    try:
        client = MistralClient(api_key=key)  # <-- FIXED
        client.models.list()
        return True, "✅ Mistral Key is valid!"
    except Exception as e:
        return False, f"❌ Mistral Key failed: {e}"


def test_openai_key(key: str):
    try:
        client = OpenAI(api_key=key)
        client.models.list()
        return True, "✅ OpenAI Key is valid!"
    except Exception as e:
        return False, f"❌ OpenAI Key failed: {e}"


# -----------------------------
# MAIN DOCUMENT PROCESS FUNCTION
# -----------------------------
def process_document(uploaded_file, mistral_key, openai_key):

    # ---- FIXED: Correct Mistral client ----
    mistral = MistralClient(api_key=mistral_key)

    openai = OpenAI(api_key=openai_key)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    all_text = ""
    progress = st.progress(0)

    # ----- OCR -----
    for i, page in enumerate(pdf):
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)

        all_text += f"\n\n--- Page {i+1} ---\n{text}"
        progress.progress((i + 1) / len(pdf))

    st.text_area("OCR Output", all_text, height=250)

    # ----- EMBEDDINGS -----
    chunks = [all_text[i:i+1000] for i in range(0, len(all_text), 1000)]
    embeddings = embedder.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    user_query = st.text_input("Ask your question:")

    if user_query:
        query_embed = embedder.encode([user_query])
        D, I = index.search(np.array(query_embed), k=3)

        context = "\n".join([chunks[i] for i in I[0]])

        st.markdown("### 📖 Relevant Context")
        st.markdown(
            f"<div class='context-box'>{context}</div>",
            unsafe_allow_html=True
        )

        # -----------------------------
        # MODEL CHOICE
        # -----------------------------
        model_choice = st.radio(
            "Choose Model:",
            ["Mistral", "OpenAI"]
        )

        # -----------------------------
        # RUN LLM CALL
        # -----------------------------
        with st.spinner("🤖 Generating Answer..."):
            try:
                if model_choice == "Mistral":
                    # ---- FIXED Mistral call syntax ----
                    response = mistral.chat(
                        model="mistral-small-latest",
                        messages=[
                            {"role": "system", "content": "You are an industrial assistant."},
                            {"role": "user",
                             "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"}
                        ]
                    )
                    answer = response.choices[0].message.content

                else:  # OpenAI
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an industrial assistant."},
                            {"role": "user",
                             "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"}
                        ]
                    )
                    answer = response.choices[0].message.content

            except Exception as e:
                answer = f"❌ LLM Error: {e}"

        st.markdown("### 🧠 Answer")
        st.markdown(
            f"<div class='answer-box'>{answer}</div>",
            unsafe_allow_html=True
        )


# -----------------------------
# STREAMLIT: KEY TEST UI
# -----------------------------
def api_key_testing_ui():
    st.subheader("🔑 API Key Testing")

    mistral_key = st.text_input("Enter Mistral API Key", type="password")
    openai_key = st.text_input("Enter OpenAI API Key", type="password")

    if st.button("Test Keys"):

        if mistral_key:
            ok, msg = test_mistral_key(mistral_key)
            st.write(msg)

        if openai_key:
            ok, msg = test_openai_key(openai_key)
            st.write(msg)

        if not mistral_key and not openai_key:
            st.warning("Enter at least one key to test.")
