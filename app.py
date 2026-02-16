import streamlit as st
from dotenv import load_dotenv
import os
import pytesseract
from PIL import Image
import fitz
import io
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Correct import for Mistral v0.4.2
from mistralai.client import MistralClient
from openai import OpenAI
import os
load_dotenv()
print("DEBUG — Loaded MISTRAL_API_KEY =", os.getenv("MISTRAL_API_KEY"))



# -----------------------------
# Tesseract Path
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------------
# Load Keys
# -----------------------------
load_dotenv()
mistral_key = os.getenv("MISTRAL_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

st.title("🤖 MIAA – OCR + Document Understanding")
st.write("🔐 Mistral Key Loaded:", bool(mistral_key))
st.write("🔐 OpenAI Key Loaded:", bool(openai_key))


# -----------------------------
# Test Keys
# -----------------------------
def test_mistral_key(key):
    try:
        client = MistralClient(api_key=key)
        client.models.list()
        return True, "✅ Mistral Key is valid!"
    except Exception as e:
        return False, f"❌ Mistral Key failed: {e}"


def test_openai_key(key):
    try:
        client = OpenAI(api_key=key)
        client.models.list()
        return True, "✅ OpenAI Key is valid!"
    except Exception as e:
        return False, f"❌ OpenAI Key failed: {e}"


def api_key_testing_ui():
    st.subheader("🔑 API Key Testing")

    mistral_in = st.text_input("Enter Mistral API Key", type="password")
    openai_in = st.text_input("Enter OpenAI API Key", type="password")

    if st.button("Test Keys"):
        if mistral_in:
            ok, msg = test_mistral_key(mistral_in)
            st.write(msg)

        if openai_in:
            ok, msg = test_openai_key(openai_in)
            st.write(msg)

        if not mistral_in and not openai_in:
            st.warning("Enter at least one key to test.")


with st.expander("🔧 Test Your API Keys"):
    api_key_testing_ui()


# -----------------------------
# MAIN PROCESS DOCUMENT
# -----------------------------
def process_document(uploaded_file, mistral_key, openai_key):

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

    # ----- Embeddings -----
    chunks = [all_text[i:i+1000] for i in range(0, len(all_text), 1000)]
    embeddings = embedder.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Ask question
    user_query = st.text_input("Ask your question:")

    if user_query:
        query_embed = embedder.encode([user_query])
        D, I = index.search(np.array(query_embed), k=3)

        context = "\n".join([chunks[i] for i in I[0]])

        st.markdown("### 📖 Relevant Context")
        st.write(context)

        model_choice = st.radio("Choose Model:", ["Mistral", "OpenAI"])

        with st.spinner("🤖 Generating Answer..."):
            try:
                if model_choice == "Mistral":
                    response = mistral.chat(
                        model="mistral-small-latest",
                        messages=[
                            {"role": "system", "content": "You are an industrial assistant."},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"}
                        ]
                    )
                    answer = response.choices[0].message.content

                else:
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an industrial assistant."},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"}
                        ]
                    )
                    answer = response.choices[0].message.content

            except Exception as e:
                answer = f"❌ LLM Error: {e}"

        st.markdown("### 🧠 Answer")
        st.write(answer)


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    process_document(uploaded_file, mistral_key, openai_key)
