import os
import re
from html import escape
from io import BytesIO

import fitz
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


APP_TITLE = "PDF & Notes Summarizer"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_PDF_CHARS = 20000


load_dotenv()
st.set_page_config(page_title=APP_TITLE, page_icon="\U0001f4da", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background: #f8faff; color: #172033; }
    .block-container { max-width: 900px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1, h2, h3, p { letter-spacing: 0; }
    .app-header { text-align: center; margin-bottom: 1.8rem; }
    .app-title { font-size: 2.7rem; font-weight: 800; color: #20283d; margin-bottom: 0.4rem; }
    .app-subtitle { font-size: 1.05rem; color: #66708a; font-weight: 500; }
    .section-card { background: #ffffff; border: 1px solid #e7ebf7; border-radius: 8px; padding: 1.3rem 1.35rem; margin: 1rem 0; box-shadow: 0 8px 20px rgba(44, 62, 120, 0.08); }
    .section-title { font-size: 1.25rem; font-weight: 800; color: #24304f; margin-bottom: 0.25rem; }
    .section-text { color: #6b7590; margin-bottom: 0.9rem; }
    .selected-file { display: inline-block; background: #eef4ff; color: #3156a3; border-radius: 8px; padding: 0.5rem 0.85rem; font-weight: 700; margin-top: 0.75rem; }
    .summary-card { background: #fbfcff; border-left: 5px solid #7c6df2; border-radius: 8px; padding: 1.15rem 1.25rem; margin-top: 0.85rem; color: #25304a; line-height: 1.65; white-space: pre-wrap; box-shadow: inset 0 0 0 1px #edf0fb; }
    .empty-card { background: #fbfcff; border: 1px dashed #c9d2ea; color: #737d96; border-radius: 8px; padding: 1rem 1.2rem; margin-top: 0.85rem; }
    .chat-row { margin: 0.75rem 0; }
    .chat-bubble { border-radius: 8px; padding: 0.9rem 1rem; line-height: 1.55; white-space: pre-wrap; box-shadow: 0 6px 16px rgba(40, 55, 105, 0.07); }
    .user-bubble { background: #eef4ff; border: 1px solid #dce7ff; color: #24304f; }
    .ai-bubble { background: #ffffff; border: 1px solid #e7ebf7; color: #25304a; }
    .speaker { font-weight: 800; margin-bottom: 0.35rem; color: #4c5fd7; }
    div.stButton > button { background: #5f6ff2; color: white; border: 0; border-radius: 8px; padding: 0.65rem 1.35rem; font-weight: 800; box-shadow: 0 8px 18px rgba(95, 111, 242, 0.24); }
    div.stButton > button:hover { color: white; background: #4d5cda; border: 0; }
    [data-testid="stFileUploader"] { background: #f8faff; border: 1px dashed #b9c8f6; border-radius: 8px; padding: 0.8rem; }
    [data-testid="stAlert"] { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">&#128218; PDF &amp; Notes Summarizer</div>
        <div class="app-subtitle">Upload your notes, get a quick summary, and ask questions.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def get_client():
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    api_key = os.getenv("OPENAI_API_KEY") or secret_key
    return OpenAI(api_key=api_key) if api_key else None


def clean_text(text):
    text = text.replace("\x00", " ").replace("\ufffd", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def quality_score(text):
    readable = len(re.findall(r"[A-Za-z0-9\u0900-\u097f]", text))
    return readable - 30 * (text.count("\ufffd") + text.count("\u25a1"))


def extract_pymupdf(pdf_bytes):
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = []
        for number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(f"Page {number}:\n{text}")
        return clean_text("\n\n".join(pages))
    finally:
        document.close()


def extract_pypdf2(pdf_bytes):
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted:
        reader.decrypt("")
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"Page {number}:\n{text}")
    return clean_text("\n\n".join(pages))


def extract_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    results = []
    for extractor in (extract_pymupdf, extract_pypdf2):
        try:
            text = extractor(pdf_bytes)
            if text:
                results.append(text)
        except Exception:
            continue
    return max(results, key=quality_score, default="")


def shorten_text(text):
    return text if len(text) <= MAX_PDF_CHARS else text[:MAX_PDF_CHARS] + "\n\n[PDF text shortened because the file is long.]"


def split_sentences(text):
    clean = re.sub(r"\s+", " ", text).strip()
    return [item.strip() for item in re.split(r"(?<=[.!?\u0964])\s+", clean) if len(item.strip()) > 30]


def local_summary(pdf_text):
    sentences = split_sentences(pdf_text)
    if not sentences:
        return "A short summary could not be created from this PDF text."
    return "\n".join(f"- {sentence[:230].strip()}" for sentence in sentences[:5])


def local_answer(pdf_text, question):
    excluded = {"what", "when", "where", "which", "about", "does", "this", "that", "from", "with"}
    question_words = set(re.findall(r"[a-zA-Z\u0900-\u097f]{3,}", question.lower())) - excluded
    not_found = "I could not find that in the PDF."
    if not question_words:
        return not_found
    best_sentence, best_score = "", 0
    for sentence in split_sentences(pdf_text):
        words = set(re.findall(r"[a-zA-Z\u0900-\u097f]{3,}", sentence.lower()))
        score = len(question_words & words)
        if score > best_score:
            best_sentence, best_score = sentence, score
    return best_sentence[:650] if best_score else not_found


def ask_ai(prompt, max_output_tokens):
    try:
        client = get_client()
        if not client:
            return ""
        response = client.responses.create(
            model=MODEL_NAME,
            instructions=(
                "You are a helpful study assistant. Use only the PDF text provided. "
                "Keep the answer short, simple, and easy for a beginner to understand. "
                "Always answer in English."
            ),
            input=prompt,
            max_output_tokens=max_output_tokens,
        )
        return response.output_text.strip()
    except Exception:
        return ""


def reset_for_new_pdf(file_id):
    st.session_state.file_id = file_id
    st.session_state.pdf_text = ""
    st.session_state.summary = ""
    st.session_state.chat_messages = []


def render_message(role, content):
    speaker = "You" if role == "user" else "AI"
    bubble = "user-bubble" if role == "user" else "ai-bubble"
    st.markdown(
        f'<div class="chat-row"><div class="chat-bubble {bubble}"><div class="speaker">{speaker}:</div><div>{escape(content)}</div></div></div>',
        unsafe_allow_html=True,
    )


for name, default in {
    "file_id": "",
    "pdf_text": "",
    "summary": "",
    "chat_messages": [],
}.items():
    if name not in st.session_state:
        st.session_state[name] = default

st.markdown('<div class="section-card"><div class="section-title">&#128196; Upload Your PDF</div><div class="section-text">Upload your notes or study material to get started.</div>', unsafe_allow_html=True)
uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")
if uploaded_pdf:
    st.markdown(f'<div class="selected-file">&#9989; Selected: {escape(uploaded_pdf.name)}</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if not uploaded_pdf:
    st.stop()

file_id = f"{uploaded_pdf.name}-{uploaded_pdf.size}"
if file_id != st.session_state.file_id:
    reset_for_new_pdf(file_id)
    with st.spinner("Reading your PDF..."):
        st.session_state.pdf_text = extract_pdf_text(uploaded_pdf)

if not st.session_state.pdf_text:
    st.warning("No readable text was found. Please try a clear text-based PDF.")
    st.stop()

if st.button("\u2728 Generate Summary"):
    with st.spinner("Creating a quick summary..."):
        pdf_text = shorten_text(st.session_state.pdf_text)
        prompt = f"Summarize this PDF in 5 short and simple bullet points. Write the summary only in English.\n\nPDF text:\n{pdf_text}"
        st.session_state.summary = ask_ai(prompt, 450) or local_summary(pdf_text)

st.markdown('<div class="section-card"><div class="section-title">&#128221; Summary</div>', unsafe_allow_html=True)
if st.session_state.summary:
    st.markdown(f'<div class="summary-card">{escape(st.session_state.summary)}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="empty-card">Click <b>&#10024; Generate Summary</b> to see a short summary here.</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">&#128172; Ask About Your PDF</div><div class="section-text">Ask anything about your uploaded notes.</div>', unsafe_allow_html=True)
for message in st.session_state.chat_messages:
    render_message(message["role"], message["content"])
st.markdown("</div>", unsafe_allow_html=True)

question = st.chat_input("Ask a question...")
if question:
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.spinner("Finding the answer in your PDF..."):
        pdf_text = shorten_text(st.session_state.pdf_text)
        not_found = "I could not find that in the PDF."
        prompt = f"Answer the user's question using only the PDF text below. If the answer is not in the PDF, say exactly: {not_found}\nWrite the answer only in English.\n\nPDF text:\n{pdf_text}\n\nQuestion:\n{question}"
        answer = ask_ai(prompt, 350) or local_answer(pdf_text, question)
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    st.rerun()
