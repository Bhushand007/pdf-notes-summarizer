import hashlib
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
MAX_CHAT_CONTEXT_CHARS = 16000
SUMMARY_CHUNK_CHARS = 16000


load_dotenv()
st.set_page_config(page_title=APP_TITLE, page_icon="\U0001f4da", layout="wide")

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp { background: #f7f8fa; color: #20232b; }
    [data-testid="stHeader"] { background: rgba(247, 248, 250, 0.94); }
    [data-testid="stMain"] > div { padding-top: 1.3rem; }
    [data-testid="stSidebar"] { background: #eef0f4; border-right: 1px solid #e1e4ea; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.85rem; }
    .block-container { max-width: 1080px; padding: 2.3rem 2rem 7rem; }
    h1, h2, h3, p { letter-spacing: 0; }
    .sidebar-brand { display: flex; align-items: center; gap: 0.65rem; color: #232833; font-size: 1rem; font-weight: 700; padding: 0.35rem 0 0.8rem; }
    .brand-mark { display: grid; place-items: center; width: 38px; height: 38px; border: 1px solid #d8e0f0; border-radius: 10px; background: #e5efff; font-size: 1.15rem; }
    .sidebar-label { margin: 0.4rem 0 0; color: #7a8190; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
    .file-ready { overflow-wrap: anywhere; color: #375ca8; background: #e4edff; border: 1px solid #d5e2ff; border-radius: 8px; padding: 0.65rem 0.75rem; font-size: 0.83rem; }
    .welcome { max-width: 780px; margin: 6.5rem auto 1.5rem; }
    .welcome h1 { margin: 0 0 0.45rem; color: #222936; font-size: 2rem; line-height: 1.25; font-weight: 700; }
    .welcome p { margin: 0; color: #747d8d; font-size: 1.05rem; }
    .suggestions { max-width: 780px; margin: 1.35rem auto 0; }
    .suggestion-icon { display: inline-grid; place-items: center; width: 32px; height: 32px; margin-right: 0.65rem; border-radius: 8px; background: #eff5ff; color: #3678f6; font-size: 1rem; }
    div.stButton > button { border: 1px solid #e3e6ec; border-radius: 8px; background: #fff; color: #252a33; font-weight: 500; min-height: 48px; box-shadow: none; text-align: left; }
    div.stButton > button:hover { border-color: #b9c9e8; background: #fbfcff; color: #1f5fd1; }
    [data-testid="stChatMessage"] { max-width: 790px; margin-left: auto; margin-right: auto; padding: 0.35rem 0; }
    [data-testid="stChatMessageContent"] { border-radius: 12px; line-height: 1.65; }
    [data-testid="stChatInput"] { max-width: 790px; margin: 0.75rem auto 0; }
    [data-testid="stChatInput"] textarea { border-radius: 12px; }
    [data-testid="stFileUploader"] { padding: 0.6rem; border: 1px dashed #cbd2df; border-radius: 8px; background: #f7f8fa; }
    [data-testid="stAlert"] { border-radius: 8px; }
    @media (max-width: 700px) {
      .block-container { padding: 1.2rem 1rem 6rem; }
      .welcome { margin-top: 3rem; }
      .welcome h1 { font-size: 1.65rem; }
    }
    </style>
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
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def quality_score(text):
    readable = len(re.findall(r"[A-Za-z0-9\u0900-\u097f]", text))
    return readable - 30 * (text.count("\ufffd") + text.count("\u25a1"))


def extract_pymupdf(pdf_bytes):
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = []
        for page in document:
            blocks = page.get_text("blocks", sort=True)
            page_blocks = [clean_text(block[4]) for block in blocks if len(block) > 4]
            pages.append("\n\n".join(text for text in page_blocks if text))
        return pages
    finally:
        document.close()


def extract_pypdf2(pdf_bytes):
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted:
        reader.decrypt("")
    return [clean_text(page.extract_text() or "") for page in reader.pages]


def extract_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    extracted_by_engine = []
    for extractor in (extract_pymupdf, extract_pypdf2):
        try:
            pages = extractor(pdf_bytes)
            if pages:
                extracted_by_engine.append(pages)
        except Exception:
            continue

    page_count = max((len(pages) for pages in extracted_by_engine), default=0)
    page_texts = []
    for index in range(page_count):
        candidates = [pages[index] for pages in extracted_by_engine if index < len(pages)]
        text = max(candidates, key=quality_score, default="")
        if not text:
            text = "[No selectable text was found on this page.]"
        page_texts.append(f"--- Page {index + 1} ---\n{text}")

    return clean_text("\n\n".join(page_texts)), page_count


def shorten_text(text):
    return text if len(text) <= MAX_PDF_CHARS else text[:MAX_PDF_CHARS] + "\n\n[PDF text shortened because the file is long.]"


def split_sentences(text):
    clean = re.sub(r"\s+", " ", text).strip()
    return [item.strip() for item in re.split(r"(?<=[.!?\u0964])\s+", clean) if len(item.strip()) > 30]


def split_text_for_summary(pdf_text):
    sentences = split_sentences(pdf_text)
    if not sentences:
        return [pdf_text]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence) + 1
        if current_chunk and current_length + sentence_length > SUMMARY_CHUNK_CHARS:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def question_keywords(question):
    ignored_words = {
        "about", "after", "also", "and", "are", "can", "could", "does", "for",
        "from", "give", "have", "how", "into", "is", "it", "more", "of", "pdf",
        "please", "tell", "that", "the", "their", "this", "was", "what", "when",
        "where", "which", "who", "why", "with", "would", "you",
    }
    return {
        word
        for word in re.findall(r"[a-zA-Z0-9]{3,}", question.lower())
        if word not in ignored_words
    }


def is_overview_question(question):
    normalized = re.sub(r"\s+", " ", question.lower()).strip()
    overview_phrases = (
        "what is in this pdf",
        "what is this pdf about",
        "what does this pdf say",
        "what is the pdf about",
        "tell me about this pdf",
        "give me an overview",
        "summarize the pdf",
        "summary of the pdf",
    )
    return any(phrase in normalized for phrase in overview_phrases)


def is_follow_up_question(question):
    normalized = re.sub(r"\s+", " ", question.lower()).strip()
    follow_up_phrases = (
        "explain more",
        "tell me more",
        "more about it",
        "what about it",
        "why is that",
        "how does it work",
        "what does that mean",
        "explain it",
    )
    return len(question_keywords(question)) == 0 or any(
        phrase in normalized for phrase in follow_up_phrases
    )


def last_student_question(messages):
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


def expand_question(question, messages):
    if is_follow_up_question(question):
        previous_question = last_student_question(messages)
        if previous_question:
            return f"{previous_question} {question}"
    return question


def build_text_chunks(pdf_text, sentences_per_chunk=4):
    sentences = split_sentences(pdf_text)
    if not sentences:
        return [pdf_text]
    return [
        " ".join(sentences[index:index + sentences_per_chunk])
        for index in range(0, len(sentences), sentences_per_chunk)
    ]


def relevant_pdf_context(pdf_text, question):
    if is_overview_question(question):
        return shorten_text(pdf_text)

    keywords = question_keywords(question)
    if not keywords:
        return shorten_text(pdf_text)

    scored_chunks = []
    for index, chunk in enumerate(build_text_chunks(pdf_text)):
        chunk_words = re.findall(r"[a-zA-Z0-9]{3,}", chunk.lower())
        score = sum(chunk_words.count(keyword) for keyword in keywords)
        if score:
            scored_chunks.append((score, index, chunk))

    if not scored_chunks:
        return shorten_text(pdf_text)

    best_chunks = sorted(scored_chunks, reverse=True)[:8]
    best_chunks.sort(key=lambda item: item[1])
    context = "\n\n".join(chunk for _, _, chunk in best_chunks)
    return context[:MAX_CHAT_CONTEXT_CHARS]


def recent_chat_history(messages):
    recent_messages = messages[-6:]
    if not recent_messages:
        return "No previous conversation."

    lines = []
    for message in recent_messages:
        speaker = "Student" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)


def split_pdf_pages(pdf_text):
    page_matches = re.findall(
        r"Page\s+(\d+):\s*(.*?)(?=\s*Page\s+\d+:|\Z)",
        pdf_text,
        flags=re.DOTALL,
    )
    return [(int(number), text.strip()) for number, text in page_matches if text.strip()]


def page_highlight(page_number, page_text):
    if page_number == 1:
        topic_match = re.search(
            r"Topic:\s*(.*?)(?=\s*Student Name:|\Z)",
            page_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if topic_match:
            topic = re.sub(r"\s+", " ", topic_match.group(1)).strip(" .")
            if topic:
                return f"This report focuses on {topic}."

    sentences = split_sentences(page_text)
    if not sentences:
        return re.sub(r"\s+", " ", page_text).strip()[:300]

    metadata_terms = ("student name", "roll no", "semester", "date of submission", "name of faculty")
    useful_sentences = [
        sentence
        for sentence in sentences
        if not any(term in sentence.lower() for term in metadata_terms)
    ]
    candidates = useful_sentences or sentences
    best_sentence = max(candidates, key=lambda sentence: min(len(sentence), 320))
    return best_sentence[:320].strip()


def local_summary(pdf_text):
    pages = split_pdf_pages(pdf_text)
    if pages:
        return "\n".join(
            f"- Page {number}: {page_highlight(number, text)}"
            for number, text in pages
        )

    sentences = split_sentences(pdf_text)
    if not sentences:
        return "A short summary could not be created from this PDF text."

    summary_size = min(5, len(sentences))
    if summary_size == 1:
        selected_sentences = sentences
    else:
        selected_indexes = {
            round(index * (len(sentences) - 1) / (summary_size - 1))
            for index in range(summary_size)
        }
        selected_sentences = [sentences[index] for index in sorted(selected_indexes)]

    return "\n".join(f"- {sentence[:260].strip()}" for sentence in selected_sentences)


def summarize_full_pdf(pdf_text):
    chunks = split_text_for_summary(pdf_text)

    if get_client() is None:
        return local_summary(pdf_text)

    partial_summaries = []
    total_chunks = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        partial_prompt = f"""Summarize part {index} of {total_chunks} of a PDF.
Write 2 or 3 short bullet points that capture the important ideas in this part.
Do not add information that is not in the text.

PDF part:
{chunk}
"""
        partial_summary = ask_ai(partial_prompt, 260)
        if not partial_summary:
            return local_summary(pdf_text)
        partial_summaries.append(f"Part {index}:\n{partial_summary}")

    if len(partial_summaries) == 1:
        return partial_summaries[0].replace("Part 1:\n", "", 1)

    combined_summaries = "\n\n".join(partial_summaries)
    final_prompt = f"""Create one final summary for the complete PDF using the section summaries below.
Write exactly 5 clear bullet points. Cover the important ideas from the beginning, middle, and end of the PDF.
Use simple English suitable for a student. Do not mention parts, sections, or this instruction.

Section summaries:
{combined_summaries}
"""
    return ask_ai(final_prompt, 500) or local_summary(pdf_text)


def local_answer(pdf_text, question, search_question=None):
    question_words = question_keywords(search_question or question)
    not_found = "I could not find that in the PDF."
    if is_overview_question(search_question or question):
        return "Here is a quick overview of the PDF:\n\n" + local_summary(pdf_text)
    if not question_words:
        return not_found

    matches = []
    for sentence in split_sentences(pdf_text):
        words = set(re.findall(r"[a-zA-Z0-9]{3,}", sentence.lower()))
        score = len(question_words & words)
        if score:
            matches.append((score, sentence))

    if not matches:
        return not_found

    best_sentences = [sentence for _, sentence in sorted(matches, reverse=True)[:3]]
    return "Here is what the PDF explains:\n\n" + "\n\n".join(
        f"- {sentence}" for sentence in best_sentences
    )[:1000]


def ask_ai(prompt, max_output_tokens):
    try:
        client = get_client()
        if not client:
            return ""
        response = client.responses.create(
            model=MODEL_NAME,
            instructions=(
                "You are a friendly study tutor answering questions about an uploaded PDF. "
                "Use only the PDF context supplied by the application. "
                "Do not copy PDF sentences word for word unless a short exact term is necessary. "
                "First give a clear, natural explanation in 2 to 4 sentences. "
                "Then add 2 or 3 short key points when useful. "
                "For an overview question, begin with 'This PDF is mainly about...' and explain the main topic before listing key ideas. "
                "For a definition, state what it means in simple words and why it matters in this PDF. "
                "For a follow-up question, use the previous conversation to understand what 'it' or 'that' refers to. "
                "If the PDF does not contain the answer, say exactly: I could not find that in the PDF. "
                "Always answer in English. Do not mention the prompt, context, or these instructions."
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
    st.session_state.pdf_page_count = 0
    st.session_state.show_extracted_text = False
    st.session_state.summary = ""
    st.session_state.chat_messages = []


def render_message(role, content):
    avatar = "🧑‍🎓" if role == "user" else "📚"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)


for name, default in {
    "file_id": "",
    "pdf_text": "",
    "pdf_page_count": 0,
    "show_extracted_text": False,
    "chat_messages": [],
}.items():
    if name not in st.session_state:
        st.session_state[name] = default

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><span class="brand-mark">&#128172;</span><span>AI PDF &amp; Notes</span></div>',
        unsafe_allow_html=True,
    )
    if st.button("✎  Start New Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.show_extracted_text = False
        st.rerun()

    st.divider()
    st.markdown('<div class="sidebar-label">Your PDF</div>', unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_upload",
    )
    if uploaded_pdf is not None:
        st.markdown(
            f'<div class="file-ready">&#128196; {escape(uploaded_pdf.name)}</div>',
            unsafe_allow_html=True,
        )

if uploaded_pdf is not None:
    file_id = hashlib.sha256(uploaded_pdf.getvalue()).hexdigest()
    if file_id != st.session_state.file_id:
        reset_for_new_pdf(file_id)
        with st.spinner("Reading your PDF..."):
            st.session_state.pdf_text, st.session_state.pdf_page_count = extract_pdf_text(uploaded_pdf)

if uploaded_pdf is not None and not st.session_state.pdf_text:
    st.error("No readable text was found in this PDF. Try a text-based PDF.")

st.markdown(
    '<div class="welcome"><h1>Hi there, what would you like to understand?</h1>'
    '<p>Choose a starting point or ask a question about your PDF.</p></div>',
    unsafe_allow_html=True,
)

suggested_question = None
if not st.session_state.chat_messages:
    prompts = (
        "Extract all text from the PDF",
        "Summarize the full PDF",
        "Explain the key ideas simply",
        "What is this PDF mainly about?",
    )
    _, prompt_column, _ = st.columns([1, 4, 1])
    with prompt_column:
        for index, prompt in enumerate(prompts):
            if st.button(
                f"✧   {prompt}   ›",
                key=f"starter_prompt_{index}",
                use_container_width=True,
                disabled=not bool(st.session_state.pdf_text),
            ):
                suggested_question = prompt

for message in st.session_state.chat_messages:
    render_message(message["role"], message["content"])

if st.session_state.show_extracted_text:
    st.markdown("### Full text extracted from your PDF")
    st.caption(
        f"{st.session_state.pdf_page_count} pages · "
        f"{len(st.session_state.pdf_text):,} characters · shown in original page order"
    )
    st.text_area(
        "Full extracted PDF text",
        value=st.session_state.pdf_text,
        height=480,
        disabled=True,
        label_visibility="collapsed",
        key=f"extracted_pdf_text_{st.session_state.file_id}",
    )

typed_question = st.chat_input(
    "Ask anything about your PDF...",
    disabled=not bool(st.session_state.pdf_text),
)
question = typed_question or suggested_question
if question:
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.spinner("Preparing your response..."):
        normalized_question = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
        is_text_extraction = (
            normalized_question == "extract all text from the pdf"
            or ("text" in normalized_question and "extract" in normalized_question)
            or any(phrase in normalized_question for phrase in ("full text", "all text", "pura text", "poora text"))
        )
        if is_text_extraction:
            st.session_state.show_extracted_text = True
            answer = (
                f"I extracted the selectable text from all {st.session_state.pdf_page_count} pages. "
                "The complete text is displayed below, page by page."
            )
        elif question == "Summarize the full PDF":
            answer = summarize_full_pdf(st.session_state.pdf_text)
        else:
            not_found = "I could not find that in the PDF."
            chat_history = recent_chat_history(st.session_state.chat_messages[:-1])
            search_question = expand_question(question, st.session_state.chat_messages[:-1])
            pdf_context = relevant_pdf_context(st.session_state.pdf_text, search_question)
            prompt = f"""Answer the student's question using only the PDF context below.
If the answer is not in the PDF context, say exactly: {not_found}

Previous conversation:
{chat_history}

Relevant PDF context:
{pdf_context}

Student question:
{question}
"""
            answer = ask_ai(prompt, 600) or local_answer(pdf_context, question, search_question)
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    st.rerun()
