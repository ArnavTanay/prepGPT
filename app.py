import streamlit as st
import pymupdf
from google import genai
import os
from dotenv import load_dotenv

# Connect to AI
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---- FUNCTIONS ----

def extract_text_from_pdf(uploaded_file):
    doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def chat_with_ai(user_question, chat_history, pdf_text=None):
    # Build context based on whether PDF is uploaded
    if pdf_text:
        system_context = f"""You are PrepGPT — an expert AI tutor for exam preparation.
You have been given the following study material by the student:

{pdf_text[:4000]}

Always prioritize this material when answering.
If the question is outside this material, answer from your own knowledge.
Be concise, exam-focused, and student-friendly."""
    else:
        system_context = """You are PrepGPT — an expert AI tutor for exam preparation.
You help students prepare for exams like NEET, JEE, CAT, GATE, and UPSC.
Be concise, exam-focused, and student-friendly.
Format answers clearly with bullet points where helpful."""

    # Build full conversation for memory
    conversation = system_context + "\n\n"
    for msg in chat_history:
        if msg["role"] == "user":
            conversation += f"Student: {msg['content']}\n"
        else:
            conversation += f"PrepGPT: {msg['content']}\n"
    conversation += f"Student: {user_question}\nPrepGPT:"

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=conversation
    )
    return response.text

def generate_notes(text):
    prompt = f"""
You are a NEET Biology expert teacher with 15 years of experience.
Based on the following textbook content, generate structured study notes.

TEXTBOOK CONTENT:
{text[:3000]}

Format EXACTLY like this:

DEFINITION:
[One clear sentence defining the main topic]

KEY CONCEPTS:
[5-7 bullet points covering the most important concepts]

IMPORTANT TERMS:
[5 key terms with their one-line meaning]

MUST REMEMBER FOR EXAM:
[3 high-yield facts that frequently appear in exams]

PREVIOUS YEAR QUESTION:
[One realistic exam-style MCQ with 4 options and correct answer]

EXAM TIP:
[One smart tip to remember this topic easily]
"""
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def generate_mcq(text):
    prompt = f"""
You are an exam expert.
Based on the following textbook content, generate 5 exam-style MCQs.

TEXTBOOK CONTENT:
{text[:3000]}

Format EXACTLY like this:

Q1: [Question]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: [Correct option + one line explanation]

Q2: [Question]
...up to Q5
"""
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def generate_flashcards(text):
    prompt = f"""
You are an exam expert.
Based on the following textbook content, generate 5 flashcards.

TEXTBOOK CONTENT:
{text[:3000]}

Format EXACTLY like this:

CARD 1:
FRONT: [Key term or concept]
BACK: [Clear concise definition or explanation]

CARD 2:
...up to CARD 5
"""
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )
    return response.text


# ---- UI ----

st.set_page_config(
    page_title="PrepGPT",
    page_icon="🧬",
    layout="wide"
)

# ---- SIDEBAR ----
with st.sidebar:
    st.title("🧬 PrepGPT")
    st.caption("Your AI Exam Preparation Assistant")
    st.divider()

    # Exam selector
    exam_type = st.selectbox(
        "Select Your Exam",
        ["NEET", "JEE", "CAT", "GATE", "UPSC", "Other"]
    )

    st.divider()

    # PDF Upload
    st.markdown("### 📄 Upload Study Material")
    st.caption("Optional — chat works without PDF too")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    # Extract text from PDFs
    pdf_text = None
    if uploaded_files:
        all_text = ""
        for file in uploaded_files:
            all_text += extract_text_from_pdf(file)
        pdf_text = all_text
        st.success(f"✅ {len(uploaded_files)} PDF(s) loaded")
        st.caption(f"{len(all_text)} characters extracted")
        st.divider()

        # Quick tools
        st.markdown("### ⚡ Quick Tools")
        notes_btn = st.button("📝 Generate Notes", use_container_width=True)
        mcq_btn = st.button("❓ Generate MCQs", use_container_width=True)
        flash_btn = st.button("🃏 Generate Flashcards", use_container_width=True)

        if notes_btn:
            with st.spinner("Generating notes..."):
                notes = generate_notes(pdf_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"📝 **NEET Study Notes**\n\n{notes}"
            })
            st.rerun()

        if mcq_btn:
            with st.spinner("Generating MCQs..."):
                mcqs = generate_mcq(pdf_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❓ **Practice MCQs**\n\n{mcqs}"
            })
            st.rerun()

        if flash_btn:
            with st.spinner("Generating flashcards..."):
                flashcards = generate_flashcards(pdf_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"🃏 **Flashcards**\n\n{flashcards}"
            })
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---- MAIN CHAT AREA ----

st.title(f"PrepGPT — {exam_type} Assistant")
st.caption("Ask anything. Upload a PDF for personalized answers from your material.")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show welcome message if no messages
if not st.session_state.messages:
    st.info("👋 Hi! I'm PrepGPT. Ask me anything about your exam, or upload your study material for personalized help.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input(f"Ask PrepGPT anything about {exam_type}...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_with_ai(
                user_input,
                st.session_state.messages[:-1],
                pdf_text
            )
        st.markdown(response)

    # Add AI response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })