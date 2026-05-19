import streamlit as st
import pymupdf
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---- FUNCTIONS ----

def get_exam_context(exam_type):
    contexts = {
        "NEET": "You are a NEET Biology/Chemistry/Physics expert with 15 years of teaching experience. Focus on NCERT content, MCQ format, and NEET exam patterns.",
        "JEE": "You are a JEE Mains and Advanced expert with 15 years of teaching experience. Focus on conceptual depth, numerical problems, and JEE exam patterns.",
        "CAT": "You are a CAT exam expert with 15 years of coaching experience. Focus on Verbal Ability, Reading Comprehension, Data Interpretation, Logical Reasoning, and Quantitative Aptitude.",
        "GATE": "You are a GATE exam expert with 15 years of teaching experience. Focus on technical concepts, numerical answer type questions, and GATE exam patterns.",
        "UPSC": "You are a UPSC Civil Services exam expert with 15 years of experience. Focus on conceptual understanding, analytical answers, and UPSC exam patterns.",
        "Other": "You are an expert exam tutor with 15 years of teaching experience. Focus on key concepts and exam-relevant content."
    }
    return contexts.get(exam_type, contexts["Other"])

def extract_text_from_pdf(uploaded_file):
    doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def chat_with_ai(user_question, chat_history, exam_type, pdf_text=None):
    if pdf_text:
        system_context = f"""You are PrepGPT — an expert AI tutor for {exam_type} exam preparation.
You have been given the following study material by the student:

{pdf_text[:4000]}

Always prioritize this material when answering.
If the question is outside this material, answer from your own knowledge.
Be concise, exam-focused, and student-friendly."""
    else:
        system_context = f"""You are PrepGPT — an expert AI tutor for {exam_type} exam preparation.
You help students prepare for {exam_type} exam.
Be concise, exam-focused, and student-friendly.
Format answers clearly with bullet points where helpful."""

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

def generate_notes(text, exam_type):
    chunk = text[:3000]
    context = get_exam_context(exam_type)
    prompt = f"""
{context}
Based on the following textbook content, generate structured study notes.

TEXTBOOK CONTENT:
{chunk}

Format EXACTLY like this:

DEFINITION:
[One clear sentence defining the main topic]

KEY CONCEPTS:
[5-7 bullet points covering the most important concepts]

IMPORTANT TERMS:
[5 key terms with their one-line meaning]

MUST REMEMBER FOR {exam_type}:
[3 high-yield facts that frequently appear in {exam_type} exam]

PREVIOUS YEAR QUESTION:
[One realistic {exam_type}-style MCQ with 4 options and correct answer]

EXAM TIP:
[One smart tip to remember this topic easily]
"""
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def generate_mcq(text, exam_type):
    chunk = text[:3000]
    context = get_exam_context(exam_type)
    prompt = f"""
{context}

Based only on the following textbook content, generate 5 {exam_type}-style MCQs.

TEXTBOOK CONTENT:
{chunk}

STRICT RULES:
1. Generate exactly 5 MCQs.
2. Each question must have exactly 4 options.
3. Each option MUST be on a separate new line.
4. NEVER place multiple options on the same line.
5. Insert a line break after every option.
6. Do NOT compress options into paragraphs.
7. Start options immediately from the next line of the ending of question.
8. Do NOT use markdown, bold text, bullet points, or extra commentary.
9. Do NOT provide answers after each question.
10. After all 5 questions, create a separate ANSWERS section.
11. In the ANSWERS section, include:
   - Correct option
   - One-line explanation
12. Follow the format EXACTLY.

EXAMPLE FORMAT:

Q1. What is the capital of France?

A. Berlin
B. Madrid
C. Paris
D. Rome

OUTPUT FORMAT:

Q1. [Question]

A. [Option]
B. [Option]
C. [Option]
D. [Option]

Q2. [Question]

A. [Option]
B. [Option]
C. [Option]
D. [Option]

Q3. [Question]

A. [Option]
B. [Option]
C. [Option]
D. [Option]

Q4. [Question]

A. [Option]
B. [Option]
C. [Option]
D. [Option]

Q5. [Question]

A. [Option]
B. [Option]
C. [Option]
D. [Option]

ANSWERS:

Q1. [Correct Option] - [One-line explanation]
Q2. [Correct Option] - [One-line explanation]
Q3. [Correct Option] - [One-line explanation]
Q4. [Correct Option] - [One-line explanation]
Q5. [Correct Option] - [One-line explanation]
"""
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )
    # Fix line breaks for Streamlit markdown rendering
    result = response.text
    result = result.replace("\nA.", "\n\nA.")
    result = result.replace("\nB.", "\n\nB.")
    result = result.replace("\nC.", "\n\nC.")
    result = result.replace("\nD.", "\n\nD.")

    return result

def generate_flashcards(text, exam_type):
    chunk = text[:3000]
    context = get_exam_context(exam_type)
    prompt = f"""
{context}
Based on the following textbook content, generate 5 flashcards.

TEXTBOOK CONTENT:
{chunk}

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

def generate_important_questions(text, exam_type):
    chunk = text[:3000]
    context = get_exam_context(exam_type)
    prompt = f"""
{context}
Based on the following textbook content, identify the most important questions for {exam_type}.

TEXTBOOK CONTENT:
{chunk}

Format EXACTLY like this:

MOST IMPORTANT QUESTIONS FOR {exam_type}:

ONE MARK QUESTIONS:
1. [Question]
2. [Question]
3. [Question]
4. [Question]
5. [Question]

TWO MARK QUESTIONS:
1. [Question]
2. [Question]
3. [Question]

FIVE MARK QUESTIONS:
1. [Question]
2. [Question]

MOST LIKELY MCQ TOPICS:
1. [Topic + why it's important]
2. [Topic + why it's important]
3. [Topic + why it's important]

EXAMINER'S FAVOURITE CONCEPTS:
[3 concepts that appear repeatedly in past papers from this topic]
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

    exam_type = st.selectbox(
        "Select Your Exam",
        ["NEET", "JEE", "CAT", "GATE", "UPSC", "Other"]
    )

    st.divider()

    st.markdown("### 📄 Upload Study Material")
    st.caption("Optional — chat works without PDF too")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    pdf_text = None
    if uploaded_files:
        all_text = ""
        for file in uploaded_files:
            all_text += extract_text_from_pdf(file)
        pdf_text = all_text
        st.success(f"✅ {len(uploaded_files)} PDF(s) loaded")
        st.caption(f"{len(all_text)} characters extracted")

    st.divider()

    #Quick Tools - always available
    st.markdown("### ⚡ Quick Tools")

    #Topic input - used when no PD uploaded
    topic_input = st.text_input(
        "Enter topic (if no PDF uploaded)",
        placeholder="e.g. Mitochondria, Thermodynamics..."
    )

    notes_btn = st.button("📝 Generate Notes", use_container_width=True)
    mcq_btn = st.button("❓ Generate MCQs", use_container_width=True)
    flash_btn = st.button("🃏 Generate Flashcards", use_container_width=True)
    imp_btn = st.button("⭐ Important Questions", use_container_width=True)

    #Determine content source
    def get_content(topic_input, pdf_text):
        if pdf_text:
            return pdf_text
        elif topic_input:
            return f"Topic: {topic_input}\nGenerate comprehensive content about this topic for exam preparation."
        else:
            return None


    if mcq_btn:
        content = get_content(topic_input, pdf_text)
        if content:
            with st.spinner("Generating MCQs..."):
                mcqs = generate_mcq(content, exam_type)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❓ **{exam_type} Practice MCQs**\n\n{mcqs}"
            })
            st.rerun()
        else:
            st.warning("Enter a topic or upload a PDF first.")

    if flash_btn:
        content = get_content(topic_input, pdf_text)
        if content:
            with st.spinner("Generating flashcards..."):
                flashcards = generate_flashcards(content, exam_type)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"🃏 **Flashcards**\n\n{flashcards}"
            })
            st.rerun()
        else:
            st.warning("Enter a topic or upload a PDF first.")

    if imp_btn:
        content = get_content(topic_input, pdf_text)
        if content:
            with st.spinner("Finding important questions..."):
                important = generate_important_questions(content, exam_type)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⭐ **Important Questions for {exam_type}**\n\n{important}"
            })
            st.rerun()
        else:
            st.warning("Enter a topic or upload a PDF first.")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---- MAIN CHAT AREA ----

st.title(f"PrepGPT — {exam_type} Assistant")
st.caption("Ask anything. Upload a PDF for personalized answers from your material.")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("👋 Hi! I'm PrepGPT. Ask me anything about your exam, or upload your study material for personalized help.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input(f"Ask PrepGPT anything about {exam_type}...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_with_ai(
                user_input,
                st.session_state.messages[:-1],
                exam_type,
                pdf_text
            )
        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })