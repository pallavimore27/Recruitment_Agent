# 🚀 AI-Powered Recruitment Agent

> Smart Resume Analysis & Interview Preparation System

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [File Structure](#-file-structure)
- [Tech Stack & Components](#-tech-stack--components)
- [Detailed Component Breakdown](#-detailed-component-breakdown)
- [How It Works](#-how-it-works)
- [Installation & Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
- [Usage Guide](#-usage-guide)
- [Supported Roles](#-supported-roles)
- [Environment Variables](#-environment-variables)
- [Troubleshooting](#-troubleshooting)

---

## 📖 Project Overview

The **AI-Powered Recruitment Agent** is a full-stack intelligent recruitment automation system built as a final-year academic project. It leverages Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and semantic vector search to automate and enhance the resume screening and interview preparation process.

The system allows recruiters or hiring managers to:
- Upload a candidate's resume (PDF or TXT)
- Select a target role or provide a custom Job Description (JD)
- Get an AI-powered ATS (Applicant Tracking System) style score
- Receive deep skill analysis, weakness identification, and actionable improvement suggestions
- Ask natural language questions directly about the resume
- Auto-generate personalized interview questions
- Generate a fully improved, ATS-optimized version of the resume

---

## ✨ Features

| Feature | Description |
|---|---|
| **Resume Parsing** | Extracts text from PDF and TXT resumes |
| **Skill Extraction from JD** | Uses LLM to extract key skills from any pasted job description |
| **ATS Score** | Calculates an overall fit score (0–100%) against role requirements |
| **Semantic Skill Analysis** | Scores each skill (0–10) using FAISS vector store + RAG pipeline |
| **Weakness Analysis** | Identifies missing/weak skills with detailed JSON-structured feedback |
| **Resume Q&A** | Ask freeform questions about the resume, answered via RAG |
| **Interview Question Generation** | Generates role-specific questions by type (Technical, Behavioral, Coding) and difficulty |
| **Resume Improvement Suggestions** | Provides structured, actionable improvement tips per section |
| **AI Resume Rewriting** | Generates a fully rewritten, ATS-optimized resume |
| **Dark-Themed UI** | Professional dark UI built with Streamlit and custom CSS |

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI (ui.py)                 │
│   Sidebar │ Tab 1 │ Tab 2 │ Tab 3 │ Tab 4 │ Tab 5           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     App Controller (app.py)                 │
│   Session State Management │ Role Requirements │ Routing    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               ResumeAnalysisAgent (agents.py)               │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ PDF/TXT     │  │  FAISS       │  │  Groq LLM         │  │
│  │ Parsing     │  │  Vector Store│  │  (LLaMA 3.3-70B)  │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              LCEL RAG Pipeline                       │   │
│  │  Retriever → PromptTemplate → LLM → StrOutputParser │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Data Flow:**
```
Resume Upload → Text Extraction → FAISS Vector Store → RAG Chain
     ↓                                                     ↓
JD / Role Skills → Skill Extraction (LLM) → Semantic Scoring
     ↓                                                     ↓
Weakness Analysis → Improvement Suggestions → Rewritten Resume
```

---

## 📁 File Structure

```
Recruitment_Agent/
│
├── agents.py           # Core AI logic — ResumeAnalysisAgent class
├── app.py              # Streamlit app controller, session management, routing
├── ui.py               # All UI components — tabs, sections, CSS, charts
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── euron.jpg           # (Optional) Logo image for the header
```

### File Responsibilities

| File | Role | Lines |
|---|---|---|
| `agents.py` | Business logic, LLM calls, RAG pipeline, FAISS | ~530 |
| `app.py` | App entry point, tab routing, session state | ~135 |
| `ui.py` | Streamlit components, CSS, charts, download | ~625 |

---

## 🛠️ Tech Stack & Components

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.31.0 | Web UI framework — renders the entire frontend |
| `langchain` | 0.1.5 | LLM orchestration framework |
| `langchain-groq` | latest | Groq integration for LangChain |
| `langchain-openai` | 0.0.5 | OpenAI-compatible LangChain adapter |
| `langchain-community` | 0.0.17 | Community integrations (FAISS, FakeEmbeddings) |
| `openai` | 1.10.0 | OpenAI SDK (used as backend for Groq-compatible calls) |
| `PyPDF2` | 3.0.1 | PDF text extraction |
| `faiss-cpu` | 1.7.4 | Facebook AI Similarity Search — vector database |
| `pandas` | 2.1.4 | Data handling for skill tables and results |
| `python-dotenv` | 1.0.0 | Environment variable management (.env support) |
| `matplotlib` | 3.8.2 | Charts and skill score visualizations |

### LLM Model

| Property | Value |
|---|---|
| Provider | Groq |
| Model | `llama-3.3-70b-versatile` |
| Access | Via `langchain_groq.ChatGroq` |
| API Key | Required (free tier available at console.groq.com) |

---

## 🔍 Detailed Component Breakdown

### 1. `agents.py` — ResumeAnalysisAgent

The brain of the entire system. A single Python class that encapsulates all AI operations.

#### Key Methods

| Method | What It Does |
|---|---|
| `get_llm()` | Returns a `ChatGroq` instance using `llama-3.3-70b-versatile` |
| `extract_text_from_pdf()` | Reads a PDF file (or BytesIO buffer from Streamlit) using `PyPDF2` |
| `extract_text_from_txt()` | Reads a plain text resume file |
| `create_rag_vector_store()` | Splits resume text into 1000-char chunks, creates FAISS vector store with `FakeEmbeddings` |
| `extract_skills_from_jd()` | Prompts the LLM to return a Python list of skills from any JD text |
| `analyze_skill()` | Uses the RAG chain to score a single skill from 0–10 |
| `semantic_skill_analysis()` | Runs skill scoring for all extracted skills sequentially, computes overall ATS score |
| `analyze_resume_weaknesses()` | For each missing skill, calls LLM to generate JSON with weakness, suggestions, and an example bullet point |
| `ask_questions()` | Retrieves top-3 relevant resume chunks via FAISS and answers via LCEL chain |
| `generate_interview_questions()` | Generates N questions of specified types and difficulty based on resume context |
| `improve_resume()` | Returns structured improvement suggestions per section |
| `get_improved_resume()` | Rewrites the entire resume optimized for the JD and ATS |
| `cleanup()` | Deletes temporary files created during execution |

#### RAG Pipeline (LCEL)

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | PromptTemplate.from_template(...)
    | ChatGroq(model="llama-3.3-70b-versatile")
    | StrOutputParser()
)
```

This is a **LangChain Expression Language (LCEL)** chain — the modern replacement for the deprecated `RetrievalQA` chain. It:
1. Takes a question
2. Retrieves relevant chunks from FAISS
3. Inserts them into a prompt template
4. Calls the LLM
5. Parses the string output

---

### 2. `app.py` — Application Controller

The Streamlit entry point. Manages:

- **Session State**: Stores `resume_agent`, `resume_analyzed`, and `analysis_result` across reruns
- **Role Requirements Dictionary**: Preloaded skill lists for 9 job roles (AI/ML Engineer, Backend, Frontend, Data Scientist, etc.)
- **Tab Routing**: Maps each of the 5 tabs to the appropriate `ui.*` function and agent method
- **Agent Initialization**: Creates `ResumeAnalysisAgent` once and reuses it via `st.session_state`
- **Cleanup Hook**: Uses `atexit.register()` to delete temp files when the app exits

#### Supported Roles (Built-in)

`AI/ML Engineer`, `Frontend Engineer`, `Backend Engineer`, `Data Engineer`, `DevOps Engineer`, `Full Stack Developer`, `Product Manager`, `Data Scientist`, `Software Engineer`

---

### 3. `ui.py` — User Interface Layer

Handles all Streamlit rendering and custom HTML/CSS. Completely decoupled from agent logic — receives data and callback functions as arguments.

#### UI Sections

| Function | Tab | Description |
|---|---|---|
| `setup_page()` | Global | Injects custom CSS, sets dark theme |
| `display_header()` | Global | Renders title, logo (or fallback emoji) |
| `setup_sidebar()` | Sidebar | Groq API key input, ATS cutoff slider |
| `create_tabs()` | Global | Creates the 5 main navigation tabs |
| `role_selection_section()` | Tab 1 | Dropdown for role or text area for custom JD |
| `resume_upload_section()` | Tab 1 | File uploader (PDF/TXT) |
| `display_analysis_results()` | Tab 1 | Score gauge, skill chart, strength/weakness cards |
| `resume_qa_section()` | Tab 2 | Chat-style Q&A with the resume |
| `interview_questions_section()` | Tab 3 | Question type checkboxes, difficulty, count |
| `resume_improvement_section()` | Tab 4 | Improvement area selector, expandable suggestions |
| `improved_resume_section()` | Tab 5 | Target role/JD input, rewrite display, download |

#### Custom Theme

The UI uses a **black background** (`#000000`) with a **red accent** (`#d32f2f`) applied via injected CSS. All input fields, buttons, alerts, and tab indicators follow this theme.

---

## 🔄 How It Works

### Step-by-Step Flow

```
1. User enters Groq API Key in the sidebar
2. User selects a Role OR pastes a custom Job Description
3. User uploads a Resume (PDF or TXT)
4. Clicks "🔎 Analyze Resume"
      ↓
5. agents.py:
   a. Extracts text from the resume (PyPDF2 or plain text)
   b. If custom JD → LLM extracts required skills from JD text
   c. If role selected → uses hardcoded skill list from app.py
   d. Creates FAISS vector store from resume chunks
   e. For each skill → RAG chain scores it 0–10
   f. Computes overall ATS score (sum / max * 100)
   g. For skills ≤ 5 → LLM generates detailed weakness JSON
      ↓
6. ui.py renders results:
   - ATS score percentage + selection status
   - Matplotlib bar chart of skill scores
   - Strengths and weakness cards
      ↓
7. Subsequent tabs are unlocked:
   - Tab 2: Ask questions about the resume
   - Tab 3: Generate interview questions
   - Tab 4: Get improvement suggestions
   - Tab 5: Download AI-rewritten resume
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- A free Groq API key → [console.groq.com](https://console.groq.com)

### Step 1: Clone the Repository

```bash
git clone https://github.com/pallavimore27/Recruitment_Agent.git
cd Recruitment_Agent
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you encounter version conflicts, install these individually:
> ```bash
> pip install streamlit==1.31.0
> pip install langchain==0.1.5
> pip install langchain-groq
> pip install langchain-community==0.0.17
> pip install PyPDF2==3.0.1
> pip install faiss-cpu==1.7.4
> pip install pandas==2.1.4
> pip install matplotlib==3.8.2
> pip install python-dotenv==1.0.0
> ```

### Step 4: Configure API Key

**Option A — Enter in the UI (Recommended for demos):**
Simply paste your Groq API key in the sidebar when the app launches.

**Option B — Environment File:**
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Running the Project

```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501` in your browser automatically.

To run on a specific port:
```bash
streamlit run app.py --server.port 8080
```

To make it accessible on your local network:
```bash
streamlit run app.py --server.address 0.0.0.0
```

---

## 📋 Usage Guide

### Tab 1 — Resume Analysis
1. Enter your **Groq API Key** in the left sidebar
2. Select a **Job Role** from the dropdown, OR paste a custom **Job Description**
3. Upload a **resume** (PDF or TXT format)
4. Click **🔎 Analyze Resume**
5. View your ATS score, skill chart, strengths, and weaknesses

### Tab 2 — Resume Q&A
- Ask any question about the candidate's resume in natural language
- Examples: *"What is the candidate's total years of experience?"*, *"Does the candidate have cloud certifications?"*

### Tab 3 — Interview Questions
- Select question types: **Technical**, **Behavioral**, **Coding**, **Situational**
- Choose difficulty: **Easy**, **Medium**, **Hard**
- Set number of questions (default: 5)
- Click generate

### Tab 4 — Resume Improvement
- Select improvement areas: Skills Highlighting, Work Experience, Summary, Education, etc.
- Optionally specify a target role
- Get structured suggestions with before/after examples

### Tab 5 — AI-Rewritten Resume
- Optionally enter a target role or paste a JD
- Click to generate a fully rewritten, ATS-optimized resume
- Download the result as a `.txt` file

---

## 💼 Supported Roles

| Role | Sample Skills Evaluated |
|---|---|
| AI/ML Engineer | Python, PyTorch, TensorFlow, NLP, HuggingFace, MLOps |
| Frontend Engineer | React, TypeScript, Next.js, Tailwind CSS, GraphQL |
| Backend Engineer | Python, Node.js, Docker, Kubernetes, REST APIs |
| Data Engineer | Spark, Kafka, Airflow, BigQuery, ETL, Snowflake |
| DevOps Engineer | Terraform, CI/CD, Helm, Prometheus, Kubernetes |
| Full Stack Developer | JavaScript, React, Node.js, MongoDB, REST APIs |
| Product Manager | Agile, Roadmapping, A/B Testing, User Research |
| Data Scientist | Python, R, Pandas, ML, Statistics, Feature Engineering |
| Software Engineer | Java, DSA, OOP, Git, SQL, REST APIs |

---

## 🔐 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key for LLaMA model access | Yes (or enter in UI) |

---

## 🐛 Troubleshooting

### `ModuleNotFoundError: No module named 'langchain_text_splitters'`
```bash
pip install langchain-text-splitters
```

### `ImportError` for FAISS
```bash
pip install faiss-cpu==1.7.4
```

### Groq Rate Limit Errors
The system processes skills **sequentially** (not in parallel) to avoid Groq rate limits. If you still encounter limits:
- Reduce the number of skills being evaluated
- Use a role with fewer required skills
- Wait a few seconds and retry

### PDF Text Not Extracting
- Ensure the PDF is not scanned/image-based (the system cannot OCR images)
- Try converting to TXT first using any PDF-to-text tool

### Streamlit Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

---

## 👨‍💻 Author

**Pallavi More** — Final Year AI/ML Project  
GitHub: [github.com/pallavimore27/Recruitment_Agent](https://github.com/pallavimore27/Recruitment_Agent)

---

## 📄 License

This project is developed for academic and educational purposes.