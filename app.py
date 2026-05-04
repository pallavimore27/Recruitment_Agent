import streamlit as st
#this must be the first streamlit command
st.set_page_config(
    page_title="Recruitment Agent",
    page_icon="🚀",
    layout="wide"
)

import ui
from agents import ResumeAnalysisAgent
import atexit

#Role requirements dictionary
ROLE_REQUIREMENTS = {
    "AI/ML Engineer": [
        "Python", "PyTorch", "TensorFlow", "Machine Learning", "Deep Learning", "Scikit-learn", "Data Analysis", "NLP", "Computer Vision",
        "Reinforcement Learning",
        "Hugging Face", "AutoML"
    ],

   "Frontend Engineer": [
        "React", "Vue", "Angular", "HTML5", "CSS3", "JavaScript", "TypeScript",
        "Next.js", "Svelte", "Bootstrap", "Tailwind CSS", "GraphQL", "Redux",
        "WebAssembly", "Three.js", "Performance Optimization"
    ],

    "Backend Engineer": [
        "Python", "Java", "Node.js", "REST APIs", "Cloud services", "Kubernetes",
        "Docker", "GraphQL", "Microservices", "gRPC", "Spring Boot", "Flask",
        "FastAPI", "SQL & NoSQL Databases", "Redis", "RabbitMQ", "CI/CD"
    ],

    "Data Engineer": [
        "Python", "SQL", "Apache Spark", "Hadoop", "Kafka", "ETL Pipelines",
        "Airflow", "BigQuery", "Redshift", "Data Warehousing", "Snowflake",
        "Azure Data Factory", "GCP", "AWS Glue", "DBT"
    ],

    "DevOps Engineer": [
        "Kubernetes", "Docker", "Terraform", "CI/CD", "AWS", "Azure", "GCP",
        "Jenkins", "Ansible", "Prometheus", "Grafana", "Helm",
        "Linux Administration", "Networking", "Site Reliability Engineering (SRE)"
    ],

    "Full Stack Developer": [
        "JavaScript", "TypeScript", "React", "Node.js", "Express", "MongoDB",
        "SQL", "HTML5", "CSS3", "RESTful APIs", "Git", "CI/CD", "Cloud Services",
        "Responsive Design", "Authentication & Authorization"
    ],

    "Product Manager": [
        "Product Strategy", "User Research", "Agile Methodologies", "Roadmapping",
        "Market Analysis", "Stakeholder Management", "Data Analysis", "User Stories",
        "Product Lifecycle", "A/B Testing", "KPI Definition", "Prioritization",
        "Competitive Analysis", "Customer Journey Mapping"
    ],

    "Data Scientist": [
        "Python", "R", "SQL", "Machine Learning", "Statistics", "Data Visualization",
        "Pandas", "NumPy", "Scikit-learn", "Jupyter", "Hypothesis Testing",
        "Experimental Design", "Feature Engineering", "Model Evaluation"
    ],
    "Software Engineer":[
    "Java", "Python", "JavaScript","Data Structures & Algorithms", "Object-Oriented Programming","SQL", "Database Management Systems","Git", "REST APIs","Debugging", "Problem Solving","Agile Methodology"
    ]
}

#Initialize session state variables
if 'resume_agent' not in st.session_state:
    st.session_state.resume_agent = None

if 'resume_analyzed' not in st.session_state:
    st.session_state.resume_analyzed = False

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None


def setup_agent(config):
    """Set up the resume analysis agent with the provided configuration"""
    if not config["groq_api_key"]:
        st.error("Please enter your Groq API key in the sidebar.")
        return None
    
    if st.session_state.resume_agent is None:
        st.session_state.resume_agent = ResumeAnalysisAgent(api_key=config["groq_api_key"])
    else:
        st.session_state.resume_agent.api_key = config["groq_api_key"]

    return st.session_state.resume_agent

def analyze_resume(agent, resume_file, role, custom_jd):
    """Analyze the resume with the agent"""
    if not resume_file:
        st.error("Please upload a resume.")
        return None
    
    try:
        with st.spinner("Analyzing resume... This may take a minute."):
            if custom_jd:
                result = agent.analyze_resume(resume_file, custom_jd=custom_jd)
            else:
                result = agent.analyze_resume(resume_file, role_requirements=ROLE_REQUIREMENTS[role])
            
            st.session_state.resume_analyzed = True
            st.session_state.analysis_result = result
            return result
    except Exception as e:
        st.error(f"Error analyzing resume: {e}")
        return None

def ask_question(agent, question):
    """Ask a question about the resume"""
    try:
        response = agent.ask_questions(question)
        return response
    except Exception as e:
        return f"Error: {e}"
    
def generate_interview_questions(agent, question_types, difficulty, num_questions):
    """Generate interview questions based on the analysis"""
    try:
        with st.spinner("Generating interview questions..."):
            questions = agent.generate_interview_questions(question_types, difficulty, num_questions)
            return questions
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []
    
def improve_resume(agent, improvement_areas, target_role):
    """Generate resume improvement suggestions"""
    try:
        with st.spinner("Analyzing and generating improvement suggestions..."):
            return agent.improve_resume(improvement_areas, target_role)
    except Exception as e:
        st.error(f"Error generating suggestions: {e}")
        return {}
    
def get_improved_resume(agent, target_role, highlight_skills):
    """Get an improved version of the resume"""
    try:
        with st.spinner("Creating improved resume..."):
            return agent.get_improved_resume(target_role, highlight_skills)
    except Exception as e:
        st.error(f"Error creating improved resume: {e}")
        return "Error generating improved resume."
    
def cleanup():
    """Clean up when the app exits"""
    if st.session_state.resume_agent:
        st.session_state.resume_agent.cleanup()

atexit.register(cleanup)

def main():
    ui.setup_page()
    ui.display_header()

    config = ui.setup_sidebar()

    agent = setup_agent(config)

    tabs = ui.create_tabs()

    # Tab 1: Resume analysis
    with tabs[0]:
        role, custom_jd = ui.role_selection_section(ROLE_REQUIREMENTS)
        uploaded_resume = ui.resume_upload_section()

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔎 Analyze Resume", type="primary"):
                if agent and uploaded_resume:
                    analyze_resume(agent, uploaded_resume, role, custom_jd)

        if st.session_state.analysis_result:
            ui.display_analysis_results(st.session_state.analysis_result)

    # Tab 2: Resume Q&A
    with tabs[1]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.resume_qa_section(
                has_resume=True,
                ask_question_func=lambda q: ask_question(st.session_state.resume_agent, q)
            )
        else:
            st.warning("Please upload and analyze a resume first in the Resume Analysis tab.")

    # Tab 3: Interview Questions
    with tabs[2]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.interview_questions_section(
                has_resume=True,
                generate_questions_func=lambda types, diff, num:
                generate_interview_questions(st.session_state.resume_agent, types, diff, num)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")
    
    # Tab 4: Resume improvement
    with tabs[3]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.resume_improvement_section(
                has_resume=True,
                improve_resume_func=lambda areas, role:
                improve_resume(st.session_state.resume_agent, areas, role),
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    # Tab 5: Improved resume generation
    with tabs[4]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.improved_resume_section(
                has_resume=True,
                get_improved_resume_func=lambda role, skills:
                get_improved_resume(st.session_state.resume_agent, role, skills)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

if __name__ == "__main__":
    main()