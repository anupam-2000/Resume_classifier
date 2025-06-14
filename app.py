# ================== IMPORT LIBRARIES ==================
import re
import pandas as pd
import streamlit as st
import pdfplumber
import docx2txt
import spacy
import pickle as pk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Load NLP model
import subprocess
import importlib.util

model_name = "en_core_web_sm"
if importlib.util.find_spec(model_name) is None:
    subprocess.run(["python", "-m", "spacy", "download", model_name])

nlp = spacy.load(model_name)

# ================== STREAMLIT CONFIG ==================
st.set_page_config(page_title="Resume Classifier", layout="wide")

# ================== CUSTOM CSS ==================
st.markdown("""
    <style>
        body {
            background-color: #fdf6fd;
        }
        h1, h2, h3 {
            color: #6A1B9A;
        }
        .stButton>button {
            background-color: #8E24AA;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #6A1B9A;
        }
        .stSelectbox, .stFileUploader {
            background-color: #f3e5f5 !important;
        }
        .stDataFrame {
            border: 2px solid #ce93d8;
            border-radius: 10px;
            padding: 1em;
        }
        .sidebar-title {
            font-size: 22px;
            font-weight: bold;
            color: #6A1B9A;
        }
        .sidebar-content {
            font-size: 16px;
            line-height: 1.5;
            color: #4A148C;
        }
    </style>
""", unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.markdown('<p class="sidebar-title">📘 About the App</p>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="sidebar-content">
This is an <b>AI-powered Resume Classifier</b> built using Machine Learning and NLP techniques.<br><br>

🔍 <b>Features</b>:
- Auto-extract candidate names & skills
- Predict candidate job profiles
- Upload and classify multiple resumes in seconds<br><br>

📂 <b>Formats</b>: PDF, DOCX<br>
📤 Just upload and get instant results!<br><br>

📧 <b>Contact</b>: youremail@example.com
</div>
""", unsafe_allow_html=True)

# ================== MAIN TITLE ==================
st.title('📄 AI-Powered Resume Classifier')
st.subheader('✨ Instantly analyze and categorize candidate resumes')

# ================== FUNCTIONS ==================
@st.cache_data
def load_skills():
    df = pd.read_csv("skills.csv")
    return set(df.columns.str.lower())

def extract_skills(resume_text, skills):
    nlp_text = nlp(resume_text)
    tokens = [token.text.lower() for token in nlp_text if not token.is_stop]
    noun_chunks = [chunk.text.lower().strip() for chunk in nlp_text.noun_chunks]
    matched_skills = set()

    for word in tokens + noun_chunks:
        if word in skills:
            matched_skills.add(word.capitalize())
    return list(matched_skills)

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Not Found"

def get_text(file):
    try:
        if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return docx2txt.process(file)
        else:
            text = ''
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            return text
    except Exception:
        return ""

def preprocess(text):
    text = re.sub(r'<.*?>', '', text.lower())
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in tokens if w not in stop_words and len(w) > 2]

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered_words]

    return " ".join(lemmatized)

# ================== LOAD MODEL AND VECTORIZER ==================
model = pk.load(open('modelDT.pkl', 'rb'))
vectorizer = pk.load(open('vector.pkl', 'rb'))
skills_list = load_skills()

# ================== FILE UPLOADER ==================
uploaded_files = st.file_uploader("📤 Upload Resumes (.pdf or .docx)", type=['pdf', 'docx'], accept_multiple_files=True)

if uploaded_files:
    result_df = pd.DataFrame(columns=['Candidate Name', 'Uploaded File', 'Predicted Profile', 'Skills'])

    with st.spinner("🔍 Analyzing Resumes... Please wait..."):
        for file in uploaded_files:
            full_text = get_text(file)
            if not full_text.strip():
                continue  # Skip empty files

            name = extract_name(full_text)
            cleaned_text = preprocess(full_text)
            vectorized_text = vectorizer.transform([cleaned_text])
            profile = model.predict(vectorized_text)[0]
            extracted_skills = extract_skills(full_text, skills_list)

            result_df = pd.concat([result_df, pd.DataFrame({
                'Candidate Name': [name],
                'Uploaded File': [file.name],
                'Predicted Profile': [profile],
                'Skills': [extracted_skills]
            })], ignore_index=True)

    if not result_df.empty:
        st.success("✅ Classification Complete!")
        st.subheader("📊 Resume Summary Table")
        st.dataframe(result_df, use_container_width=True)

        st.markdown("### 🔎 Filter Resumes by Predicted Profile")
        unique_profiles = sorted(result_df['Predicted Profile'].unique())
        selected_profile = st.selectbox("🎯 Select Profile", unique_profiles)

        filtered = result_df[result_df['Predicted Profile'] == selected_profile]
        if not filtered.empty:
            st.dataframe(filtered, use_container_width=True)
        else:
            st.warning("⚠️ No resumes found for the selected profile.")
    else:
        st.warning("⚠️ No valid resumes were processed.")
else:
    st.info("👈 Upload one or more resumes to get started!")
