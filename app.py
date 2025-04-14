import streamlit as st
import PyPDF2
import joblib
import re
import spacy
from difflib import get_close_matches
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load model and vectorizer
model = joblib.load("C:/Users/shaur/Desktop/resume_classifier_model.pkl")
vectorizer = joblib.load("C:/Users/shaur/Desktop/tfidf_vectorizer.pkl")

# ---------- Text Extraction ----------
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ---------- Improved Skill Extraction ----------
def extract_skills(text):
    doc = nlp(text.lower())
    tokens = [token.text.strip() for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    
    filtered = []
    for tok in tokens:
        if (len(tok) > 2 and tok.isalpha() and
            not tok in nlp.Defaults.stop_words and
            not re.match(r'^\+?\d+$', tok) and
            not re.search(r'@\w+|\.com|http', tok)):
            filtered.append(tok)

    freq = pd.Series(filtered).value_counts().head(5)
    return list(freq.index)

# ---------- Education Extraction ----------
def extract_education(text):
    keywords = ['bachelor', 'master', 'b.tech', 'm.tech', 'phd', 'graduation', 'degree', 'university']
    return list(set([k for k in keywords if k in text.lower()]))

# ---------- Experience Extraction ----------
def extract_experience(text):
    pattern = r'(\d+)\+?\s+(years|yrs)\s+(of)?\s+experience'
    matches = re.findall(pattern, text.lower())
    return matches[0][0] + " years" if matches else "Not found"

# ---------- Skill Matching ----------
def fuzzy_skill_match(resume_skills, jd_text):
    matched, missing = [], []
    for skill in resume_skills:
        if get_close_matches(skill, jd_text.split(), cutoff=0.7):
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("📄 Resume Matcher ")
st.markdown("Upload a PDF resume and (optionally) paste a job description to analyze compatibility.")

resume_file = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("📋 Paste Job Description (Optional)", "")

if resume_file:
    resume_text = extract_text_from_pdf(resume_file)
    cleaned_resume = re.sub(r'[^a-zA-Z ]', ' ', resume_text.lower())

    # 🔮 Predict Category
    vectorized = vectorizer.transform([cleaned_resume])
    prediction = model.predict(vectorized)[0]
    st.success(f"📂 Predicted Resume Category: `{prediction}`")

    # 🧠 Extract Key Info
    skills = extract_skills(resume_text)
    education = extract_education(resume_text)
    experience = extract_experience(resume_text)

    # 🎓 Education
    st.markdown("### 🎓 Education")
    st.write(", ".join(education) if education else "Not found")

    # 💼 Experience
    st.markdown("### 💼 Experience")
    st.write(experience)

    # 🛠️ Skills
    st.markdown("### 🛠️ Top 5 Skills Extracted")
    st.write(", ".join(skills) if skills else "No relevant skills found")

    # 🔍 Skill Matching
    if job_description.strip():
        matched, missing = fuzzy_skill_match(skills, job_description.lower())
        st.markdown("### ✅ Matched Skills with Job Description")
        st.write(", ".join(matched) if matched else "No matches")

        st.markdown("### ❌ Missing Skills from Job Description")
        st.write(", ".join(missing) if missing else "None – Great Match!")

        # 📊 Match Score
        if matched or missing:
            match_score = round((len(matched) / (len(matched) + len(missing))) * 100, 2)
            st.markdown("### 📊 Match Score")
            st.success(f"Your resume matches **{match_score}%** of the required skills.")
    else:
        st.info("📝 Job Description not provided. Showing extracted resume data only.")
else:
    st.info("📤 Please upload a PDF resume to begin.")
