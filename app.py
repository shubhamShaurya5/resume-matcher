import streamlit as st
import PyPDF2
import joblib
import re
import spacy
import os
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure spaCy model is available
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load model and vectorizer
model = joblib.load("resume_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# PDF Text Extraction
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text

# Extract section-specific info
def extract_section(text, keyword):
    pattern = rf"{keyword}.*?(?=\n[A-Z][^\n]*?:|\Z)"
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    return matches[0] if matches else ""

# Extract top 5 skills using NLP
def extract_skills(text):
    skill_text = extract_section(text, "Skills")
    if not skill_text:
        return ["No skills found"]
    doc = nlp(skill_text)
    words = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2]
    filtered = [w for w in words if not nlp.vocab[w].is_stop]
    return list(dict.fromkeys(filtered))[:5] if filtered else ["No skills found"]

# Education extractor
def extract_education(text):
    edu_section = extract_section(text, "Education")
    keywords = ['bachelor', 'master', 'b.tech', 'm.tech', 'phd', 'mba', 'msc', 'degree']
    edu_found = [kw for kw in keywords if kw in edu_section.lower()]
    return edu_found if edu_found else ["Not found"]

# Experience extractor
def extract_experience(text):
    exp_section = extract_section(text, "Experience")
    match = re.search(r"(\d+)\+?\s*(years|yrs)", exp_section.lower())
    return match.group(0) if match else "Not found"

# Fuzzy matching
def fuzzy_skill_match(skills, jd_text):
    matched, missing = [], []
    jd_words = jd_text.lower().split()
    for skill in skills:
        if get_close_matches(skill, jd_words, cutoff=0.7):
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing

# UI Setup
st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="centered")
st.title("📄 Resume Matcher App")
st.markdown("Upload your resume and (optionally) a job description to check compatibility.")

# Upload resume
resume_file = st.file_uploader("📤 Upload Resume (PDF Only)", type=["pdf"])
job_desc = st.text_area("📋 Paste Job Description (Optional)", "")

if resume_file:
    resume_text = extract_text_from_pdf(resume_file)
    cleaned_text = re.sub(r'[^a-zA-Z ]', ' ', resume_text.lower())

    # Prediction
    vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized)[0]
    st.success(f"📂 Predicted Resume Category: **{prediction}**")

    # Section Extraction
    skills = extract_skills(resume_text)
    education = extract_education(resume_text)
    experience = extract_experience(resume_text)

    st.subheader("🛠️ Top 5 Skills Extracted")
    st.write(", ".join(skills))

    st.subheader("🎓 Education")
    st.write(", ".join(education))

    st.subheader("💼 Experience")
    st.write(experience)

    # Job Description Matching
    if job_desc.strip():
        matched, missing = fuzzy_skill_match(skills, job_desc)
        st.subheader("✅ Skills Matched with Job Description")
        st.write(", ".join(matched) if matched else "No matched skills")

        st.subheader("❌ Missing Skills from Job Description")
        st.write(", ".join(missing) if missing else "None – Perfect match!")
    else:
        st.info("ℹ️ Job Description not provided. Showing extracted resume details only.")
