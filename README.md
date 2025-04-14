📄 Resume-Job Matching App
AI-Powered Resume Analyzer
Streamline the hiring process by analyzing resumes and matching them with job descriptions using Machine Learning and NLP.
🚀 Project Overview
This application uses Natural Language Processing (NLP) and Machine Learning to:
- Extract skills, education, and experience from uploaded resumes (PDF)
- Predict resume job category (e.g., Data Scientist, HR, Developer, etc.)
- Match resume skills against job description skills
- Output a match score and summary
🧱 Project Pipeline
  Phase 1: Understanding & Setup
- Defined the problem: Resume-job compatibility.
- Dataset: UpdatedResumeDataSet.csv (resumes with categories), and sample job descriptions.
- Objective: Classification + NLP-based similarity.
  Phase 2: Research & Data Preprocessing
- Cleaned resume text using regex, removed stopwords, etc.
- Feature engineering using TF-IDF for modeling.
- Extracted key features: skills, education, experience.
  Phase 3: Model Development
- Model used: Random Forest Classifier with GridSearchCV.
- Tuned using parameters like n_estimators, max_depth, etc.
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score.
- Model saved as: resume_classifier_model.pkl and vectorizer as tfidf_vectorizer.pkl.
 Phase 4: Resume Extraction & Matching Logic
- Resume section detection using rule-based extraction: Education, Experience, Skills
- Used spaCy for named entity and POS tagging.
- Skills matched with JD using fuzzy matching or token overlap.
- Top 5 matched skills displayed with a match percentage.
⿦ Phase 5: Deployment
- Built frontend using Streamlit.
- Optional deployment to PythonAnywhere or local server.
📁 File Structure
├── app.py                       # Streamlit application
├── resume_classifier_model.pkl  # Trained classification model
├── tfidf_vectorizer.pkl         # TF-IDF vectorizer for resume text
├── UpdatedResumeDataSet.csv     # Resume dataset
├── skills.txt                   # Optional skill keywords list
├── requirements.txt             # Python package dependencies
⚙️ Setup Instructions
✅ Requirements
Install necessary packages:
pip install -r requirements.txt

Or install manually:
pip install streamlit scikit-learn pandas numpy PyPDF2 spacy joblib
python -m spacy download en_core_web_sm
▶️ Run the App
streamlit run app.py
📌 Features
- ✅ Upload PDF Resume
- 🧠 Predict Resume Category using ML
- 🧠 Extract Sections: Skills, Education, Experience
- 🔍 Compare Resume Skills with Job Description
- 📊 Show Top 5 Skills & Match Score
- 🔒 All data processed locally (no cloud upload)
🧪 Model Evaluation
Best Parameters: {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}
Accuracy: 98.96%
F1-Score (Macro): 98.7%
Evaluated using: classification_report, confusion_matrix, cross_val_score
🌍 Deployment Options
You can deploy using:
- Streamlit Cloud (1-click)
- PythonAnywhere (WSGI setup)
- Flask or FastAPI + ONNX for REST APIs
🧠 Future Improvements
- Add real-time resume parsing with OCR for scanned PDFs.
- Include more diverse skill detection with named entity recognition.
- Add job recommendation feature.
- Improve skill matching with large embeddings (e.g., BERT).
📜 License
This project is for educational and internship-level use. You are free to modify, extend, and share it with proper credit.
