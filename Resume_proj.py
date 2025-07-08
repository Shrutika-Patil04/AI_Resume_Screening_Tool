from google.colab import drive
drive.mount('/content/drive')

import os

resume_folder = '/content/drive/My Drive/Colab Notebooks/Resumes'
resume_files = [f for f in os.listdir(resume_folder) if f.endswith(('.pdf', '.docx'))]

print("Found resumes:")
for file in resume_files:
    print(file)

!pip install python-docx pdfplumber nltk scikit-learn

import docx
import pdfplumber

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file_path):
    text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

resume_texts = []

for file in resume_files:
    path = os.path.join(resume_folder, file)
    if file.endswith('.docx'):
        text = extract_text_from_docx(path)
    elif file.endswith('.pdf'):
        text = extract_text_from_pdf(path)
    resume_texts.append((file, text))

import nltk
import re
from nltk.corpus import stopwords

# Download only stopwords
nltk.download('stopwords', quiet=True)

# Define stopwords
stop_words = set(stopwords.words('english'))

# Clean text using simple tokenizer to avoid punkt_tab error
def clean_text(text):
    text = re.sub(r'\W+', ' ', text.lower())       # Remove non-alphanumeric
    words = text.split()                           # Simple split instead of word_tokenize
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# Apply cleaning to all resumes
cleaned_resumes = [(filename, clean_text(text)) for filename, text in resume_texts]

# Print preview
for filename, cleaned in cleaned_resumes[:3]:  # Preview first 3
    print(f"Filename: {filename}\nCleaned Text: {cleaned[:300]}...\n")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 👉 Enter your job description here
job_description = """
Looking for a data analyst with experience in Python, SQL, machine learning, and dashboarding using tools like Power BI or Tableau. Familiarity with statistics and communication skills is a must.
"""

# Combine JD and resumes
texts = [job_description] + [text for _, text in cleaned_resumes]

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(texts)

# Compute cosine similarity
job_vec = vectors[0]
resume_vecs = vectors[1:]
similarities = cosine_similarity(job_vec, resume_vecs).flatten()

import numpy as np

# Get top N matching resumes (e.g., top 5)
top_n = 5
top_indices = np.argsort(similarities)[::-1][:top_n]

print("📄 Top Matching Resumes with Accuracy Percentage:\n")
for idx in top_indices:
    filename = cleaned_resumes[idx][0]
    score = similarities[idx]
    accuracy_percent = round(score * 100, 2)  # Convert to percentage
    print(f"{filename} --> Match: {accuracy_percent}%")
