import re
import random
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from difflib import get_close_matches
import streamlit as st

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    training = pd.read_csv('Data/Training.csv')
    testing = pd.read_csv('Data/Testing.csv')

    # Clean column names
    training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
    testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)

    # Remove duplicate columns
    training = training.loc[:, ~training.columns.duplicated()]
    testing = testing.loc[:, ~testing.columns.duplicated()]
    return training, testing

training, testing = load_data()
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Encode target
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# ------------------ Train Model ------------------
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x, y)
    return model

model = train_model()

# ------------------ Load Dictionaries ------------------
def load_description():
    d = {}
    with open('MasterData/symptom_Description.csv') as f:
        for row in csv.reader(f):
            d[row[0]] = row[1]
    return d

def load_precaution():
    d = {}
    with open('MasterData/symptom_precaution.csv') as f:
        for row in csv.reader(f):
            d[row[0]] = [row[1], row[2], row[3], row[4]]
    return d

def load_symptom_severity():
    d = {}
    with open('MasterData/symptom_severity.csv') as f:
        for row in csv.reader(f):
            try:
                d[row[0]] = int(row[1])
            except:
                pass
    return d

description_list = load_description()
precautionDictionary = load_precaution()
severityDictionary = load_symptom_severity()
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

# ------------------ Symptom Extractor ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy pain": "stomach_pain",
    "loose motion": "diarrhea",
    "motions": "diarrhea",
    "high temperature": "fever",
    "temperature": "fever",
    "feaver": "fever",
    "coughing": "cough",
    "throat pain": "sore_throat",
    "cold": "chills",
    "breathing issue": "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",
}

def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")

    # 1. Synonym replacement
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    # 2. Exact match
    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

    # 3. Fuzzy match
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.8)
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)

    return list(set(extracted))

# ------------------ Prediction ------------------
def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)
    return disease, confidence

# ------------------ Streamlit App ------------------
st.title("💊 HealthCare ChatBot")
st.write("Describe your symptoms and answer a few questions to get a probable diagnosis.")

name = st.text_input("What is your name?")
age = st.number_input("Your age", min_value=0, max_value=120, value=25)
gender = st.selectbox("Your gender", ["M", "F", "Other"])

symptoms_input = st.text_area("Describe your symptoms (e.g., 'I have fever and stomach pain'):")
num_days = st.number_input("For how many days have you had these symptoms?", min_value=0, max_value=100)
severity_scale = st.slider("On a scale of 1–10, how severe is it?", 1, 10, 5)

pre_exist = st.text_input("Do you have any pre-existing conditions (e.g., diabetes)?")
lifestyle = st.text_input("Do you smoke, drink alcohol, or have irregular sleep?")
family = st.text_input("Any family history of similar illness?")

if st.button("Submit"):
    symptoms_list = extract_symptoms(symptoms_input, cols)
    if not symptoms_list:
        st.error("Sorry, no valid symptoms detected. Try describing in more detail.")
    else:
        disease, confidence = predict_disease(symptoms_list)
        st.success(f"Based on your symptoms, you may have: **{disease}**")
        st.info(f"Confidence: {confidence}%")
        st.write(f"**About:** {description_list.get(disease, 'No description available.')}")

        if disease in precautionDictionary:
            st.write("**Suggested precautions:**")
            for i, p in enumerate(precautionDictionary[disease], 1):
                st.write(f"{i}. {p}")

        st.write("💡 " + random.choice([
            "Health is wealth, take care of yourself.",
            "A healthy outside starts from the inside.",
            "Every day is a chance to get stronger and healthier.",
            "Take a deep breath, your health matters the most.",
            "Remember, self-care is not selfish."
        ]))
