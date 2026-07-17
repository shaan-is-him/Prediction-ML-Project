import streamlit as st
import pandas as pd
import joblib


def load_model():
    return joblib.load("Models/placement_model.joblib")

model_data = load_model()

model = model_data["model"]
scaler = model_data["scaler"]
feature_columns = model_data["columns"]


st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓")

st.title(" Student Placement Predictor")
st.write("Fill in the student's details below.")


col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
    internships = st.number_input("Internships", 0, 20, 1)
    projects = st.number_input("Projects", 0, 20, 2)
    certifications = st.number_input("Certifications", 0, 20, 2)
    aptitude = st.slider("Aptitude Score", 0, 100, 75)
    backlogs = st.number_input("Backlogs", 0, 20, 0)
    hackathons = st.number_input("Hackathons", 0, 20, 1)

with col2:
    coding = st.slider("Coding Skills", 1, 10, 7)
    dsa = st.slider("DSA Score", 1, 10, 6)
    communication = st.slider("Communication Skills", 1, 10, 7)
    ml = st.slider("ML Knowledge", 1, 10, 5)
    system_design = st.slider("System Design", 1, 10, 5)
    open_source = st.number_input("Open Source Contributions", 0, 100, 0)
    extracurricular = st.number_input("Extracurricular Activities", 0, 20, 2)

branch = st.selectbox(
    "Branch",
    [
        "Others",
        "CSE",
        "Chemical",
        "ECE",
        "EE",
        "IT",
        "ME"
    ]
)

college_tier = st.selectbox(
    "College Tier",
    [
        "Tier-1",
        "Tier-2",
        "Tier-3"
    ]
)


if st.button("🔮 Predict Placement", type="primary"):

    # making a dataframe with all expected columns
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

    # all the numerical features/columns of dataset
    input_data["cgpa"] = cgpa
    input_data["backlogs"] = backlogs
    input_data["coding_skills"] = coding
    input_data["dsa_score"] = dsa
    input_data["aptitude_score"] = aptitude
    input_data["communication_skills"] = communication
    input_data["ml_knowledge"] = ml
    input_data["system_design"] = system_design
    input_data["internships"] = internships
    input_data["projects_count"] = projects
    input_data["certifications"] = certifications
    input_data["hackathons"] = hackathons
    input_data["open_source_contributions"] = open_source
    input_data["extracurriculars"] = extracurricular

    # encoding the branch
    if branch != "Others":
        column = f"branch_{branch}"
        if column in input_data.columns:
            input_data[column] = 1

    # encoding college tier
    if college_tier != "Tier-1":
        column = f"college_tier_{college_tier}"
        if column in input_data.columns:
            input_data[column] = 1

    # scaling before prediction
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Student is likely to be PLACED")
        st.progress(float(probability))
        st.write(f"Confidence: **{probability:.2%}**")
    else:
        st.error("❌ Student is likely NOT to be placed")
        st.progress(float(1 - probability))
        st.write(f"Confidence: **{(1-probability):.2%}**")


    st.subheader("Recommended Job Roles")

    roles = []

    if coding >= 7 and dsa >= 6 and system_design >= 5:
        roles.append("**Software Engineer / Backend Developer**")
    if coding >= 8 and dsa >= 7:
        roles.append("**Full Stack Developer**")
    if ml >= 7 and cgpa >= 7.5:
        roles.append("**Machine Learning Engineer / Data Scientist**")
    if system_design >= 7 and communication >= 7:
        roles.append("**System Architect / Solutions Architect**")
    if communication >= 8 and cgpa >= 7.0:
        roles.append("**Technical Product Manager**")
    if projects >= 3 and open_source >= 5:
        roles.append("**Open Source Contributor / Software Developer**")
    if internships >= 2 and aptitude >= 70:
        roles.append("**SDE Intern → Full-time Role**")
    if ml >= 6 and certifications >= 3:
        roles.append("**AI/ML Specialist**")
    if coding >= 6 and branch in ["ECE", "EE"]:
        roles.append("**Embedded Systems Engineer**")


    if not roles:
        roles.append("**General Software Developer** (Build more projects and skills)")

    for role in roles[:5]: 
        st.write(f"• {role}")

    st.info("*These recommendations are based on your input skills. Focus on improving weaker areas for better opportunities.*")


st.sidebar.title("About")

st.sidebar.info(
"""
This application predicts whether a student is likely to be placed based on academic performance and skills.

**Model:** Logistic Regression

**Preprocessing**
- One-Hot Encoding
- StandardScaler
- SMOTE

Built using Streamlit & Scikit-learn.
"""
)
