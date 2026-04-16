import streamlit as st
import joblib
import numpy as np

classifier = joblib.load("../model/placement_classifier.pkl")
regressor = joblib.load("../model/salary_regressor.pkl")
course_encoder = joblib.load("../model/course_encoder.pkl")
sector_encoder = joblib.load("../model/sector_encoder.pkl")

def risk_level(cgpa, internship_score, demand):
    score = 0

    # Low CGPA increases risk
    if cgpa < 6.5:
        score += 30

    # Weak internship performance
    if internship_score < 5:
        score += 30

    # Low market demand for field
    if demand < 5:
        score += 40

    # Final risk classification
    if score >= 70:
        return "High Risk"
    elif score >= 40:
        return "Medium Risk"
    else:
        return "Low Risk"
    
st.title("AI Placement Risk Prediction System")

cgpa = st.number_input("CGPA", 0.0, 10.0)
academic = st.number_input("Academic Consistency", 0, 10)
course = st.selectbox("Course Type", ["Engineering", "MBA", "Nursing"])
internship_duration = st.number_input("Internship Duration (months)", 0, 12)
internship_score = st.number_input("Internship Score", 0, 10)
skills = st.number_input("Skills Score", 0, 10)
tier = st.number_input("Institute Tier", 1, 3)
historic_rate = st.number_input("Historic Placement Rate", 0, 100)
salary_benchmark = st.number_input("Salary Benchmark")
recruiters = st.number_input("Recruiter Count")
demand = st.number_input("Market Demand Score", 0, 10)
density = st.number_input("Region Job Density", 0, 10)
sector = st.selectbox("Sector", ["IT", "BFSI", "Healthcare"])
macro = st.number_input("Macro Index", 0, 10)
portal = st.number_input("Job Portal Activity", 0, 20)
interview = st.number_input("Interview Progress", 0, 10)

if st.button("Predict"):
    course_encoded = course_encoder.transform([course])[0]
    sector_encoded = sector_encoder.transform([sector])[0]
    
    features = np.array([[
        cgpa, academic, course_encoded, internship_duration,
        internship_score, skills, tier, historic_rate,
        salary_benchmark, recruiters, demand, density,
        sector_encoded, macro, portal, interview
    ]])

    placement = classifier.predict(features)
    salary = regressor.predict(features)

    st.write("Placed in 3 months:", placement[0][0])
    st.write("Placed in 6 months:", placement[0][1])
    st.write("Placed in 12 months:", placement[0][2])
    st.write("Predicted Salary:", salary[0])

    risk = risk_level(cgpa, internship_score, demand)
st.write("Risk Level:", risk)
