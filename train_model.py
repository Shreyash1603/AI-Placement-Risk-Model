import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
import joblib

# Load data
df = pd.read_csv("../data/placement_risk_data.csv")

# Encode categorical columns
le_course = LabelEncoder()
le_sector = LabelEncoder()

df["course_type"] = le_course.fit_transform(df["course_type"])
df["sector"] = le_sector.fit_transform(df["sector"])

# Features
X = df[
    [
        "cgpa",
        "academic_consistency",
        "course_type",
        "internship_duration",
        "internship_score",
        "skills_score",
        "institute_tier",
        "historic_placement_rate",
        "salary_benchmark",
        "recruiter_count",
        "market_demand_score",
        "region_job_density",
        "sector",
        "macro_index",
        "job_portal_activity",
        "interview_progress",
    ]
]

# Classification target
y_class = df[
    ["placed_3_months", "placed_6_months", "placed_12_months"]
]

# Salary target
y_salary = df["starting_salary"]

# Train classification model
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

classifier = MultiOutputClassifier(
    XGBClassifier(n_estimators=100, max_depth=5)
)

classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

print("3 Month Accuracy:",
      accuracy_score(y_test["placed_3_months"], pred[:, 0]))

# Train salary model
regressor = XGBRegressor(n_estimators=100, max_depth=5)

regressor.fit(X, y_salary)

salary_pred = regressor.predict(X)

print("Salary MAE:",
      mean_absolute_error(y_salary, salary_pred))

# Save models
joblib.dump(classifier, "placement_classifier.pkl")
joblib.dump(regressor, "salary_regressor.pkl")
joblib.dump(le_course, "course_encoder.pkl")
joblib.dump(le_sector, "sector_encoder.pkl")

print("Model saved successfully")