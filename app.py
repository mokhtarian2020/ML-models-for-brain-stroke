import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Load and preprocess the dataset
def load_data():
    brain_stroke = pd.read_csv('brain_stroke.csv')

    # Encoding categorical variables
    label_encoders = {}
    for col in ['gender', 'work_type', 'Residence_type', 'smoking_status']:
        label_encoders[col] = LabelEncoder()
        brain_stroke[col] = label_encoders[col].fit_transform(brain_stroke[col])

    # Convert binary variables to integers
    binary_cols = ['hypertension', 'heart_disease']
    for col in binary_cols:
        brain_stroke[col] = brain_stroke[col].astype(int)

    # Convert 'ever_married' to binary
    brain_stroke['ever_married'] = (brain_stroke['ever_married'] == 'Yes').astype(int)

    # Removing outliers in numerical variables using IQR method
    def remove_outliers(df, cols):
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    brain_stroke = remove_outliers(brain_stroke, numerical_cols)

    return brain_stroke

# Prepare the data for training
def prepare_data(brain_stroke):
    X = brain_stroke.drop(columns=['stroke'])
    y = brain_stroke['stroke']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled, X_test, y_test

# Train the logistic regression model
def train_model(X_resampled, y_resampled):
    model = LogisticRegression()
    model.fit(X_resampled, y_resampled)
    return model

# Make predictions
def predict(model, X_test):
    return model.predict(X_test)

# Load and preprocess the data
brain_stroke = load_data()
X_resampled, y_resampled, X_test, y_test = prepare_data(brain_stroke)

# Train the model
model = train_model(X_resampled, y_resampled)

# Streamlit interface
st.title("Brain Stroke Prediction")

st.write("""
### Enter the following details:
""")

# Collect user input
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 100)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.slider("Average Glucose Level", 0.0, 300.0)
bmi = st.slider("BMI", 0.0, 50.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Encode user input
user_data = {
    "gender": [label_encoders['gender'].transform([gender])[0]],
    "age": [age],
    "hypertension": [1 if hypertension == "Yes" else 0],
    "heart_disease": [1 if heart_disease == "Yes" else 0],
    "ever_married": [1 if ever_married == "Yes" else 0],
    "work_type": [label_encoders['work_type'].transform([work_type])[0]],
    "Residence_type": [label_encoders['Residence_type'].transform([residence_type])[0]],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "smoking_status": [label_encoders['smoking_status'].transform([smoking_status])[0]]
}

user_df = pd.DataFrame(user_data)

# Preprocess the user input
user_df = imputer.transform(user_df)
user_df = scaler.transform(user_df)

# Predict the probability of stroke
prediction = model.predict(user_df)

if prediction[0] == 1:
    st.write("## The model predicts that you are at risk of a brain stroke.")
else:
    st.write("## The model predicts that you are not at risk of a brain stroke.")
