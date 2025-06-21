import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
X = df
y = data.target

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("🩺 Breast Cancer Prediction App")
st.write("Enter values for tumor features to predict if it's benign or malignant.")

# Sidebar for user input
def user_input_features():
    input_data = {}
    for feature in data.feature_names[:5]:  # Using first 5 features for simplicity
        input_data[feature] = st.sidebar.slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# Output
st.subheader("Prediction")
st.write("Malignant" if prediction == 0 else "Benign")
st.subheader("Prediction Probability")
st.write(f"Malignant: {prediction_proba[0]:.2f}, Benign: {prediction_proba[1]:.2f}")

# Feature Importance Graph
st.subheader("Feature Importances (Top 10)")
importances = model.feature_importances_
indices = importances.argsort()[-10:][::-1]
plt.figure(figsize=(8,5))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [data.feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
st.pyplot(plt)
