import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.title("Heart Disease Data Preprocessing")

data = pd.read_csv("heart.csv")

st.subheader("Original Dataset")
st.write(data.head())

# Check missing values
st.subheader("Missing Values")
st.write(data.isnull().sum())

# Fill missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Feature scaling
scaler = StandardScaler()
num_cols = ['age','trestbps','chol','thalach','oldpeak']
data[num_cols] = scaler.fit_transform(data[num_cols])

st.subheader("After Scaling")
st.write(data.head())

# Graph 1
st.subheader("Heart Disease Count")
fig1, ax1 = plt.subplots()
sns.countplot(x='target', data=data, ax=ax1)
st.pyplot(fig1)

# Graph 2
st.subheader("Age vs Target")
fig2, ax2 = plt.subplots()
sns.boxplot(x='target', y='age', data=data, ax=ax2)
st.pyplot(fig2)

# Heatmap
st.subheader("Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(data.corr(), cmap='coolwarm', ax=ax3)
st.pyplot(fig3)
