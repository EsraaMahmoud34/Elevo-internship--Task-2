import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# ======================
# Page Config
# ======================
st.set_page_config(page_title="Customer Segmentation", page_icon="🟠", layout="wide")

# ======================
# UI Theme (Dark + Orange)
# ======================
st.markdown(
    """
    <style>
    body {background-color: #0e1117; color: #f5f5f5;}
    .stButton button {
        background-color: #ff6600;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5em 1em;
    }
    .stButton button:hover {
        background-color: #e65c00;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# Load dataset
# ======================
df = pd.read_csv("Mall_Customers.csv")

# Rename for consistency
df.rename(columns={"Annual Income (k$)": "Income", "Spending Score (1-100)": "Score"}, inplace=True)

# ======================
# Training Model (3 features)
# ======================
X = df[["Age", "Income", "Score"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Save model and scaler
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ======================
# Title
# ======================
st.title("🟠 Customer Segmentation with KMeans")

# ======================
# Sidebar User Input
# ======================
st.sidebar.header("Enter Customer Details")
user_id = st.sidebar.text_input("Customer ID")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
income = st.sidebar.number_input("Annual Income (k$)", min_value=10, max_value=150, value=50)
score = st.sidebar.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.sidebar.button("Predict Cluster"):
    # Load model
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Prepare features
    features = np.array([[age, income, score]])
    features_scaled = scaler.transform(features)

    cluster = model.predict(features_scaled)[0]

    st.subheader(f"🟠 Customer belongs to **Cluster {cluster}**")

    # Append new customer
    new_customer = pd.DataFrame({
        "CustomerID": [user_id],
        "Gender": [gender],
        "Age": [age],
        "Income": [income],
        "Score": [score],
        "Cluster": [cluster]
    })
    df_with_new = pd.concat([df, new_customer], ignore_index=True)

    # ======================
    # Visualizations
    # ======================

    st.subheader("📊 Cluster Visualization")

    # Scatter Plot (Income vs Score)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="Income", y="Score", hue="Cluster", palette="Set2", s=60, ax=ax)
    plt.scatter(income, score, c="red", s=150, marker="*", label="New Customer")
    plt.legend()
    st.pyplot(fig)

    # 3D Visualization
    st.subheader("🌐 3D Cluster Visualization (Age, Income, Score)")
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(df["Age"], df["Income"], df["Score"], c=df["Cluster"], cmap="Set2", s=50)
    ax.scatter(age, income, score, c="red", s=200, marker="*", label="New Customer")
    ax.set_xlabel("Age")
    ax.set_ylabel("Income")
    ax.set_zlabel("Score")
    ax.legend()
    st.pyplot(fig)

    # PCA 2D Projection
    st.subheader("🔎 PCA Projection (2D View of Clusters)")
    pca = PCA(2)
    df_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.scatter(df_pca[:,0], df_pca[:,1], c=df["Cluster"], cmap="Set2", s=50)
    plt.title("Clusters (2D PCA Projection)")
    plt.scatter(pca.transform(features_scaled)[:,0], pca.transform(features_scaled)[:,1],
                c="red", s=200, marker="*", label="New Customer")
    plt.legend()
    st.pyplot(fig)

    # Cluster Count
    st.subheader("📊 Cluster Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Cluster", data=df_with_new, palette="Set2", ax=ax)
    plt.title("Number of Customers per Cluster")
    st.pyplot(fig)

    # Average Characteristics
    st.subheader("📊 Average Characteristics by Cluster")
    cluster_summary = df_with_new.groupby("Cluster")[["Age", "Income", "Score"]].mean()

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.barplot(x=cluster_summary.index, y=cluster_summary["Age"], palette="Set2", ax=ax[0])
    ax[0].set_title("Average Age")
    sns.barplot(x=cluster_summary.index, y=cluster_summary["Income"], palette="Set2", ax=ax[1])
    ax[1].set_title("Average Income")
    sns.barplot(x=cluster_summary.index, y=cluster_summary["Score"], palette="Set2", ax=ax[2])
    ax[2].set_title("Average Spending Score")
    st.pyplot(fig)

