🛍️ Mall Customer Segmentation
📌 Project Overview

This project applies Unsupervised Learning (Clustering) to the Mall Customer Dataset (Kaggle)
.
The goal is to segment mall customers into distinct groups based on their Annual Income and Spending Score.

By identifying different customer types, businesses can better design marketing strategies, personalized offers, and customer retention programs.

👉 This project is also deployed as an interactive Streamlit web app where users can explore clustering results in real-time.

🎯 Objectives

✔️ Perform data preprocessing and feature scaling
✔️ Explore data with visualizations
✔️ Apply K-Means Clustering
✔️ Determine the optimal number of clusters using the Elbow Method & Silhouette Score
✔️ Visualize customer groups in 2D scatter plots
✔️ Provide business insights for each segment
✔️ (Bonus) Try other clustering methods (e.g., DBSCAN) and compare results
✔️ Deploy with Streamlit for easy interaction

🗂️ Dataset

Columns:

CustomerID (ignored for clustering)

Gender

Age

Annual Income (k$)

Spending Score (1–100)

For clustering, the main features are:
Annual Income (k$) & Spending Score (1–100)

🛠️ Tools & Libraries

Python 3.9+

Pandas
 – data handling

NumPy
 – numerical operations

Matplotlib
 & Seaborn
 – visualization

Scikit-Learn
 – clustering & evaluation

Streamlit
 – deployment & interactive app

⚡ How to Run
🔹 Option 1: Run Locally (Notebook/Script)

Clone the repository:

git clone https://github.com/your-username/mall-customer-segmentation.git
cd mall-customer-segmentation


Install dependencies:

Run the notebook or script:

notebook Mall_Customer_Clustering.ipynb
or

python mall_clustering.py

🔹 Option 2: Run with Streamlit (Interactive App)

Clone the repository (if not already done):

git clone https://github.com/your-username/mall-customer-segmentation.git
cd mall-customer-segmentation


Install dependencies (make sure Streamlit is included):
Run the Streamlit app:
streamlit run app.py

Open the app in your browser at:
👉 http://localhost:8501

📊 Results
🔹 Optimal Clusters

Determined using the Elbow Method & Silhouette Score → Best k ≈ 5

🔹 Identified Segments

High Income, High Spending → “Target Customers” 💎

Low Income, Low Spending → “Budget Customers” 💸

High Income, Low Spending → “Potential Customers” 🧐

Moderate Income, High Spending → “Impulsive Shoppers” 🎉

Middle Income, Average Spending → “Regular Customers” 👨‍👩‍👧

🔹 Visualization Example

(insert a screenshot of your Streamlit app or cluster plot here)

🔮 Bonus Work

Applied DBSCAN for density-based clustering.

Compared results with K-Means.

Calculated average spending per cluster to derive business insights.

📈 Business Insights

Mall should focus marketing on Cluster 1 (High Income, High Spending) and Cluster 4 (Impulsive Shoppers).

Cluster 3 (High Income, Low Spending) → Can be targeted with personalized offers to increase spending.

Cluster 2 (Budget Customers) → Less priority for luxury goods but can be offered discount products.

📌 Covered Topics

Clustering

K-Means

Unsupervised Learning

Data Preprocessing & Scaling

Model Evaluation (Elbow Method, Silhouette Score)

Streamlit Deployment

✨ Author

👩‍💻 Esraa Mahmoud
🎓 AI Student @ Ain Shams University
🏆 USAID Alumni | DEPI Microsoft ML Engineer Trainee
💼 Passionate about Machine Learning & Data Science
