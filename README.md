# customer-churn-prediction-ml
A predictive analytics project to identify high-risk customers using Python and Gradient Boosting.

---

## 📌 Project Overview
In the E-commerce industry, identifying dissatisfied customers before they leave is critical. This project builds an end-to-end pipeline to:
*   **Target:** Predict "Churn" based on low satisfaction scores and zero purchase intent.
*   **Balance:** Use **SMOTE** to handle imbalanced datasets.
*   **Model:** Implement a **Gradient Boosting Classifier** optimized for Recall.

---

## 📊 Dataset Features
The model analyzes several facets of consumer data:
*   **Demographics:** Age, Gender, Income, Occupation, Location.
*   **Behavioral:** Research time, Product Rating, Return Rate, Purchase Frequency.
*   **Transaction:** Purchase Amount, Payment Method, Discount Usage.
*   **Psychographic:** Brand Loyalty, Social Media Influence, Satisfaction.

---

## 🛠️ Tech Stack
*   **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
*   **Data Science:** Pandas, NumPy
*   **Visualization:** Seaborn, Matplotlib

---

## ⚙️ Pipeline Architecture
The project utilizes a `ColumnTransformer` within an `ImbPipeline` to ensure a clean workflow:
1.  **Preprocessing:** Standard Scaling for numbers; One-Hot Encoding for categories.
2.  **Resampling:** SMOTE applied only to training data to prevent leakage.
3.  **Classification:** Tuned Gradient Boosting ($learning\_rate=0.01$, $max\_depth=2$).

---

## 📈 Results
The model is evaluated using **Stratified K-Fold Cross-Validation** and a **Confusion Matrix** to ensure we are accurately catching churners (Recall) rather than just achieving high accuracy on non-churners.

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   
Install dependencies:

Bash
pip install pandas scikit-learn imbalanced-learn seaborn

3. Run the analysis:
   ```bash
   python main.py
