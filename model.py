# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# load data
df = pd.read_csv(r"C:\Users\HW\Desktop\New folder (3)\Ecommerce_Consumer_Behavior_Analysis_Data.csv")

# Strip column spaces just in case
df.columns = df.columns.str.strip()

# FIX: Clean Purchase_Amount (remove $ and spaces)
df["Purchase_Amount"] = (
    df["Purchase_Amount"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.strip()
    .astype(float)
)

# CHECK what values exist in key columns
print("Brand_Loyalty values:", df["Brand_Loyalty"].unique())
print("Customer_Satisfaction values:", df["Customer_Satisfaction"].unique())
print("Purchase_Intent values:", df["Purchase_Intent"].unique())

# CREATE CHURN TARGET
# Using low satisfaction OR low purchase intent as churn signal
df["Customer_Satisfaction"] = pd.to_numeric(df["Customer_Satisfaction"], errors="coerce")

# Churn = low satisfaction (1 or 2) OR no purchase intent
df["Churn"] = (
    (df["Customer_Satisfaction"] <= 2) |
    (df["Purchase_Intent"].str.strip().str.lower() == "no intent")
).astype(int)

print("\nChurn distribution:\n", df["Churn"].value_counts())

# If still all zeros, force a fallback split using median satisfaction
if df["Churn"].sum() == 0:
    print("WARNING: No churn detected — using median satisfaction as fallback")
    median_sat = df["Customer_Satisfaction"].median()
    df["Churn"] = (df["Customer_Satisfaction"] < median_sat).astype(int)
    print("Churn distribution after fallback:\n", df["Churn"].value_counts())


# DROP USELESS FEATURES
df = df.drop(["Customer_ID"], axis=1, errors="ignore")

 
# DEFINE NUMERIC & CATEGORICAL FEATURES
numeric_cols = [
    "Age",
    "Purchase_Amount",
    "Product_Rating",
    "Time_Spent_on_Product_Research(hours)",
    "Return_Rate",
    "Customer_Satisfaction",
]

categorical_cols = [
    "Gender",
    "Income_Level",
    "Marital_Status",
    "Education_Level",
    "Occupation",
    "Location",
    "Purchase_Category",
    "Frequency_of_Purchase",
    "Purchase_Channel",
    "Brand_Loyalty",
    "Social_Media_Influence",
    "Discount_Sensitivity",
    "Engagement_with_Ads",
    "Device_Used_for_Shopping",
    "Payment_Method",
    "Time_of_Purchase",
    "Discount_Used",
    "Customer_Loyalty_Program_Member",
    "Purchase_Intent",
    "Shipping_Preference",
    "Time_to_Decision",
]

# SPLIT DATA
X = df[numeric_cols + categorical_cols]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# PREPROCESSING PIPELINE
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

pipeline = ImbPipeline([
    ("prep", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.01,
        max_depth=2,
        subsample=0.8,
        max_features="sqrt",
        random_state=42
    ))
])

# CROSS-VALIDATION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv)
print("\nCross Validation Accuracy:", round(cv_score.mean(), 3))

# TRAIN MODEL
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# EVALUATION
print("\nTest Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Churn", "Churn"],
            yticklabels=["Non-Churn", "Churn"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# CORRELATION HEATMAP
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()