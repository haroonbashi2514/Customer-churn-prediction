import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    df[column] = le.fit_transform(df[column])

# Train-test split
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print
print("Model Accuracy:", accuracy)
print(report)

# Save results
with open("output_report.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy:.2f}\n\n")
    f.write(report)

predictions_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
predictions_df.to_csv("predictions.csv", index=False)

# Save model
joblib.dump(model, 'churn_model.pkl')

# Visualization
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.savefig("churn_distribution.png")
plt.show()

plt.figure(figsize=(6, 6))
df['InternetService'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Internet Service Distribution')
plt.savefig("internet_service_pie.png")
plt.show()
