import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("ILDC_single\ILDC_single\ILDC_single.csv\ILDC_single.csv")

# Assuming "text" is the feature column and "label" is the target column
X = df["text"].values
y = df["label"].values

# Convert text data to numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X).toarray()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = logistic_regression_model.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, predictions)
confusion_mat = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions, output_dict=True)

# Print the results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_mat)
print("\nClassification Report:")
print(classification_rep)

labels = ["Precision", "Recall", "F1-score", "Accuracy"]
scores = [
    classification_rep["0"]["precision"],
    classification_rep["0"]["recall"],
    classification_rep["0"]["f1-score"],
    accuracy,
]

plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=scores, palette="viridis")
plt.title("Performance Metrics for Logistic Regression Classifier")
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()
