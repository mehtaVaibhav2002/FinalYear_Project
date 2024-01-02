import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ILDC_single\ILDC_single\ILDC_single.csv\ILDC_single.csv")

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)

# Train a Multinomial Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(train_features, train_labels)

# Make predictions on the test set
predictions = naive_bayes_classifier.predict(test_features)

# Evaluate the performance of the classifier
accuracy = accuracy_score(test_labels, predictions)
confusion_mat = confusion_matrix(test_labels, predictions)
classification_rep = classification_report(test_labels, predictions, output_dict=True)

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
plt.title("Performance Metrics for Naive Bayes Classifier")
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()
