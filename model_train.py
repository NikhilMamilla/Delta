import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

file_path = 'Final_delta.csv'
df = pd.read_csv(file_path)

df.head()

df.tail()

print("Shape of the DataFrame:")
print(df.shape)

print("Columns of the DataFrame:")
print(df.columns)

print("Number of duplicated rows in the DataFrame:")
print(df.duplicated().sum())

df = df.drop_duplicates()

duplicates = df[df.duplicated()]
print("Duplicate rows:")
print(duplicates)

print("\nMissing values in the DataFrame:")
print(df.isnull().sum())

print("Information about the DataFrame:")
print(df.info())

print("\nSummary statistics of the DataFrame:")
print(df.describe())

feature_columns = ['age', 'gender', 'physical_health', 'mental_health', 'employment', 'sleep', 'stress', 'smoking', 'drinking']
target_column = 'mental_health'

X = data[feature_columns]
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=RandomForestClassifier(random_state=42)

model.fit(X_train,y_train)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


clf = RandomForestClassifier(n_estimators=100, random_state=42)


clf.fit(X_train_scaled, y_train)


y_pred = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


X.hist(bins=30, figsize=(15, 10))
plt.suptitle('Feature Distributions')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


corr = data[feature_columns + [target_column]].corr()

plt.figure(figsize=(8,6))

sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Feature Correlation Matrix')
plt.show()

feature_importances = clf.feature_importances_
features = feature_columns
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(6, 6))
plt.title('Feature Importances')
plt.bar(range(len(features)), feature_importances[indices], color='b', align='center')
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_test, y_pred)





cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

print("Cross-validation scores: ", cv_scores)
print("Mean cross-validation score: {:.2f}".format(cv_scores.mean()))


from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]


from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], clf.predict_proba(X_test_scaled)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
