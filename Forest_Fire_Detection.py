import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load Data
data_path = "../data/data.csv"
df = pd.read_csv(data_path)

print("\nSample Data:")
print(df.head())

# Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("../output/heatmap.png")
plt.show()

# Scatter Plot
plt.figure(figsize=(6,4))
sns.scatterplot(x="Temperature", y="Humidity", hue="Fire", data=df, palette={0: "green", 1: "red"})
plt.title("Temperature vs Humidity (Red = Fire)")
plt.tight_layout()
plt.savefig("../output/scatter.png")
plt.show()

# Train Model
X = df[['Temperature', 'Humidity', 'WindSpeed', 'Rainfall']]
y = df['Fire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate Model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../output/confusion_matrix.png")
plt.show()

# Test New Sample
sample = [[40, 30, 14, 1]]  # [Temp, Humidity, WindSpeed, Rainfall]
prediction = model.predict(sample)[0]
print(f"\nPrediction for {sample}: {'🔥 Fire Risk' if prediction == 1 else '✅ No Fire'}")
