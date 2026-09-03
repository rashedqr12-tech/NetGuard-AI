import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load network data
data = pd.read_csv("network_data.csv")

# Select network performance features
features = [
    "latency_ms",
    "packet_loss_pct",
    "bandwidth_mbps",
    "cpu_usage_pct",
    "memory_usage_pct",
    "connections"
]

X = data[features]
y = data["network_status"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create the machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("===================================")
print("       NetGuard AI Results")
print("===================================")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save the trained model
joblib.dump(model, "network_failure_model.pkl")

print("\nModel saved successfully!")
