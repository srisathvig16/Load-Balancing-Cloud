import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Sample data generation
def generate_sample_data():
    # Creating a sample dataset
    data = {
        'Source.Port': np.random.randint(1000, 5000, 1000),
        'Destination.IP': np.random.randint(1, 255, 1000),
        'Destination.Port': np.random.randint(1000, 5000, 1000),
        'Protocol': np.random.choice(['TCP', 'UDP'], 1000),
        'Flow.Duration': np.random.randint(1, 1000, 1000),
        'Total.Fwd.Packets': np.random.randint(1, 100, 1000),
        'Total.Backward.Packets': np.random.randint(1, 100, 1000),
        'Label': np.random.choice([0, 1], 1000)  # 0 for no load, 1 for load
    }
    df = pd.DataFrame(data)
    return df

# Load data
df = generate_sample_data()

# Preprocess data
df = pd.get_dummies(df, columns=['Protocol'])

# Feature selection
features = df.drop('Label', axis=1)
labels = df['Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model
joblib.dump(model, 'dforest_model.pkl')