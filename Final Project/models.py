# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data_path = r'C:\Users\Linata04\Desktop\Final Project\Dataset\heart.csv'  # Use a raw string
heart_data = pd.read_csv(data_path)

# Define categorical and numerical features
categorical_features = heart_data.select_dtypes(include=['object']).columns.tolist()
numeric_features = [col for col in heart_data.columns if col not in categorical_features + ['HeartDisease']]

# Data preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(random_state=0),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
    "SVM": SVC(random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0)
}

# Split data into features and target variable
X = heart_data.drop('HeartDisease', axis=1)
y = heart_data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize dictionary to hold trained pipelines for saving
trained_pipelines = {}

# Train each model and evaluate
accuracies = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy
    trained_pipelines[name] = pipeline  # Store the trained pipeline

# Sort models based on accuracy and save the top 3
top_models = sorted(accuracies, key=accuracies.get, reverse=True)[:3]
for model_name in top_models:
    joblib.dump(trained_pipelines[model_name], f'{model_name.replace(" ", "_")}_model.pkl')

# Print model accuracies and top models saved
print("Model Accuracies:")
for model_name, acc in accuracies.items():
    print(f"{model_name}: {acc:.3f}")
print("\nTop 3 Models Saved:")
for model_name in top_models:
    print(f"- {model_name}")
    