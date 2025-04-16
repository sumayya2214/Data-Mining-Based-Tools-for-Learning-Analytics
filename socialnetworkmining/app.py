from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

# Load dataset
DATA_PATH = "synthetic_learning_analytics.csv"
data = pd.read_csv(DATA_PATH)

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    # Separate numerical and categorical columns for dynamic dropdowns
    numerical_columns = list(data.select_dtypes(include=np.number).columns)
    categorical_columns = list(data.select_dtypes(include=['object', 'category', 'bool']).columns)
    return render_template(
        'index.html',
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns
    )

@app.route('/run_models', methods=['POST'])
def run_models():
    # Parse the JSON data
    request_data = request.json
    features = request_data.get('features', [])
    target_class = request_data.get('target_class', None)
    target_reg = request_data.get('target_reg', None)

    # Validate inputs
    if not features or not target_class or not target_reg:
        return jsonify({"error": "Please select features, classification target, and regression target."}), 400

    try:
        # Features and targets for modeling
        X = data[features]
        y_classification = data[target_class]
        y_regression = data[target_reg]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Classification
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_scaled, y_classification, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train_clf, y_train_clf)
        y_pred_clf = clf.predict(X_test_clf)

        classification_metrics = {
            "Accuracy": accuracy_score(y_test_clf, y_pred_clf),
            "Precision": precision_score(y_test_clf, y_pred_clf, average='weighted'),
            "Recall": recall_score(y_test_clf, y_pred_clf, average='weighted'),
            "F1 Score": f1_score(y_test_clf, y_pred_clf, average='weighted')
        }

        # Regression
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y_regression, test_size=0.3, random_state=42)
        reg = RandomForestRegressor(random_state=42)
        reg.fit(X_train_reg, y_train_reg)
        y_pred_reg = reg.predict(X_test_reg)

        regression_metrics = {
            "Mean Squared Error": mean_squared_error(y_test_reg, y_pred_reg),
            "R2 Score": r2_score(y_test_reg, y_pred_reg)
        }

        # Return the results
        return jsonify({
            "classification_metrics": classification_metrics,
            "regression_metrics": regression_metrics
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
