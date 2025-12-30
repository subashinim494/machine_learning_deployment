import joblib
from flask import Flask, request, jsonify
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# -------------------------------
# Initialize Flask application
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Filenames (must already exist)
# -------------------------------
model_filename = 'svm_model.joblib'
tfidf_vectorizer_filename = 'tfidf_vectorizer.joblib'
onehot_encoder_filename = 'onehot_encoder.joblib'

# -------------------------------
# Load model & preprocessors
# -------------------------------
try:
    svm_model = joblib.load(model_filename)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_filename)
    onehot_encoder = joblib.load(onehot_encoder_filename)
    print("Models and preprocessors loaded successfully.")
except Exception as e:
    print("Error loading model files:", e)
    raise e

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_input(data):
    input_df = pd.DataFrame([data])

    # Text columns
    text_cols = ['title', 'description', 'requirements', 'company_profile', 'benefits']
    for col in text_cols:
        if col not in input_df:
            input_df[col] = ''
        input_df[col] = input_df[col].fillna('')

    # Categorical columns
    categorical_cols = [
        'location', 'department', 'employment_type',
        'required_experience', 'required_education',
        'industry', 'function'
    ]
    for col in categorical_cols:
        if col not in input_df:
            input_df[col] = 'Unknown'
        input_df[col] = input_df[col].fillna('Unknown')

    # Combine text
    input_df['all_text'] = (
        input_df['title'] + ' ' +
        input_df['description'] + ' ' +
        input_df['requirements'] + ' ' +
        input_df['company_profile'] + ' ' +
        input_df['benefits']
    )

    # TF-IDF
    X_text = tfidf_vectorizer.transform(input_df['all_text'])

    # One-hot encoding
    X_cat = onehot_encoder.transform(input_df[categorical_cols])

    # Numerical / binary features
    num_cols = ['telecommuting', 'has_company_logo', 'has_questions']
    for col in num_cols:
        if col not in input_df:
            input_df[col] = 0

    X_num = csr_matrix(input_df[num_cols].values)

    # Combine all features
    X_final = hstack([X_text, X_cat, X_num])

    return X_final

# -------------------------------
# Prediction API
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Expected JSON input'}), 400

    try:
        data = request.get_json()
        X = preprocess_input(data)
        prediction = svm_model.predict(X)[0]

        result = "Fraudulent" if prediction == 1 else "Legitimate"

        return jsonify({
            "prediction": result,
            "raw_prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------------
# Run app
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

