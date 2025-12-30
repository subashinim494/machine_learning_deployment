# import joblib
# from flask import Flask, request, jsonify
# import pandas as pd
# from scipy.sparse import hstack
# import numpy as np

# # 3. Initialize Flask application
# app = Flask(__name__)

# # Define filenames for the preprocessing tools
# tfidf_vectorizer_filename = 'tfidf_vectorizer.joblib'
# onehot_encoder_filename = 'onehot_encoder.joblib'

# # Save the TF-IDF Vectorizer
# joblib.dump(tfidf_vectorizer, tfidf_vectorizer_filename)
# print(f"TF-IDF Vectorizer successfully saved as '{tfidf_vectorizer_filename}'")

# Save the OneHotEncoder
joblib.dump(ohe, onehot_encoder_filename)
print(f"OneHotEncoder successfully saved as '{onehot_encoder_filename}'")
# # 1. Define filenames for the trained model and preprocessing tools
# model_filename = 'svm_model.joblib'
# tfidf_vectorizer_filename = 'tfidf_vectorizer.joblib'
# onehot_encoder_filename = 'onehot_encoder.joblib'

# # 2. Load the trained SVM model and preprocessing tools
# try:
#     svm_model = joblib.load(model_filename)
#     tfidf_vectorizer = joblib.load(tfidf_vectorizer_filename)
#     onehot_encoder = joblib.load(onehot_encoder_filename)
#     print("Models and preprocessors loaded successfully.")
# except FileNotFoundError:
#     print(f"Error: One or more files not found. Make sure '{model_filename}', '{tfidf_vectorizer_filename}', and '{onehot_encoder_filename}' are in the same directory.")
#     exit()



# # 4. Define preprocessing function
# def preprocess_input(data):
#     # Ensure data is a dictionary containing all expected fields
#     # For simplicity, we assume the input data dictionary has all necessary keys.
#     # In a real-world scenario, you might want to add more robust error handling
#     # or default values for missing keys.

#     # Convert input data to a DataFrame, maintaining original column names
#     input_df = pd.DataFrame([data])

#     # a. Fill missing values in text columns with an empty string
#     text_cols = ['title', 'description', 'requirements', 'company_profile', 'benefits']
#     for col in text_cols:
#         if col in input_df.columns:
#             input_df[col] = input_df[col].fillna('')
#         else:
#             input_df[col] = '' # Add column if not present

#     # b. Fill missing values in categorical columns with 'Unknown'
#     categorical_cols = ['location', 'department', 'employment_type', 'required_experience',
#                         'required_education', 'industry', 'function']
#     for col in categorical_cols:
#         if col in input_df.columns:
#             input_df[col] = input_df[col].fillna('Unknown')
#         else:
#             input_df[col] = 'Unknown' # Add column if not present

#     # c. Create the 'all_text' feature
#     input_df['all_text'] = input_df['title'] + ' ' + input_df['description'] + ' ' + \
#                            input_df['requirements'] + ' ' + input_df['company_profile'] + ' ' + \
#                            input_df['benefits']

#     # e. Apply TF-IDF vectorizer
#     X_text_tfidf = tfidf_vectorizer.transform(input_df['all_text'])

#     # f. Apply OneHotEncoder to categorical features
#     # Ensure the categorical columns are passed in the same order as during training
#     X_categorical_ohe = onehot_encoder.transform(input_df[categorical_cols])

#     # g. Extract numerical/binary features
#     numerical_binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']
#     # Ensure these columns exist and have a default if not provided
#     for col in numerical_binary_cols:
#         if col not in input_df.columns:
#             input_df[col] = 0 # Default to 0 if not provided
#     X_numerical_binary = input_df[numerical_binary_cols].values

#     # h. Combine all features
#     X_combined = hstack([X_text_tfidf, X_categorical_ohe, X_numerical_binary])

#     return X_combined

# # 5. Define the /predict API endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     if not request.json:
#         return jsonify({'error': 'Invalid input, expected JSON'}), 400

#     job_posting_data = request.json

#     try:
#         # Preprocess the input data
#         processed_data = preprocess_input(job_posting_data)

#         # Make prediction using the SVM model
#         prediction = svm_model.predict(processed_data)[0]

#         # Convert prediction to human-readable format
#         human_readable_prediction = "Fraudulent" if prediction == 1 else "Legitimate"

#         return jsonify({'prediction': human_readable_prediction, 'raw_prediction': int(prediction)})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # 6. Run the Flask application
# if __name__ == '__main__':
#     # Running on 0.0.0.0 makes it accessible externally, debug=True for development
#     app.run(debug=True, host='0.0.0.0', port=5000)


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

