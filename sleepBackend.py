from flask import Flask, request, jsonify
import pickle
import pandas as pd
import sys  # Import sys for command-line arguments

app = Flask(__name__)

# Load the SVM model from disk
with open('best_svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def merge_age_groups(age):
    if age <= 10:
        return 'Child'
    elif 10 < age <= 20:
        return 'Teenager'
    elif 20 < age <= 30:
        return 'Young Adult'
    elif 30 < age <= 40:
        return 'Adult'
    elif 40 < age <= 50:
        return 'Middle-Aged Adult'
    elif 50 < age <= 60:
        return 'Aged Adult'
    else:
        return 'Senior'

@app.route('/predict', methods=['POST'])
def predict():
    
        data = request.get_json(force=True)
        features = pd.DataFrame(data['features'])

        # Applying age group merging logic and drop the 'Age' feature
        features['Age_Group'] = features['Age'].apply(merge_age_groups)
        features.drop('Age', axis=1, inplace=True)  # Drop the 'Age' column as it's not used directly

        # One-hot encode the 'Age_Group'
        features = pd.get_dummies(features, columns=['Age_Group'], drop_first=True)

        # Ensure all columns necessary for the model are present
        expected_columns = model.feature_names_in_  # This assumes the model has the 'feature_names_in_' attribute
        missing_cols = set(expected_columns) - set(features.columns)
        for col in missing_cols:
            features[col] = 0
        
        # Reorder columns to match the model's training order
        features = features[expected_columns]

        prediction = model.predict(features)
        return jsonify(prediction.tolist())

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000  # Default port 5000
    app.run(debug=True, port=port)
