import logging
import pandas as pd  
from flask import Flask, request, jsonify
from flask import Flask, render_template, send_from_directory
import sys
import joblib
import signal
from werkzeug.serving import make_server
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# Load the trained model
model = joblib.load('best_svm_model.pkl')

if hasattr(model, 'feature_names_in_'):
    expected_features = model.feature_names_in_
    print("Features expected by the model:", expected_features)
else:
    expected_features = ['Your', 'List', 'Of', 'Expected', 'Features']
    print("Define expected features manually as the model does not provide them.")

class Server:
    def __init__(self, app, hostname='localhost', port=5000):
        self._server = make_server(hostname, port, app)
        self._ctx = app.app_context()
        self._ctx.push()

    def run(self):
        print(f"Starting server on port {self._server.port}")
        self._server.serve_forever()

    def stop(self):
        print("Stopping server...")
        self._server.shutdown()

server = None  # This will hold the server instance

def signal_handler(signal, frame):
    print('Signal received, stopping server...')
    if server:
        server.stop()
    sys.exit(0)

# Register the signal handler for SIGINT (usually triggered by Ctrl-C)
signal.signal(signal.SIGINT, signal_handler)
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

def preprocess_input(user_input):
    # Define default mappings
    gender_mapping = {'Female': 0, 'Male': 1}
    smoking_mapping = {'Yes': 1, 'No': 0}
    age_groups = ['Aged Adult', 'Child', 'Middle-Aged Adult', 'Senior', 'Teenager', 'Young Adult']

    # Convert Gender to numerical format, default if gender is invalid or not present
    user_input['Gender'] = gender_mapping.get(user_input.get('Gender', ''), -1)
    
    # Smoking status encoding with default if status is invalid or not present
    user_input['Smoking status'] = smoking_mapping.get(user_input.get('Smoking status', ''), -1)

    # Map age group to one-hot encoding
    age_group_mapping = {age: [1 if user_input.get('Age_Group', '') == age else 0 for age in age_groups] for age in age_groups}
    
    # Flatten the age group mapping
    for age_group, values in age_group_mapping.items():
        user_input[f'Age_Group_{age_group}'] = values[0]

    # Ensure all necessary features are in the input, fill missing ones with zeros
    features_order = ['Gender', 'Sleep duration', 'REM sleep percentage', 'Deep sleep percentage',
                      'Light sleep percentage', 'Awakenings', 'Caffeine consumption', 'Alcohol consumption',
                      'Smoking status', 'Exercise frequency', 'Caffeine cup'] + [f'Age_Group_{ag}' for ag in age_groups]
    
    # Construct final dictionary for DataFrame creation
    final_data = {feature: user_input.get(feature, 0) for feature in features_order}

    return pd.DataFrame([final_data])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    logging.debug(f"Received JSON data: {data}")

    # Call the preprocessing function
    input_df = preprocess_input(data)
    logging.debug(f"DataFrame sent to model for prediction: {input_df}")

    try:
        prediction = model.predict(input_df)
        result = "Good Sleep" if prediction[0] == 1 else "Bad Sleep"
        return jsonify({'prediction': result})
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': 'Error making prediction', 'details': str(e)}), 500





# Route to serve the HTML page
@app.route('/index.html')
@app.route('/')  # Optional: serve the same page on the root URL as well
def index():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000  # Default port 5000
    server = Server(app, port=port)  # Initialize the server with the specified port
    server.run()








  