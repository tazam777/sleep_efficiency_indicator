from flask import Flask, request, jsonify
import sys
import joblib
import signal
from werkzeug.serving import make_server

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_svm_model.pkl')


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







def preprocess_input(user_input):
    # Convert Gender to numerical format
    gender_mapping = {'Female': 0, 'Male': 1}
    user_input['Gender'] = gender_mapping[user_input['Gender']]
    
    # Smoking status encoding (Yes: 1, No: 0)
    smoking_mapping = {'Yes': 1, 'No': 0}
    user_input['Smoking status'] = smoking_mapping[user_input['Smoking status']]

    # Map age group to one-hot encoding
    age_groups = ['Aged Adult', 'Child', 'Middle-Aged Adult', 'Senior', 'Teenager', 'Young Adult']
    age_group_mapping = {age: [1 if user_input['Age_Group'] == age else 0 for age in age_groups] for age in age_groups}
    
    # Flatten the age group mapping
    for age_group, value in age_group_mapping.items():
        user_input[f'Age_Group_{age_group}'] = value[0]

    # Return preprocessed input features as a list in the same order as expected by the model
    features_order = ['Gender', 'Sleep duration', 'REM sleep percentage', 'Deep sleep percentage',
                      'Light sleep percentage', 'Awakenings', 'Caffeine consumption', 'Alcohol consumption',
                      'Smoking status', 'Exercise frequency', 'Caffeine cup', 'Age_Group_Aged Adult', 
                      'Age_Group_Child', 'Age_Group_Middle-Aged Adult', 'Age_Group_Senior', 
                      'Age_Group_Teenager', 'Age_Group_Young Adult']
    return [user_input[feature] for feature in features_order]

# Example user_input to preprocess
user_input = {
    'Gender': 'Male',
    'Age_Group': 'Young Adult',
    'Smoking status': 'No',
    'Sleep duration': 8,
    'REM sleep percentage': 20,
    'Deep sleep percentage': 25,
    'Light sleep percentage': 55,
    'Awakenings': 2,
    'Caffeine consumption': 3,
    'Alcohol consumption': 1,
    'Exercise frequency': 4,
    'Caffeine cup': 2
}

# Call the preprocess_input function with a sample input
preprocessed_input = preprocess_input(user_input)
print(preprocessed_input)




def predict_good_sleep(input_features):
    # Make prediction using the trained model
    prediction = model.predict([input_features])
    result = "good sleep" if prediction == 1 else "bad sleep"
    return result

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  # Get data sent to the API
    preprocessed_input = preprocess_input(data)
    if isinstance(preprocessed_input, tuple):  # Check if preprocessing failed
        return preprocessed_input  # Return the error if there's any
    result = predict_good_sleep(preprocessed_input)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000  # Default port 5000
    server = Server(app, port=port)  # Initialize the server with the specified port
    server.run()