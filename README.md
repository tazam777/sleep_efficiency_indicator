### Running the Program:

1. **Setup Environment:**
   - Make sure first you run the python nodebook entirely with updated dataset so that model is created in your local repo as that is important for our backend to run
   - Make sure you have Python installed on your system.
   - Install Flask and other required dependencies. You can do this using pip:
     ```bash
     pip install Flask joblib pandas
     ```

3. **Download the Model:**
   - Make sure you have the trained model file (`best_svm_model.pkl`) available in the same directory as your Flask application.

4. **Run the Flask Application:**
   - Save the provided code in a Python file, for example, `app.py`.
   - Open a terminal or command prompt.
   - Navigate to the directory containing `app.py`.
   - Run the Flask application using the following command:
     ```bash
     python app.py [port_number]
     ```
     Replace `[port_number]` with the desired port number. If not specified, the default port is 5000.

### Sending Requests using cURL:

You can interact with the Flask API using cURL commands in the terminal or command prompt. Here's an example of how to send a POST request with JSON data:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"Gender": "Male", "Age_Group": "Adult", "Sleep duration": 7.5, "REM sleep percentage": 20, "Deep sleep percentage": 60, "Light sleep percentage": 20, "Awakenings": 2, "Caffeine consumption": 50, "Alcohol consumption": 0, "Smoking status": "No", "Exercise frequency": 3, "Caffeine cup": 1}' http://localhost:<portnumber>/predict


** To load Front End :

In Browser enter the url http://localhost:<portnumber>/index.html
