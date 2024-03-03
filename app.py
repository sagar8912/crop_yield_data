from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained linear regression model
with open('crop_yield_data.pkl', 'rb') as f:
    model = pickle.load(f)

# Define route for index page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve feature values from form
        features = [float(request.form['rainfall']),
                    float(request.form['fertilizer']),
                    float(request.form['temperature']),
                    float(request.form['nitrogen']),
                    float(request.form['phosphorus']),
                    float(request.form['potassium'])]

        # Reshape feature array and make prediction
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        # Render result template with predicted value
        return render_template('result.html', prediction_text=output)
    except Exception as e:
        return "An error occurred: " + str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=False)






