from flask import Flask, request, render_template
import numpy as np
import joblib
app = Flask(__name__)

# Load model
model = joblib.load('wine_quality_project.pkl')

scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect features from form using exact input names
    features = [
        float(request.form['fixed_acidity']),
        float(request.form['volatile_acidity']),
        float(request.form['citric_acid']),
        float(request.form['residual_sugar']),
        float(request.form['chlorides']),
        float(request.form['free_sulfur_dioxide']),
        float(request.form['total_sulfur_dioxide']),
        float(request.form['density']),
        float(request.form['pH']),
        float(request.form['sulphates']),
        float(request.form['alcohol'])
    ]
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    if(prediction == 0):
        pred = 'NOT GOOD'
    else:
        pred = 'GOOD'
    return render_template('index.html', prediction_text=f'Predicted Wine Quality: {pred}')

if __name__ == '__main__':
    app.run(debug=True)
