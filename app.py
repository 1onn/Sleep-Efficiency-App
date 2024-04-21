from flask import Flask, render_template, request, redirect, url_for
from wtforms import Form, IntegerField, FloatField, RadioField
import joblib
import numpy as np

# Define a form for user input
class SleepForm(Form):
    # Your form fields here...
    age = IntegerField('Age:')
    gender = RadioField('Gender:', choices=[('0', 'Female'), ('1', 'Male')])
    rem_sleep = FloatField('REM sleep percentage:')
    deep_sleep = FloatField('Deep sleep percentage:')
    light_sleep = FloatField('Light sleep percentage:')
    awakenings = IntegerField('Number of awakenings:')
    caffeine = FloatField('Caffeine consumption (mg):')
    alcohol = FloatField('Alcohol consumption (oz):')
    exercise = RadioField('Exercise frequency:', choices=[('0', 'Sedentary'), ('1', 'Light'), ('2', 'Moderate'), ('3', 'Heavy')])
    smoking = RadioField('Smoking status:', choices=[('0', 'No'), ('1', 'Yes')])

# Load the trained model and scaler using joblib
model = joblib.load('sleep_efficiency_model.pkl')
scaler = joblib.load('sleep_efficiency_scaler.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Define the main route for input form
@app.route('/', methods=['GET', 'POST'])
def predict():
    form = SleepForm(request.form)
    prediction = None
    error = None

    if request.method == 'POST':
        if form.validate():
            try:
                # Retrieve values from the form
                # (Modify this section based on your form structure)
                age = form.age.data
                gender = int(form.gender.data)
                rem_sleep = form.rem_sleep.data
                deep_sleep = form.deep_sleep.data
                light_sleep = form.light_sleep.data
                awakenings = form.awakenings.data
                caffeine = form.caffeine.data
                alcohol = form.alcohol.data
                exercise = int(form.exercise.data)
                smoking = int(form.smoking.data)

                # Validate sleep percentages
                if round(rem_sleep + deep_sleep + light_sleep) != 100:
                    error = 'Invalid sleep percentages!'
                else:
                    # Prepare the data for prediction
                    data = np.array([[age, gender, rem_sleep, deep_sleep, light_sleep, awakenings, caffeine, alcohol, exercise, smoking]])

                    # Predict sleep efficiency
                    predicted_sleep_efficiency = model.predict(data)[0]

                    # Redirect to results page with the prediction
                    return redirect(url_for('results', prediction=predicted_sleep_efficiency))

            except Exception as e:
                error = f'Prediction error: {str(e)}'

    return render_template('index.html', form=form, prediction=prediction, error=error)

# Define route for displaying results
@app.route('/results/<prediction>')
def results(prediction):
    return render_template('results.html', prediction=prediction)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
