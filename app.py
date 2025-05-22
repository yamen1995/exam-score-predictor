from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('exam_score_predictor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'study_hours_per_day': float(request.form['study_hours']),
            'mental_health_rating': float(request.form['mental_health']),
            'exercise_frequency': float(request.form['exercise']),
            'social_media_hours': float(request.form['social_media'])
        }

        X = pd.DataFrame([data])
        prediction = model.predict(X)[0]

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'prediction': f'{prediction:.1f}'})
        else:
            return render_template('index.html', 
                                   prediction_text=f'Predicted Exam Score: {prediction:.1f}/100',
                                   **data)
    except Exception as e:
        return jsonify({'error': str(e)})