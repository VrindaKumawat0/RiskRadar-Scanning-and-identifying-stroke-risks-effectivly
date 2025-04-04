from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd

# Load pre-trained model and scaler
model = joblib.load("stroke_risk_model.pkl")
scaler = joblib.load("scaler.pkl")  
X_train_columns = joblib.load("X_train_columns.pkl")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get form data
            data = request.form
            
            # Process age with validation
            age = int(data.get('age', 30))
            age = max(18, min(85, age))  # Enforce 18-85 range
            
            # Process symptoms (unchecked boxes won't submit, default to 0)
            symptoms = [
                int(data.get('chest_pain', 0)),
                int(data.get('high_blood_pressure', 0)),
                int(data.get('irregular_heartbeat', 0)),
                int(data.get('shortness_of_breath', 0)),
                int(data.get('fatigue_weakness', 0)),
                int(data.get('dizziness', 0)),
                int(data.get('swelling_edema', 0)),
                int(data.get('neck_jaw_pain', 0)),
                int(data.get('excessive_sweating', 0)),
                int(data.get('persistent_cough', 0)),
                int(data.get('nausea_vomiting', 0)),
                int(data.get('chest_discomfort', 0)),
                int(data.get('cold_hands_feet', 0)),
                int(data.get('snoring_sleep_apnea', 0)),
                int(data.get('anxiety_doom', 0))
            ]
            
            # Process gender
            gender = data.get('gender', 'female').lower()
            gender_male = 1 if gender == 'male' else 0
            
            # Create DataFrame
            features = [age] + symptoms + [gender_male]
            input_df = pd.DataFrame([features], columns=X_train_columns)
            
            # Scale and predict
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            # Prepare results
            if prediction == 1:
                result = {
                    'risk_level': 'POTENTIAL RISK',
                    'recommendations': [
                        "🚑 Consult a doctor immediately for evaluation",
                        "💊 Monitor blood pressure and cholesterol weekly",
                        "🏥 Schedule a full cardiovascular check-up",
                        "🏃 Start with 30 mins daily walking if inactive",
                        "🧘 Practice daily stress management (yoga/meditation)",
                        "🚭 Quit smoking completely - seek help if needed",
                        "🍷 Eliminate alcohol or limit to 1 drink/week"
                    ]
                }
            else:
                result = {
                    'risk_level': 'LOW RISK',
                    'recommendations': [
                        "🩺 Continue annual health check-ups",
                        "⚖️ Maintain current healthy habits",
                        "🥦 Consider adding more leafy greens to your diet",
                        "🚶‍♂️ Try adding 10% more daily steps",
                        "💧 Stay well hydrated (2-3L water daily)",
                        "🌱 Explore new healthy recipes monthly",
                        "🧠 Learn about stroke warning signs (FAST method)",
                        "🛌 Ensure consistent sleep schedule"
                    ]
                }
            
            return redirect(url_for('result', **result))
            
        except Exception as e:
            return redirect(url_for('result', 
                                risk_level='ERROR',
                                recommendations=[f"An error occurred: {str(e)}"]))
    
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template(
        'result.html',
        risk_level=request.args.get('risk_level', 'No result'),
        recommendations=request.args.getlist('recommendations')
    )

if __name__ == '__main__':
    app.run(debug=True)