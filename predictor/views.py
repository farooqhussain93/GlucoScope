from django.shortcuts import render
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from django.conf import settings
from datetime import datetime

# Load model once at module level
model = joblib.load("diabetes_model.pkl")

def home(request):
    return render(request, 'predictor/home.html')

def predict(request):
    if request.method == 'POST':
        def safe_cast(val, to_type, default=0):
            try:
                return to_type(val)
            except (ValueError, TypeError):
                return default

        # 1. Get user input
        Pregnancies = safe_cast(request.POST['Pregnancies'], int)
        Glucose = safe_cast(request.POST['Glucose'], int)
        BloodPressure = safe_cast(request.POST['BloodPressure'], int)
        SkinThickness = safe_cast(request.POST['SkinThickness'], int)
        Insulin = safe_cast(request.POST['Insulin'], int)
        BMI = safe_cast(request.POST['BMI'], float)
        DPF = safe_cast(request.POST['DiabetesPedigreeFunction'], float)
        Age = safe_cast(request.POST['Age'], int)

        features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                              Insulin, BMI, DPF, Age]])

        # 2. Prediction & confidence
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        # 3. Define result and suggestions
        if prediction == 1:
            result = "You are likely diabetic."
            diet = "Low sugar, high-fiber, low-carb diet. Avoid sweets. Eat oats, vegetables, lean proteins."
            medicine = "Metformin or insulin therapy (consult a doctor)."
            timings = "Take medicine after meals, morning and evening."
        else:
            result = "You are not diabetic."
            diet = "Balanced diet with fruits, vegetables, and controlled carbs."
            medicine = "No medicine needed, but stay active and healthy."
            timings = "N/A"

        # 4. Plot saving path
        static_dir = os.path.join(settings.BASE_DIR, 'predictor', 'static')
        os.makedirs(static_dir, exist_ok=True)

        # Unique filename logic (optional): just reuse same names
        input_graph_filename = 'input_graph.png'
        result_graph_filename = 'result_graph.png'

        input_graph_path = os.path.join(static_dir, input_graph_filename)
        result_graph_path = os.path.join(static_dir, result_graph_filename)

        # 5. Plot 1: Input Overview
        plt.figure(figsize=(10, 4))
        sns.barplot(
            x=['Pregnancies', 'Glucose', 'BP', 'Skin', 'Insulin', 'BMI', 'DPF', 'Age'],
            y=features[0],
            palette='coolwarm'
        )
        plt.title("Patient's Medical Input Overview")
        plt.tight_layout()
        plt.savefig(input_graph_path)
        plt.close()

        # 6. Plot 2: Confidence %
        plt.figure(figsize=(6, 4))
        sns.barplot(x=['Non-Diabetic', 'Diabetic'], y=probabilities, palette='Blues_d')
        plt.title("Prediction Confidence (%)")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(result_graph_path)
        plt.close()

        # 7. Send to result.html with timestamp for cache-busting
        timestamp = datetime.now().timestamp()
        return render(request, 'predictor/result.html', {
            'prediction_text': result,
            'diet': diet,
            'medicine': medicine,
            'timings': timings,
            'input_graph': f'/static/{input_graph_filename}?v={timestamp}',
            'result_graph': f'/static/{result_graph_filename}?v={timestamp}'
        })

    return render(request, 'predictor/home.html')
