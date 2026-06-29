from django.shortcuts import render
from django.conf import settings
from datetime import datetime
import os

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from .models import PredictionRecord


# Load model once at module level
model_path = os.path.join(settings.BASE_DIR, "diabetes_model.pkl")
model = joblib.load(model_path)


def home(request):
    return render(request, "predictor/home.html")


def predict(request):
    if request.method == "POST":

        def safe_cast(value, to_type, default=0):
            try:
                return to_type(value)
            except (ValueError, TypeError):
                return default

        # 1. Get user input
        Pregnancies = safe_cast(request.POST.get("Pregnancies"), int)
        Glucose = safe_cast(request.POST.get("Glucose"), int)
        BloodPressure = safe_cast(request.POST.get("BloodPressure"), int)
        SkinThickness = safe_cast(request.POST.get("SkinThickness"), int)
        Insulin = safe_cast(request.POST.get("Insulin"), int)
        BMI = safe_cast(request.POST.get("BMI"), float)
        DPF = safe_cast(request.POST.get("DiabetesPedigreeFunction"), float)
        Age = safe_cast(request.POST.get("Age"), int)

        features = np.array([
            [
                Pregnancies,
                Glucose,
                BloodPressure,
                SkinThickness,
                Insulin,
                BMI,
                DPF,
                Age
            ]
        ])

        # 2. Prediction and confidence
        prediction = model.predict(features)[0]
        prediction_value = int(prediction)

        probabilities = model.predict_proba(features)[0]
        probabilities_percent = probabilities * 100

        # 3. Save prediction record in database
        PredictionRecord.objects.create(
            pregnancies=Pregnancies,
            glucose=Glucose,
            blood_pressure=BloodPressure,
            skin_thickness=SkinThickness,
            insulin=Insulin,
            bmi=BMI,
            dpf=DPF,
            age=Age,
            result="Diabetic" if prediction_value == 1 else "Not Diabetic"
        )

        # 4. Define result and safe guidance
        if prediction_value == 1:
            result = "You are likely diabetic."
            diet = "Low sugar, high-fiber, low-carb diet. Avoid sweets. Eat oats, vegetables, and lean proteins."
            medicine = "Please consult a qualified healthcare professional for medication guidance."
            timings = "Medication timing should only be followed as advised by a doctor."
        else:
            result = "You are not diabetic."
            diet = "Balanced diet with fruits, vegetables, and controlled carbs."
            medicine = "No medication suggestion is provided by this educational app."
            timings = "N/A"

        # 5. Plot saving path
        static_dir = os.path.join(settings.BASE_DIR, "predictor", "static")
        os.makedirs(static_dir, exist_ok=True)

        input_graph_filename = "input_graph.png"
        result_graph_filename = "result_graph.png"

        input_graph_path = os.path.join(static_dir, input_graph_filename)
        result_graph_path = os.path.join(static_dir, result_graph_filename)

        # 6. Plot 1: Input overview
        plt.figure(figsize=(10, 4))
        sns.barplot(
            x=["Pregnancies", "Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age"],
            y=features[0],
            palette="coolwarm"
        )
        plt.title("Patient's Medical Input Overview")
        plt.ylabel("Input Value")
        plt.tight_layout()
        plt.savefig(input_graph_path)
        plt.close()

        # 7. Plot 2: Prediction confidence percentage
        plt.figure(figsize=(6, 4))
        sns.barplot(
            x=["Non-Diabetic", "Diabetic"],
            y=probabilities_percent,
            palette="Blues_d"
        )
        plt.title("Prediction Confidence (%)")
        plt.ylabel("Probability (%)")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(result_graph_path)
        plt.close()

        # 8. Send data to result page with timestamp for cache busting
        timestamp = datetime.now().timestamp()

        return render(request, "predictor/result.html", {
            "prediction_text": result,
            "diet": diet,
            "medicine": medicine,
            "timings": timings,
            "input_graph": f"/static/{input_graph_filename}?v={timestamp}",
            "result_graph": f"/static/{result_graph_filename}?v={timestamp}",
        })

    return render(request, "predictor/home.html")