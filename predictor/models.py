from django.db import models

class PredictionRecord(models.Model):
    pregnancies = models.IntegerField()
    glucose = models.IntegerField()
    blood_pressure = models.IntegerField()
    skin_thickness = models.IntegerField()
    insulin = models.IntegerField()
    bmi = models.FloatField()
    dpf = models.FloatField(verbose_name="Diabetes Pedigree Function")
    age = models.IntegerField()
    result = models.CharField(max_length=20)  # "Diabetic" or "Not Diabetic"
    created_at = models.DateTimeField(auto_now_add=True)

