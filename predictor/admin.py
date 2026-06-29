from django.contrib import admin
from .models import PredictionRecord

# Register your models here.

@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ("glucose", "bmi", "age", "result", "created_at")
    search_fields = ("result",)
    list_filter = ("result", "created_at")