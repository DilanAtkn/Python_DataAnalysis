# Create your tests here.
from rest_framework.serializers import ModelSerializer

from app.models import Prediction, PredictData

class PredictionSerializer(ModelSerializer):
    class Meta:
        model = Prediction
        fields = "__all__"

class PredictDataSerializer(ModelSerializer):
    class Meta:
        model = PredictData
        fields = ["predic"]