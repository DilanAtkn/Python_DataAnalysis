from app.views import PredictionView, PredictDataView
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register('prediction', PredictionView)
router.register('predic', PredictDataView)

urlpatterns = []
urlpatterns += router.urls

