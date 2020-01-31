from django.db import models

# Create your models here.
class Prediction(models.Model):
	incident_state = models.CharField(max_length=255)
	reassignment_count = models.IntegerField()
	reopen_count  = models.IntegerField()
	sys_mod_count = models.IntegerField() 
	contact_type = models.CharField(max_length=255)
	location  = models.IntegerField()
	category  = models.IntegerField()
	subcategory = models.IntegerField()
	priority  = models.IntegerField()

	def __str__(self):
		return self.incident_state

class PredictData(models.Model):
	prediction_fk = models.ForeignKey(Prediction, on_delete=models.DO_NOTHING)
	predic = models.CharField(max_length=255)
	def __str__(self):
		return self.predic
