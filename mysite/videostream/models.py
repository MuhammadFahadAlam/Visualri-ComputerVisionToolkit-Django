from django.db import models
from django.utils import timezone


# Create your models here.
INPUT_CHOICES = (
    ("1", "Camera"),
    ("2", "IP Webcam"),
)
class Video(models.Model):
    video_choice = models.CharField(max_length=50,choices=INPUT_CHOICES,default='1')
    confidence = models.FloatField(default=0.3)
    ip = models.CharField(max_length=50)
    action = models.CharField(max_length=50,default='FACES')
    created = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.video_choice