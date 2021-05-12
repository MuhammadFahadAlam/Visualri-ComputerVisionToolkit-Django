'''from django.db import models
from django.utils import timezone

# Create your models here.
class Upload(models.Model):
    date_posted = models.DateTimeField(default=timezone.now)
    title = models.CharField(max_length=100,default='Blurring')
    kernel = models.IntegerField(default=5)
    image = models.ImageField(upload_to='profile_pics',null=True,blank=False)

    def __str__(self):
        return self.title'''

#------------------------------------------Saad Test---------------------------------#
import datetime
from django.core.files.base import ContentFile
from django.db import models
from .utils import get_filtered_image
from PIL import Image
import numpy as np
from io import BytesIO
from django.utils import timezone



class Upload(models.Model):
    # total = Upload.objects.all().count()
    ACTION_CHOICES = [
        ('THRESHOLDING', 'Thresholding'),
        ('BGR', 'BGR'),
        ('HSV', 'HSV'),
        ('GRAYSCALE', 'Grayscale'),
        ('STYLIZATION', 'Stylization'),
        ('DETAIL ENHANCE', 'Detail Enhance'),
        ('COLOR SKETCH', 'Color Sketch'),
        ('BLURRED', 'Bipolar Blur'),
        ('MEDIAN BLUR', 'Median Blur'),
        ('POLARIZE', 'Polarize'),
        ('EDGE PRESERVING FILTER', 'Edge Preserving Filter'),
        ('BRIGHTNESS', 'Brightness'),
        ('CONTRAST', 'Contrast'),
        ('COLORIZED', 'Colorize BW image'),
        ('SUPERRES', 'Super Resolution'),

    ]

    image = models.ImageField(upload_to='profile_pics',null=True,blank=True, default='default.jpg')
    action = models.CharField(max_length=50,choices=ACTION_CHOICES, default='POLARIZE')
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(default=timezone.now)
    #kernel = models.IntegerField(default=5)

    def __str__(self):
        return self.action



class Upload_Color_Map(models.Model):
    ACTION_CHOICES = [
        ('0', 'Autumn'),
        ('1', 'Bone'),
        ('8', 'Cool'),
        ('11', 'Hot'),
        ('9', 'HSV'),
        ('2', 'Jet'),
        ('5', 'Ocean'),
        ('12', 'Parula'),
        ('10', 'Pink'),
        ('4', 'Rainbow'),
        ('7', 'Spring'),
        ('6', 'Summer'),
        ('3', 'Winter'),
        ('13', 'Magma'),
        ('14', 'Inferno'),
        ('15', 'Plasma'),
        ('16', 'VIRIDIS'),
        ('17', 'CIVIDIS'),
        ('18', 'Twilight'),
        ('19', 'Twilight_shifted'),
        ('20', 'Turbo'),
        ('21', 'DeepGreen'),
]

    image = models.ImageField(upload_to='images',null=True,blank=True, default='default.jpg')
    action = models.CharField(max_length=50,choices=ACTION_CHOICES, default=15)
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(default=timezone.now)
    def __str__(self):
        return self.action



class Upload_Style(models.Model):
    ACTION_CHOICES = [
        ('rain', 'rain'),
        ('pink', 'pink'),
        ('traingle', 'triangle'),
        ('gold_black', 'gold_black'),
        ('flame', 'flame'),
        ('Fire_Style', 'Fire_Style'),
        ('landscape', 'landscape'),
        ('feathers', 'feathers'),
        ('candy', 'candy'),
        ('composition_vii', 'composition_vii'),
        ('udnie', 'udnie'),
        ('the_wave', 'the_wave'),
        ('the_scream','the_scream'),
        ('mosaic', 'mosaic'),
        ('la_muse', 'la_muse'),
        ('starry_night', 'starry_night'),
        ]

    image = models.ImageField(upload_to='images',null=True,blank=True, default='default.jpg')
    action = models.CharField(max_length=50,choices=ACTION_CHOICES, default=15)
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(default=timezone.now)
    def __str__(self):
        return self.action


class Upload_Detect(models.Model):
    image = models.ImageField(upload_to='images',null=True,blank=True, default='default.jpg')
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(default=timezone.now)
