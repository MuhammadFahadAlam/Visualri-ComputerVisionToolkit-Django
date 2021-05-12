from django.urls import path

from . import views

app_name = 'videostream'
urlpatterns = [
    path('capture', views.capture, name='capture'),
    path('face', views.face, name='face'),
    path('gender', views.gender, name='gender'),
    path('emotion', views.emotion, name='emotion'),
    path('ip_input',views.ip_input,name='ip_input'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('webcam_feed', views.webcam_feed, name='webcam_feed'),
    path('save_image', views.save_image, name='save_image'),
]