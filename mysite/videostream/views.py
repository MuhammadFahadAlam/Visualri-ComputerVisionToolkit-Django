import base64
from visualri import models
import numpy as np
from io import BytesIO
import cv2
from django.conf import settings
from visualri.models import Upload
from visualri.forms import UploadForm
import os
from django.core.files import File  # you need this somewhere
import urllib


# The following actually resides in a method of my model


from PIL import Image
from videostream.models import Video
from django.shortcuts import render,redirect,reverse
from django.http.response import StreamingHttpResponse
from videostream.camera import VideoCamera, IPWebCam
from .forms import VideoForm,IpForm
from .camera import init_detectgender,init_emotion
# Create your views here.

model_instance = None

def capture(request):
	global model_instance
	action="CAPTURE"
	#print(request.method)
	if request.method == 'POST':
		form = VideoForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
			#print('form',form)
			model_instance = Video.objects.latest('created')
			model_instance.action = action
			#print(model_instance.action)
			if model_instance.video_choice=='1':
				return render(request, 'videostream/camera.html', {'form': form,'action':action})
			else:
				
				return redirect(reverse('videostream:ip_input'))
	else: 
		form = VideoForm()
		return render(request, 'videostream/main.html', {'form':form})

def save_image(request):
	print(request.method)
	if request.method == 'POST':
	
		cap = cv2.VideoCapture(0)
		success, image = cap.read()
		cap.release()
		image = cv2.flip(image,1)
		model = Upload()
		print(settings.MEDIA_ROOT)
		cv2.imwrite('media/captured/image.jpg',image)
		model.image = 'captured/image.jpg'
		model.save()
	
		return redirect(reverse('imageProcessing'))
	else:
		return redirect(reverse('imageInput'))





def face(request):
	global model_instance
	action="FACE"
	#print(request.method)
	if request.method == 'POST':
		form = VideoForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
			#print('form',form)
			model_instance = Video.objects.latest('created')
			model_instance.action = action
			#print(model_instance.action)
			if model_instance.video_choice=='1':
				return render(request, 'videostream/camera.html', {'form': form,'action':action})
			else:
				
				return redirect(reverse('videostream:ip_input'))
	else: 
		form = VideoForm()
		return render(request, 'videostream/face.html', {'form':form})



def emotion(request):
	global model_instance
	action = "EMOTION"
	print(request.method)
	if request.method == 'POST':
		form = VideoForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
			#print('form',form)
			model_instance = Video.objects.latest('created')
			model_instance.action = action
			#print(model_instance.action)
			if model_instance.video_choice=='1':
				return render(request, 'videostream/camera.html', {'form': form,'action':action})
			else:
				return redirect(reverse('videostream:ip_input'))
	else:
		form = VideoForm()
		return render(request, 'videostream/emotion.html', {'form':form})

def gender(request):
	global model_instance
	action="GENDER"
	print(request.method)
	if request.method == 'POST':
		form = VideoForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
			#print('form',form)
			model_instance = Video.objects.latest('created')
			model_instance.action = action
			#print(model_instance.action)
			if model_instance.video_choice=='1':
				return render(request, 'videostream/camera.html', {'form': form,'action':action})
			else:
				return redirect(reverse('videostream:ip_input'))
	else:
		form = VideoForm()
		return render(request, 'videostream/gender.html', {'form':form})


def ip_input(request):
	global model_instance
	if request.method == 'POST':
		form = IpForm(request.POST)
		if form.is_valid():
			print(form.cleaned_data.get('ip'))
			print(form.cleaned_data.get('ip'))
			if model_instance == None:
				return redirect(reverse('videostream:face'))
			else:
				model_instance.ip = form.cleaned_data.get('ip')
				print(model_instance.ip)
				return render(request, 'videostream/ip.html', {'form': form})

	else:
		form = IpForm()
		return render(request, 'videostream/ip.html', {'form':form})


def gen(camera):
	global model_instance
	net = None
	if model_instance.action == "GENDER":
		net = init_detectgender()
	elif model_instance.action == "EMOTION":
		net = init_emotion()
	while True:
		global frame
		frame = camera.get_frame(model_instance,net)
		
		
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')


def webcam_feed(request):
	return StreamingHttpResponse(gen(IPWebCam(model_instance.ip)),
					content_type='multipart/x-mixed-replace; boundary=frame')
