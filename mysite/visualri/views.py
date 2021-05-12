'''from django.shortcuts import render, redirect
from .forms import UploadForm
from .models import Upload
from PIL import Image
import numpy as np
from .utils import blur,to_data_uri

def index(request):
    #request.method = 'GET'
    print(request.method)
    upload = Upload.objects.latest('date_posted')
    if request.method == 'POST':
        form = UploadForm(request.POST,request.FILES)
        print(form.is_valid())
        #print(form.errors)
        if form.is_valid():
            #print(form.save())
            form.save()
            upload = Upload.objects.latest('date_posted')
            print(upload.kernel)
            #print(form.cleaned_data['image'])
            #model = Upload.get_instance()
            file_name = request.FILES['image']
            pil_img = Image.open(file_name)
            cv_img = np.array(pil_img)
            processed=blur(cv_img,upload.kernel)
            im_pil = Image.fromarray(processed)
            image_uri = to_data_uri(im_pil)
            #print(image_uri)
            return render(request, 'visualri/index.html', {'form': form, 'upload': upload,'image_uri':image_uri})

            #processed_image = model.run_png
            #upload = Upload.objects.latest('date_posted')
            #return render(request, 'visualri/index.html',{'form':form})
    else:
        form = UploadForm()
    return render(request, 'visualri/index.html',{'form':form,'upload':upload})
# Create your views here.

def uploaded(request):
    return render(request,'visualri/uploaded.html')'''

#-------------------------------------------------------SAAD----------------------------------#

from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render, redirect
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from .forms import UploadForm, TextForm
from .models import Upload
from PIL import Image
import numpy as np
from .utils import get_filtered_image, to_data_uri, style_filter, color_filter
from django.conf import settings
import cv2
import tkinter as tk
from tkinter import simpledialog

image_uri = None

def home(request):
    return render(request, 'visualri/home.html')

def imageProcessing(request):
    global image_uri

    model_instance = Upload.objects.latest('created')

    return render(request, 'visualri/imageProcessing.html', {'model_instance': model_instance})

def computerVision(request):
    return render(request, 'visualri/computerVision.html')

def imageInput(request):
    global image_uri
    		
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            # print(form.save())
            #upload = Upload.objects.latest('created')
            # print(upload.kernel)
            # print(form.cleaned_data['image'])
            # model = Upload.get_instance()
            #file_name = request.FILES['image']
            if bool(request.FILES) == False:
                model_instance = form.save(commit=False)
                if model_instance.action == 'NO_FILTER':
                    pass
                elif model_instance.action == 'COLORIZED':
                    model_instance.image = 'colorized.png'
                elif model_instance.action == 'BLURRED':
                    pass
                elif model_instance.action == 'POLARIZE':
                    pass
                elif model_instance.action == 'COOLING':
                    pass
                elif model_instance.action == 'WARMING':
                    pass
                elif model_instance.action == 'BRIGHTNESS':
                    model_instance.image = 'bright.jpg'
                elif model_instance.action == 'CONTRAST':
                    model_instance.image = 'contrast.png'
                elif model_instance.action =='MEDIAN BLUR':
                    model_instance.image = 'median.jpeg'

                #upload = Upload.objects.latest('created')[0]
                #q= Upload.objects.all().order_by('created')
                #upload=q.reverse()[1]
                try:
                    pil_img = Image.open(model_instance.image[1:])
                except:
                    pil_img = Image.open(model_instance.image)

                cv_img = np.array(pil_img)
                processed=get_filtered_image(cv_img,model_instance.action)
                im_pil = Image.fromarray(processed)
                image_uri = to_data_uri(im_pil)
            else:
                model_instance = form.save()
                file_name = request.FILES['image']
            #upload.image = file_name
                pil_img = Image.open(file_name)
                cv_img = np.array(pil_img)
                processed=get_filtered_image(cv_img,model_instance.action)
                im_pil = Image.fromarray(processed)
                image_uri = to_data_uri(im_pil)
            # print(upload.image)
            #upload.action = 'BLURRED'
            # print(upload.action)
            # uri = upload.save_model()
            # print(a)
            
        #return redirect(request, 'visualri/computerVision.html', {'form': form, 'model_instance': model_instance, 'image_uri': image_uri})
        
        return redirect('imageProcessing')    
        
            # processed_image = model.run_png
            # upload = Upload.objects.latest('date_posted')
            # return render(request, 'visualri/index.html',{'form':form})
    else:
        form = UploadForm()

        return render(request, 'visualri/imageInput.html', {'form': form})



#@login_required
def filter(request):
    
    global action
    #request.method = 'GET'
    
    if request.method == 'POST':

        #print(request.POST)
        form = TextForm(request.POST)

        # print(form.is_valid())
        # print(form.errors)
        
        if form.is_valid():
            action = form.cleaned_data['btn']
            print(action)
            model_instance = Upload.objects.latest('created')
            model_instance.action = action
            model_instance.save()


            print(model_instance.action)

            '''
            #upload = Upload.objects.latest('created')
            # print(upload.kernel)
            # print(form.cleaned_data['image'])
            # model = Upload.get_instance()
            #file_name = request.FILES['image']
            if bool(request.FILES) == False:
                model_instance = form.save(commit=False)
                if model_instance.action == 'NO_FILTER':
                    pass
                elif model_instance.action == 'COLORIZED':
                    model_instance.image = 'colorized.png'
                elif model_instance.action == 'BLURRED':
                    pass
                elif model_instance.action == 'POLARIZE':
                    pass
                elif model_instance.action == 'COOLING':
                    pass
                elif model_instance.action == 'WARMING':
                    pass
                elif model_instance.action == 'BRIGHTNESS':
                    model_instance.image = 'bright.jpg'
                elif model_instance.action == 'CONTRAST':
                    model_instance.image = 'contrast.png'
                elif model_instance.action =='MEDIAN BLUR':
                    model_instance.image = 'median.jpeg'

                #upload = Upload.objects.latest('created')[0]
                #q= Upload.objects.all().order_by('created')
                #upload=q.reverse()[1]
                print(model_instance.image)
                print('asasas')
                print(model_instance.image)
                pil_img = Image.open(model_instance.image)
                cv_img = np.array(pil_img)
                processed=get_filtered_image(cv_img,model_instance.action)
                im_pil = Image.fromarray(processed)
                image_uri = to_data_uri(im_pil)
            else:
                model_instance = form.save()
                file_name = request.FILES['image']
            #upload.image = file_name
                pil_img = Image.open(file_name)
                cv_img = np.array(pil_img)
                processed=get_filtered_image(cv_img,model_instance.action)
                im_pil = Image.fromarray(processed)
                image_uri = to_data_uri(im_pil)
            # print(upload.image)
            #upload.action = 'BLURRED'
            # print(upload.action)
            # uri = upload.save_model()
            # print(a)
            return render(request, 'visualri/index.html',{'form': form, 'model_instance': model_instance, 'image_uri': image_uri})

            # processed_image = model.run_png
            # upload = Upload.objects.latest('date_posted')
            # return render(request, 'visualri/index.html',{'form':form})
    else:
        form = UploadForm()

        return render(request, 'visualri/index.html', {'form': form})

        '''

    return render(request, 'visualri/filter.html')

def colorMap(request):
    return render(request, 'visualri/colorMap.html')

def styleTransfer(request):
    return render(request, 'visualri/styleTransfer.html')

def result(request):

    global image_uri
    global action
    #request.method = 'GET'
    
    if request.method == 'POST':

        #print(request.POST)
        form = TextForm(request.POST)

        # print(form.is_valid())
        # print(form.errors)
        
        if form.is_valid():
            action = form.cleaned_data['btn']
            model_instance = Upload.objects.latest('created')
            model_instance.action = action
            model_instance.save()
            try:
                pil_img = Image.open(model_instance.image.url[1:])
            except:
                pil_img = Image.open(model_instance.image.url)
            cv_img = np.array(pil_img)
            processed=get_filtered_image(cv_img,model_instance.action)
            try:
                cv2.imwrite('media/captured/temp.jpg',processed[:,:,::-1])
            except:
                cv2.imwrite('media/captured/temp.jpg',processed)
            im_pil = Image.fromarray(processed)
            image_uri = to_data_uri(im_pil)
            
            print(model_instance.image, model_instance.action)
            

    return render(request, 'visualri/result.html', {'model_instance': model_instance, 'image_uri': image_uri})

def download(request):
    try:
        print(request.method)
        image = cv2.imread('media/captured/temp.jpg')
        ROOT = tk.Tk()
        ROOT.withdraw()
        # the input dialog
        USER_INP = simpledialog.askstring(title="DOWNLOAD IMAGE",
                                        prompt="ENTER IMAGE NAME")
        name = 'media/downloads/'+ str(USER_INP) +'.jpg'
        print(name)
        print('processed',image)
        cv2.imwrite(name,image)
    except: pass

    return render(request, 'visualri/imageProcessing.html')

def resultStyle(request):

    global image_uri
    global action
    #request.method = 'GET'
    
    if request.method == 'POST':

        #print(request.POST)
        form = TextForm(request.POST)

        # print(form.is_valid())
        # print(form.errors)
        
        if form.is_valid():
            action = form.cleaned_data['btn']
            model_instance = Upload.objects.latest('created')
            model_instance.action = action
            model_instance.save()

            pil_img = Image.open(model_instance.image)
            cv_img = np.array(pil_img)
            processed=style_filter(cv_img,model_instance.action)
            cv2.imwrite('media/captured/temp.jpg',processed[:,:,::-1])
            im_pil = Image.fromarray(processed)
            image_uri = to_data_uri(im_pil)

            print(model_instance.image, model_instance.action)

    return render(request, 'visualri/result.html', {'model_instance': model_instance, 'image_uri': image_uri})


def resultColor(request):

    global image_uri
    global action
    #request.method = 'GET'
    
    if request.method == 'POST':

        #print(request.POST)
        form = TextForm(request.POST)

        # print(form.is_valid())
        # print(form.errors)
        
        if form.is_valid():
            action = form.cleaned_data['btn']
            model_instance = Upload.objects.latest('created')
            model_instance.action = action
            model_instance.save()

            pil_img = Image.open(model_instance.image)
            cv_img = np.array(pil_img)
            processed=color_filter(cv_img,model_instance.action)
            cv2.imwrite('media/captured/temp.jpg',processed[:,:,::-1])
            im_pil = Image.fromarray(processed)
            image_uri = to_data_uri(im_pil)

            print(model_instance.image, model_instance.action)

    return render(request, 'visualri/result.html', {'model_instance': model_instance, 'image_uri': image_uri})