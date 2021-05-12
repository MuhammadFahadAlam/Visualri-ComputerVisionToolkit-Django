"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from users import views as user_views
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static
from visualri import views as visualri_views

app_name = 'mysite'

urlpatterns = [
    path('', include('visualri.urls')),
    path('',visualri_views.home, name='home'),
    path('filter/', visualri_views.filter, name='filter'),
    path('result/', visualri_views.result, name='result'),
    path('resultStyle/', visualri_views.resultStyle, name='resultStyle'),
    path('resultColor/', visualri_views.resultColor, name='resultColor'),
    path('styleTransfer/', visualri_views.styleTransfer, name='styleTransfer'),
    path('colorMap', visualri_views.colorMap, name='colorMap'),
    path('download', visualri_views.download, name='download'),
    path('computerVision/',visualri_views.computerVision, name='computerVision'),
    path('imageProcessing/', visualri_views.imageProcessing, name='imageProcessing'),
    path('imageInput/', visualri_views.imageInput, name='imageInput'),
    path('register/',user_views.register, name = 'register'),
    path('profile/',user_views.profile, name = 'profile'),
    path('videostream/',include('videostream.urls')),
    path('login/',auth_views.LoginView.as_view(template_name='users/login.html'), name = 'login'),
    path('logout/',auth_views.LogoutView.as_view(template_name='users/logout.html'), name = 'logout'),
    path('admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)