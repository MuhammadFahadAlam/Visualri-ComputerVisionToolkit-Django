from django import forms
from .models import Video
# iterable
from django.contrib.auth.forms import UserCreationForm


# creating a form
class VideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ['video_choice', 'confidence']

        widgets = {
            #'video_choice':forms.TextInput(attrs={'':'**'}),
            'confidence': forms.TextInput(attrs={'type': 'range', 'step': '0.1', 'min': '0.1', 'max': '1'}),
                    }

class IpForm(forms.Form):
    ip = forms.CharField(max_length=50,required=True,label='Ip Address',widget=forms.TextInput(attrs={'placeholder': 'Enter Ip Address of Webcam'}))