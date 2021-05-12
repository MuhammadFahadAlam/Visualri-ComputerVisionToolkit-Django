'''from django import forms
from .models import Upload

class UploadForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ['image','kernel']
        widgets = {
            'kernel': forms.TextInput(attrs={'type': 'range','step':'1','min':'1','max':'10'})
        }
        #widgets = {'parameters': forms.NumberInput(attrs={'type': 'range','step':'0.1'})}'''


#------------------------------------------------Saad Modification---------------------------------#

from django import forms
from django.contrib.auth.decorators import login_required
from .models import Upload, Upload_Color_Map, Upload_Style, Upload_Detect




class UploadForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ['image']


class UploadForm_Color_Map(forms.ModelForm):
    class Meta:
        model = Upload_Color_Map
        fields = ['image', 'action']

class UploadForm_Style(forms.ModelForm):
    class Meta:
        model = Upload_Style
        fields = ['image', 'action']

class UploadForm_Detect(forms.ModelForm):
    class Meta:
        model = Upload_Detect
        fields = ['image']


class TextForm(forms.Form):
    btn = forms.CharField(max_length='50')

