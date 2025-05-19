from django import forms
class UploadAudioForm(forms.Form):
    audio = forms.FileField(label='Загрузите WAV файл', required=True)
