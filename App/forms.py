from django import forms

class Image(forms.Form):
    image1 = forms.ImageField()
    image2 = forms.ImageField()