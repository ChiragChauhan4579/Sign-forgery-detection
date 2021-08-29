from django.shortcuts import render
from keras_preprocessing import image
import numpy as np
import tensorflow as tf
import os
from .forms import Image
from .models import Image_field


DIR = 'images/'

model = tf.keras.models.load_model("App/my_model.h5")

def process(i1,i2):
    x = image.load_img(i1, target_size=(100, 100))    
    x = image.img_to_array(x)
    x = tf.image.rgb_to_grayscale(x)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    y = image.load_img(i2, target_size=(100, 100))    
    y = image.img_to_array(y)
    y = tf.image.rgb_to_grayscale(y)
    y = np.expand_dims(y, axis=0)
    y = y/255.0
    y_pred = model.predict([x,y])
    y_pred = np.argmax(y_pred)
    return y_pred

def home(request):
    form = Image(request.POST,request.FILES)
    if request.method == "POST":
        if form.is_valid():
            i1 = request.FILES['image1']
            i2 = request.FILES['image2']
            obj = Image_field.objects.create(image1=i1,image2=i2)
            obj.save()
            res = process(DIR + i1.name,DIR + i2.name)
            if res==1:
                note = 'Forged Signature' 
            else:
                note = 'Real Signature'
            return render(request, "home2.html",{'Form':form,'Note':note})
    else:
        form = Image()
    return render(request, "home2.html",{'Form':form})
