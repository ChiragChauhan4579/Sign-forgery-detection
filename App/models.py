from django.db import models

class Image_field(models.Model):
    image1 = models.ImageField(upload_to = "images/")
    image2 = models.ImageField(upload_to = "images/")
