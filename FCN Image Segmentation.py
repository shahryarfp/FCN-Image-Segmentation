#!/usr/bin/env python
# coding: utf-8

# # FCN Image Segmentation
# ## Installing mxnet and gluoncv and importing needed Libraries

# In[1]:


get_ipython().system('pip install mxnet')
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
from IPython import display
get_ipython().system('pip install gluoncv')
import gluoncv
ctx = mx.cpu(0) # Using CPU


# ## Loading the Image

# In[2]:


image_path = "FCN image.png"
img = image.imread(image_path)
#normalizing image
from gluoncv.data.transforms.presets.segmentation import test_transform
img = test_transform(img, ctx)

display.display(display.Image(image_path, width=1024))


# ## Loading Pre-Trained Model and Predict the Image

# In[3]:


model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)


# In[4]:


# Predicting
output = model.predict(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()


# In[5]:


# Add color to the mask to display the result
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
mask = get_color_pallete(predict, 'pascal_voc')
mask.save('output.png')


# In[6]:


# Displaying the result
display.display(display.Image('output.png', width=1024))


# In[8]:




