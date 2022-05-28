#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,Activation,BatchNormalization,Add,Multiply,Concatenate,GlobalAveragePooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
# import datetime
# import pandas as pd
# import time


# In[2]:
`

train_files=['images/train/'+filename for filename in os.listdir('images/train')]
test_files=['images/test/'+filename for filename in os.listdir('images/test')]


# In[3]:


def _parse_function(filename):
    '''This function performs adding noise to the image given by Dataset'''
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)/255.
    resize_image = tf.image.resize(image, [40,40])
    image=resize_image
    
    noise_level=np.random.choice(NOISE_LEVELS)
    noisy_image=image+tf.random.normal(shape=(40,40,3),mean=0,stddev=noise_level/255)
    noisy_image=tf.clip_by_value(noisy_image, clip_value_min=0., clip_value_max=1.)

    return noisy_image,image


# In[4]:


BATCH_SIZE=64
NOISE_LEVELS=[15,25,50] 

#Creating the Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(np.array(train_files)) 
train_dataset = train_dataset.map(_parse_function)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(np.array(test_files))
test_dataset = test_dataset.map(_parse_function)
test_dataset = test_dataset.batch(BATCH_SIZE)


# In[5]:


iterator = iter(train_dataset)
a, b = iterator.get_next()

print('Shape of single batch of x : ',a.shape)
print('Shape of single batch of y : ',b.shape)


# In[6]:


#Plotting the images from dataset to verify the dataset
fig, axs = plt.subplots(1,10,figsize=(20,4))
for i in range(10):
  axs[i].imshow(a[i])
  axs[i].get_xaxis().set_visible(False)
  axs[i].get_yaxis().set_visible(False)
fig.suptitle('Images with added noise',fontsize=20)
plt.show()
fig, axs = plt.subplots(1,10,figsize=(20,4))
for i in range(10):
  axs[i].imshow(b[i])
  axs[i].get_xaxis().set_visible(False)
  axs[i].get_yaxis().set_visible(False)
fig.suptitle('Original Images',fontsize=20)
plt.show()


# In[12]:


def get_patches(file_name,patch_size,crop_sizes):
    '''This functions creates and return patches of given image with a specified patch_size'''
    image = cv2.imread(file_name) 
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    height, width , channels= image.shape
    patches = []
    for crop_size in crop_sizes: #We will crop the image to different sizes
        crop_h, crop_w = int(height*crop_size),int(width*crop_size)
        image_scaled = cv2.resize(image, (crop_w,crop_h), interpolation=cv2.INTER_CUBIC)
        for i in range(0, crop_h-patch_size+1, patch_size):
            for j in range(0, crop_w-patch_size+1, patch_size):
                x = image_scaled[i:i+patch_size, j:j+patch_size] # This gets the patch from the original image with size patch_size x patch_size
                patches.append(x)
    return patches

def create_image_from_patches(patches,image_shape):
  '''This function takes the patches of images and reconstructs the image'''
  image=np.zeros(image_shape) # Create a image with all zeros with desired image shape
  patch_size=patches.shape[1]
  p=0
  for i in range(0,image.shape[0]-patch_size+1,patch_size):
    for j in range(0,image.shape[1]-patch_size+1,patch_size):
      image[i:i+patch_size,j:j+patch_size]=patches[p] # Assigning values of pixels from patches to image
      p+=1
  return np.array(image)

def predict_fun(model,image_path,noise_level=30):
  #Creating patches for test image
  patches=get_patches(image_path,40,[1])
  test_image=cv2.imread(image_path)

  patches=np.array(patches)
  ground_truth=create_image_from_patches(patches,test_image.shape)

  #predicting the output on the patches of test image
  patches = patches.astype('float32') /255.
  patches_noisy = patches+ tf.random.normal(shape=patches.shape,mean=0,stddev=noise_level/255) 
  patches_noisy = tf.clip_by_value(patches_noisy, clip_value_min=0., clip_value_max=1.)
  noisy_image=create_image_from_patches(patches_noisy,test_image.shape)

  denoised_patches=model.predict(patches_noisy)
  denoised_patches=tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)

  #Creating entire denoised image from denoised patches
  denoised_image=create_image_from_patches(denoised_patches,test_image.shape)

  return patches_noisy,denoised_patches,ground_truth/255.,noisy_image,denoised_image


def plot_patches(patches_noisy,denoised_patches):
  fig, axs = plt.subplots(2,10,figsize=(20,4))
  for i in range(10):

    axs[0,i].imshow(patches_noisy[i])
    axs[0,i].title.set_text(' Noisy')
    axs[0,i].get_xaxis().set_visible(False)
    axs[0,i].get_yaxis().set_visible(False)

    axs[1,i].imshow(denoised_patches[i])
    axs[1,i].title.set_text('Denoised')
    axs[1,i].get_xaxis().set_visible(False)
    axs[1,i].get_yaxis().set_visible(False)
  plt.show()

def plot_predictions(ground_truth,noisy_image,denoised_image):
  fig, axs = plt.subplots(1,3,figsize=(15,15))
  axs[0].imshow(ground_truth)
  axs[0].title.set_text('Ground Truth')
  axs[1].imshow(noisy_image)
  axs[1].title.set_text('Noisy Image')
  axs[2].imshow(denoised_image)
  axs[2].title.set_text('Denoised Image')
  plt.show()


#https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def PSNR(gt, image, max_value=1):
    """"Function to calculate peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((gt - image) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


# In[8]:


def EAM(input):

  x=Conv2D(64, (3,3), dilation_rate=1,padding='same',activation='relu')(input)
  x=Conv2D(64, (3,3), dilation_rate=2,padding='same',activation='relu')(x)

  y=Conv2D(64, (3,3), dilation_rate=3,padding='same',activation='relu')(input)
  y=Conv2D(64, (3,3), dilation_rate=4,padding='same',activation='relu')(y)

  z=Concatenate(axis=-1)([x,y])
  z=Conv2D(64, (3,3),padding='same',activation='relu')(z)
  add_1=Add()([z, input])

  z=Conv2D(64, (3,3),padding='same',activation='relu')(add_1)
  z=Conv2D(64, (3,3),padding='same')(z)
  add_2=Add()([z,add_1])
  add_2 = Activation('relu')(add_2)

  z=Conv2D(64, (3,3),padding='same',activation='relu')(add_2)
  z=Conv2D(64, (3,3),padding='same',activation='relu')(z)
  z=Conv2D(64, (1,1),padding='same')(z)
  add_3=Add()([z,add_2])
  add_3 = Activation('relu')(add_3)

  z = GlobalAveragePooling2D()(add_3)
  z = tf.expand_dims(z,1)
  z = tf.expand_dims(z,1)
  z=Conv2D(4, (3,3),padding='same',activation='relu')(z)
  z=Conv2D(64, (3,3),padding='same',activation='sigmoid')(z)
  mul=Multiply()([z, add_3])

  return mul


# In[9]:


def RIDNET():

  input = Input((40, 40, 3),name='input')
  feat_extraction =Conv2D(64, (3,3),padding='same')(input)
  eam_1=EAM(feat_extraction)
  eam_2=EAM(eam_1)
  eam_3=EAM(eam_2)
  eam_4=EAM(eam_3)
  x=Conv2D(3, (3,3),padding='same')(eam_4)
  add_2=Add()([x, input])
  
  model=Model(input,add_2)

  return model


# In[10]:


tf.keras.backend.clear_session()
tf.random.set_seed(6908)
ridnet = RIDNET()


# In[11]:


ridnet.compile(optimizer=tf.keras.optimizers.Adam(1e-03), loss=tf.keras.losses.MeanAbsoluteError())


# In[12]:


ridnet.summary()


# In[13]:


def scheduler(epoch,lr):
  return lr*0.9


# In[16]:


ridnet.fit( train_dataset,shuffle=True,epochs=20,validation_data= test_dataset)


# In[17]:


patches_noisy,denoised_patches,ground_truth,noisy_image,denoised_image=predict_fun(ridnet,'images/test/2 (2).jpg',noise_level=65)
print('PSNR of Noisy Image : ',PSNR(ground_truth,noisy_image))
print('PSNR of Denoised Image : ',PSNR(ground_truth,denoised_image))
plot_patches(patches_noisy,denoised_patches)


# In[19]:


plot_predictions(ground_truth,noisy_image,denoised_image)


# In[26]:


results=pd.DataFrame(columns=['Noise Level','RIDNET'])


# In[27]:


def get_results(results,noise_level):

 # patches_noisy_a,denoised_patches_a,ground_truth,noisy_image_a,denoised_image_a=predict_fun(ridnet,'data/test/102061.jpg',noise_level=noise_level)
  #patches_noisy_d,denoised_patches_d,ground_truth,noisy_image_d,denoised_image_d=predict_fun(dncnn,'data/test/102061.jpg',noise_level=noise_level)
  patches_noisy_r,denoised_patches_r,ground_truth,noisy_image_r,denoised_image_r=predict_fun(ridnet,'images/test/2 (2).jpg',noise_level=noise_level)
  results.loc[len(results.index)]=[noise_level,PSNR(ground_truth,denoised_image_r)]

  return results


# In[28]:


results=get_results(results,15)
results=get_results(results,20)
results=get_results(results,25)
results=get_results(results,30)
results=get_results(results,40)
results=get_results(results,45)


# In[29]:


print('Tabulating the model results with different noise level \n')
results.head(6)


# In[32]:


ridnet.save('ridnet.hd5')


# In[33]:


ridnet.save_weights('ridnetweights')


# In[5]:


import tensorflow as tf
ridnet=tf.keras.models.load_model('ridnet.hd5')


# In[6]:


from tensorflow.keras.utils import plot_model


# In[10]:


# ! pip install pydot
# plot_model(ridnet,show_shapes=True,to_file='ridnet.png')


# In[8]:


ridnet.summary()


# In[ ]:





# In[17]:


noise_levels=[i for i in range(5,70,5)]
psnr_noisy=[]
psnr_denoised=[]
for i in noise_levels:
  patches_noisy,denoised_patches,ground_truth,noisy_image,denoised_image=predict_fun(ridnet,'1.jpg',noise_level=i)
  psnr_denoised.append(PSNR(ground_truth,denoised_image))
  psnr_noisy.append(PSNR(ground_truth,noisy_image))

plt.scatter(noise_levels,psnr_denoised,marker='^',s=50,c='green',label='PSNR of Denoised Images')
plt.scatter(noise_levels,psnr_noisy,marker='o',s=50,c='red',label='PSNR of Noisy Images')
#plt.axhline(20.2044,linestyle='--',label='PSNR of Noisy Image')
plt.xlabel('Noise Level')
plt.ylabel('PSNR')
plt.title('Performance of model  with different noise levels')
plt.legend()
plt.show()


# In[ ]:




