import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog

import PIL.Image, PIL.ImageTk

# Loading the model
ridnet=tf.keras.models.load_model('ridnet.hd5')




# Helper Functions


def PSNR(gt, image, max_value=1):
    mse = np.mean((gt - image) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def get_patches_of_image(filename,patch_size,crop_sizes):
    image = cv2.imread(filename) 
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    height, width , channels= image.shape
    patches = []
    for crop_size in crop_sizes:
        crop_h, crop_w = int(height*crop_size),int(width*crop_size)
        image_scaled = cv2.resize(image, (crop_w,crop_h), interpolation=cv2.INTER_CUBIC)
        for i in range(0, crop_h-patch_size+1, patch_size):
            for j in range(0, crop_w-patch_size+1, patch_size):
                x = image_scaled[i:i+patch_size, j:j+patch_size] 
                patches.append(x)
    return patches

def create_image_from_patches(patches,image_shape):
  image=np.zeros(image_shape)
  patch_size=patches.shape[1]
  p=0
  for i in range(0,image.shape[0]-patch_size+1,patch_size):
    for j in range(0,image.shape[1]-patch_size+1,patch_size):
      image[i:i+patch_size,j:j+patch_size]=patches[p]
      p+=1
  return np.array(image)

def predict_fun(model,image_path,noise_level=30):
  patches=get_patches_of_image(image_path,40,[1])
  test_image=cv2.imread(image_path)

  patches=np.array(patches)
  ground_truth=create_image_from_patches(patches,test_image.shape)

  patches = patches.astype('float32') /255.
  noised_patches = patches+ tf.random.normal(shape=patches.shape,mean=0,stddev=noise_level/255) 
  noised_patches = tf.clip_by_value(noised_patches, clip_value_min=0., clip_value_max=1.)
  noisy_image=create_image_from_patches(noised_patches,test_image.shape)

  denoised_patches=model.predict(noised_patches)
  denoised_patches=tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)
  denoised_image=create_image_from_patches(denoised_patches,test_image.shape)

  return noised_patches,denoised_patches,ground_truth/255.,noisy_image,denoised_image


def plot_patches(noised_patches,denoised_patches):
  fig, axs = plt.subplots(2,10,figsize=(20,4))
  for i in range(10):

    axs[0,i].imshow(noised_patches[i])
    axs[0,i].title.set_text('Noised Image')
    axs[0,i].get_xaxis().set_visible(False)
    axs[0,i].get_yaxis().set_visible(False)

    axs[1,i].imshow(denoised_patches[i])
    axs[1,i].title.set_text('Denoised Image')
    axs[1,i].get_xaxis().set_visible(False)
    axs[1,i].get_yaxis().set_visible(False)
  plt.show()

def plot_predictions(ground_truth,noisy_image,denoised_image):
  fig, axs = plt.subplots(1,3,figsize=(15,15))
  axs[0].imshow(ground_truth)
  axs[0].title.set_text('Original Image')
  axs[1].imshow(noisy_image)
  axs[1].title.set_text('Noisy Image')
  axs[2].imshow(denoised_image)
  axs[2].title.set_text('Denoised Image')
  plt.show()


window = Tk()

window.title('Image Denoising Application')

window.geometry("700x500")





bg= Image.open("download.jfif")
def on_resize(event):
    image = bg.resize((event.width, event.height), Image.ANTIALIAS)
    l.image = ImageTk.PhotoImage(image)
    l.config(image=l.image)



def browseFiles():
	filename = filedialog.askopenfilename(initialdir = "Users/",
										title = "Select a File",
										filetypes = (("Text files",
														"*.jpg*"),
													("all files",
														"*.*")))
	label_file_explorer.configure(text="File Opened: "+filename)
	inp = blur.get(1.0, "end-1c")
	print(filename,inp)
	noised_patches,denoised_patches,ground_truth,noisy_image,denoised_image=predict_fun(ridnet,filename,noise_level=int(inp))
	print('PSNR of Noisy Image : ',PSNR(ground_truth,noisy_image))
	print('PSNR of Denoised Image : ',PSNR(ground_truth,denoised_image))
	plot_patches(noised_patches,denoised_patches)
	plot_predictions(ground_truth,noisy_image,denoised_image)
											


l = Label(window)
l.place(x=0, y=0, relwidth=1, relheight=1)
l.bind('<Configure>', on_resize) 



label_file_explorer = Label(window,
							text = "This is tool for denoising your image or adding different levels of noise to you image",
							width = 100, height = 4,
							fg = "blue")

label = Label(window,
							text = "Enter the level Blur or noise : ",
							width = 60, height = 4,
							fg = "blue")


button_explore = Button(window,text = "Select Image", width=15, height=2, command = browseFiles)
button_exit = Button(window,
					text = "Exit", width=15, height=2, command = exit)

blur=Text(window,
          height = 3,width = 20)
blur.insert(INSERT, "0")



label_file_explorer.grid(column = 0, row = 1,pady=10)
label.grid(column = 0, row = 2,sticky="w",pady=10)
blur.grid(column = 0, row = 2,sticky="e",pady=10)
button_explore.grid(column = 0, row = 3,sticky="w",pady=10)
button_exit.grid(column = 0,row = 3,sticky="w",pady=10,padx=150)

window.mainloop()
