#This is the sample code of discrete hopfield network

import numpy as np
import random
from PIL import Image
import os
import re
from skimage.filters import threshold_otsu

#convert matrix to a vector
def mat2vec(x):
    return x.flatten()

#Create Weight matrix for a single image
def create_W(x):
    if len(x.shape) != 1:
        print("The input is not vector")
        return
    else:
        w = np.outer(x, x)
        np.fill_diagonal(w, 0)
    return w

#Read Image file and convert it to Numpy array with adaptive threshold
def readImg2array(file, size):
    pilIN = Image.open(file).convert(mode="L")
    pilIN = pilIN.resize(size)
    imgArray = np.asarray(pilIN, dtype=np.uint8)
    t = threshold_otsu(imgArray)
    x = np.where(imgArray > t, 1, -1)
    return x

#Convert Numpy array to Image file like Jpeg
def array2img(data, outFile = None):
    y = np.zeros(data.shape,dtype=np.uint8)
    y[data==1] = 255
    y[data==-1] = 0
    img = Image.fromarray(y,mode="L")
    if outFile is not None:
        img.save(outFile)
    return img

#Asynchronous update
def update(w,y_vec,theta=0.0,time=30000):
    m = len(y_vec)
    for _ in range(time):
        i = random.randint(0, m-1)
        u = np.dot(w[i][:], y_vec) - theta
        if u > 0:
            y_vec[i] = 1
        elif u < 0:
            y_vec[i] = -1
    return y_vec

#The following is training pipeline
def hopfield(train_files, test_files, theta=0.0, time=30000, size=(28,28), current_path=None):

    print ("Importing images and creating weight matrix....")

    num_files = 0
    for path in train_files:
        print (path)
        x = readImg2array(file=path, size=size)
        x_vec = mat2vec(x)
        if num_files == 0:
            w = create_W(x_vec)
        else:
            tmp_w = create_W(x_vec)
            w += tmp_w
        num_files += 1

    w = w / num_files  # Normalisasi bobot
    print ("Weight matrix is done!!")

    counter = 0
    for path in test_files:
        y = readImg2array(file=path, size=size)
        oshape = y.shape
        array2img(y).show()
        print ("Imported test data")

        y_vec = mat2vec(y)
        print ("Updating...")
        y_vec_after = update(w=w, y_vec=y_vec, theta=theta, time=time)
        y_vec_after = y_vec_after.reshape(oshape)
        if current_path is not None:
            outfile = os.path.join(current_path, f"after_{counter}.jpeg")
            array2img(y_vec_after, outFile=outfile)
        else:
            array2img(y_vec_after).show()
        counter += 1

#Main
current_path = os.getcwd()

train_paths = [os.path.join(current_path, "train_pics", f) for f in os.listdir(os.path.join(current_path, "train_pics")) if f.lower().endswith((".jpg", ".jpeg"))]
test_paths = [os.path.join(current_path, "test_pics", f) for f in os.listdir(os.path.join(current_path, "test_pics")) if f.lower().endswith((".jpg", ".jpeg"))]

hopfield(train_files=train_paths, test_files=test_paths, theta=0.0, time=30000, size=(28,28), current_path=current_path)