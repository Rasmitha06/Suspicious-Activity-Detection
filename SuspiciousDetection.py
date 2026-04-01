from tkinter import *
import tkinter
from tkinter.filedialog import askopenfilename
import cv2
import shutil
import os
import numpy as np
from imutils import paths
import json
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import resnet50

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

MODEL_PATH = "model.h5"
JSON_PATH  = "model_class.json"

with open(JSON_PATH, "r") as f:
    class_data = json.load(f)
idx_to_label = {int(k): v for k, v in class_data.items()}
num_classes  = len(idx_to_label)

def build_resnet50_imageai(num_classes):
    base   = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
    x      = base.output
    x      = layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
    x      = layers.Dense(num_classes, name="dense")(x)
    output = layers.Activation("softmax", name="activation_49")(x)
    model  = Model(inputs=base.input, outputs=output)
    return model

model = build_resnet50_imageai(num_classes)
model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)

def preprocess_frame(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x   = tf.keras.preprocessing.image.img_to_array(img)
    x   = np.expand_dims(x, axis=0)
    x   = resnet50.preprocess_input(x)
    return x

main = tkinter.Tk()
main.title("Suspicious Activity Detection")
main.geometry("1200x1200")

global filename
filename = ""

def upload():
    global filename
    filename = askopenfilename(initialdir="videos")
    pathlabel.config(text=filename)

def generateFrame():
    global filename
    text.delete('1.0', END)
    if not filename:
        pathlabel.config(text="Please upload a video first!")
        return
    if not os.path.exists('frames'):
        os.mkdir('frames')
    else:
        shutil.rmtree('frames')
        os.mkdir('frames')
    vidObj = cv2.VideoCapture(filename)
    count  = 0
    while True:
        success, image = vidObj.read()
        if not success or image is None:
            break
        if count < 500:
            cv2.imwrite("frames/frame%d.jpg" % count, image)
            text.insert(END, "frames/frame." + str(count) + " saved\n")
            text.update_idletasks()
        else:
            break
        count += 1
    vidObj.release()
    pathlabel.config(text="Frame generation process completed. All frames saved inside frame folder")

def detectActivity():
    if not os.path.exists('frames'):
        text1.insert(END, "Please generate frames first.\n")
        return
    imagePaths = sorted(list(paths.list_images("frames")))
    count  = 0
    option = 0
    text1.delete('1.0', END)
    for imagePath in imagePaths:
        try:
            x        = preprocess_frame(imagePath)
            preds    = model.predict(x, verbose=0)[0]
            top_idx  = int(np.argmax(preds))
            top_prob = float(preds[top_idx]) * 100
            label    = idx_to_label.get(top_idx, str(top_idx))
            if float(top_prob) > 80:
                count = count + 1
            if float(top_prob) < 80:
                count = 0
            if count > 10:
                option = 1
                print(imagePath + " is predicted as " + label + " with probability : " + str(top_prob))
                text1.insert(END, imagePath + " is predicted as " + label + " with probability : " + str(top_prob) + "\n\n")
                count = 0
        except Exception as e:
            print(f"Error on {imagePath}: {e}")
        print(imagePath + " processed")
    if option == 0:
        text1.insert(END, "No suspicious activity found in given footage")

font = ('times', 20, 'bold')
title = Label(main, text='Suspicious Activity Detection From CCTV Footage')
title.config(bg='brown', fg='white')
title.config(font=font)
title.config(height=3, width=80)
title.place(x=5, y=5)

font1 = ('times', 14, 'bold')
uploadBtn = Button(main, text="Upload CCTV Footage", command=upload)
uploadBtn.place(x=50, y=100)
uploadBtn.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=300, y=100)

depthbutton = Button(main, text="Generate Frames", command=generateFrame)
depthbutton.place(x=50, y=150)
depthbutton.config(font=font1)

userinterest = Button(main, text="Detect Suspicious Activity Frame", command=detectActivity)
userinterest.place(x=280, y=150)
userinterest.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=25, width=50)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=200)
text.config(font=font1)

text1 = Text(main, height=25, width=50)
scroll1 = Scrollbar(text1)
text1.configure(yscrollcommand=scroll1.set)
text1.place(x=550, y=200)
text1.config(font=font1)

main.config(bg='brown')
main.mainloop()