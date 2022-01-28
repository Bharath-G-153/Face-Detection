
# Face Detection using Mediapipe API and Haar Cascade Classifier

## About the Project
The definition of face detection refers to computer technology that is able to identify the presence of people’s faces within digital images. In order to work, face detection applications use machine learning and formulas known as algorithms to detecting human faces within larger images. These larger images might contain numerous objects that aren’t faces such as landscapes, buildings and other parts of humans.

Face detection is a broader term than face recognition. Face detection just means that a system is able to identify that there is a human face present in an image or video. Face detection has several applications, only one of which is facial recognition. Face detection can also be used to auto focus cameras. And it can be used to count how many people have entered a particular area. It can even be used for marketing purposes. For example, advertisements can be displayed the moment a face is recognized.

### How does it work?
While the process is somewhat complex, face detection algorithms often begin by searching for human eyes. Eyes constitute what is known as a valley region and are one of the easiest features to detect. Once eyes are detected, the algorithm might then attempt to detect facial regions including eyebrows, the mouth, nose, nostrils and the iris. Once the algorithm surmises that it has detected a facial region, it can then apply additional tests to validate whether it has, in fact, detected a face.
Here we are using the MediaPipe API which is a RESTful API used for Face recognition and detection applications.




## Mediapipe
MediaPipe Face Detection is an ultrafast face detection solution that comes with 6 landmarks and multi-face support. It is based on BlazeFace, a lightweight and well-performing face detector tailored for mobile GPU inference. The detector’s super-realtime performance enables it to be applied to any live viewfinder experience that requires an accurate facial region of interest as an input for other task-specific models, such as 3D facial keypoint or geometry estimation (e.g., MediaPipe Face Mesh), facial features or expression classification, and face region segmentation. BlazeFace uses a lightweight feature extraction network inspired by, but distinct from MobileNetV1/V2, a GPU-friendly anchor scheme modified from Single Shot MultiBox Detector (SSD), and an improved tie resolution strategy alternative to non-maximum suppression. For more information about BlazeFace, please see the Resources section

## Haar Cascade Classifier
A Haar classifier, or a Haar cascade classifier, is a machine learning object detection program that identifies objects in an image and video.It’s important to remember that this algorithm requires a lot of positive images of faces and negative images of non-faces to train the classifier, similar to other machine learning models.

The first step is to collect the Haar features. A Haar feature is essentially calculations that are performed on adjacent rectangular regions at a specific location in a detection window. The calculation involves summing the pixel intensities in each region and calculating the differences between the sums. Here are some examples of Haar features below.These features can be difficult to determine for a large image. This is where integral images come into play because the number of operations is reduced using the integral image.
## Getting Started

This is an example of how you may give instructions on setting up your project locally. To get a local copy up and running follow these simple example steps.

### Pre-requisites and installation

First step is to install all the required libraries
```bash
!pip install mediapipe
!pip install cartopy
!pip install face_recognition
!pip install dlib
!pip install RealSense2
```
Importing all the libraries
```bash
import cv2
from google.colab.patches import cv2_imshow
import math
import numpy as np
import os
import mediapipe as mp
from IPython.display import clear_output
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time
```
### Usage
 Face recognition describes a biometric technology that goes way beyond recognizing when a human face is present. It actually attempts to establish whose face it is. The process works using a computer application that captures a digital image of an individual’s face (sometimes taken from a video frame) and compares it to images in a database of stored records. While facial recognition isn’t 100% accurate, it can very accurately determine when there is a strong chance that an person’s face matches someone in the database
### Importing all the images required for Face Detection
```bash
from google.colab import files
uploaded_short_range = files.upload()

```
### Adding the Haar Cascade Detector file directory

```bash
face_cascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
```
### Resizing images to required number of pixels

OpenCV does the hard work of finding shapes that resemble a human face and returning the coordinates. It includes a python version but there are other libraries that wrap it up and make it a bit easier to work with.

Since we don’t have to do the hard work of finding faces why don’t we just get really big images and automate the process of making smaller ones!
```bash
def resize_and_show(input_image,height,width):
'''
input = input_image,required_height,required_width
output = resized_image
'''
  return cv2.resize(input_img,(height,width), interpolation= cv2.INTER_LINEAR)
```

### Detection of images from short range images and long range images

The accuracy and the bounding box drawn from the detection from the video feed based on the distance of the object placed from the camera and scale and resizing of the following can be termed and confirmed.

### Javascript dependencies and functions to convert the object into an OpenCV image
Reading image files in b64 and decoding the images and also using pillow to read the images


## Acknowledgements

 - [Haar Cascade Classifier](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Mediapipe](https://github.com/matiassingers/awesome-readme)
 - [Face Detection and Face Recognition](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)

