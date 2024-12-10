import cv2
from fer import FER
import numpy as np
import time
def liveface():
    emo = FER(mtcnn=True)
    facename = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'  #getting a haarcascade xml file
    facecas = cv2.CascadeClassifier()  #processing it for our project
    if not facecas.load(cv2.samples.findFile(facename)):  #adding a fallback event
        print("Error loading xml file")

    video = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not video.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        _, frame = video.read()
        ##    frame = cv2.flip(cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),1)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #changing the video to grayscale to make the face analisis work properly
        face=facecas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

        for x,y,w,h in face:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)  #making a recentangle to show up and detect the face and setting it position and colour
       
          #making a try and except condition in case of any errors
            try:
                
                cap_emo= emo.detect_emotions(frame)#same thing is happing here as the previous example, we are using the analyze class from deepface and using ‘frame’ as input
                dom_emo, emo_sc = emo.top_emotion(frame)
                print(dom_emo, emo_sc)#here we will only go print out the dominant emotion also explained in the previous example
                time.sleep(2)
            except:
                print("no face")

        cv2.imshow('Input', frame)

        c = cv2.waitKey(1)
        if c == ord('q'):
            break

    video.release() 
    cv2.destroyAllWindows()

