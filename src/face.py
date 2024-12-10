from fer import FER
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
def face():
    #sample image
##    image1= plt.imread(r"C:\Users\HP\Desktop\sad.jfif")
    image1= plt.imread(r"C:\Users\HP\Desktop\happy.jfif")
    #initializing FER object 
    emo = FER(mtcnn=True)#mtcnn is an auto trained facial expression recognition model

    #to capture emotions from the image
    cap_emo= emo.detect_emotions(image1)
    print(cap_emo)

    #to display dominant emotion and emotion score
    dom_emo, emo_sc = emo.top_emotion(image1)
    print(dom_emo, emo_sc)

    #to display image using pyplot
    plt.imshow(image1)
    plt.show()

