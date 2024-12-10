from face import *
from facevideo import *
from liveface import *

def menu():
    print('press 1 for emotion detection in image\npress 2 for emotion detection in video\npress 3 for emotion detection in real time\n')
    a=int(input('enter choice'))
    if a==1:
        face()
    elif a==2:
        facevideo()
    elif a==3:
        liveface()
    else:
        print('invalid choice')
        menu()
menu()
