from fer import Video
from fer import FER
import os
import sys
import pandas as pd
def facevideo():
    #the location of the video file that has to be processed
    videopath = r"C:\Users\HP\Desktop\happy-clap.gif"
    # Build the Face detection object
    face= FER(mtcnn=True)
    # Input the video for processing
    input_video = Video(videopath)

    # The Analyze() function will run analysis on every frame of the input video. 
    # It will create a rectangular box around every image and show the emotion values next to that.
    # Finally, the method will publish a new video that will have a box around the face of the human with live emotion values.
    data = input_video.analyze(face, display=False)

    # We will now convert the analysed information into a dataframe.
    # This will help us import the data as a .CSV file to perform analysis over it later
    vid = input_video.to_pandas(data)
    vid = input_video.get_first_face(vid)
    vid = input_video.get_emotions(vid)

    # Plotting the emotions against time in the video
    pltfig = vid.plot(figsize=(20, 8), fontsize=16).get_figure()

    # We will now work on the dataframe to extract which emotion was prominent in the video
    angry = sum(vid.angry)
    disgust = sum(vid.disgust)
    fear = sum(vid.fear)
    happy = sum(vid.happy)
    sad = sum(vid.sad)
    surprise = sum(vid.surprise)
    neutral = sum(vid.neutral)

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emoval = [angry, disgust, fear, happy, sad, surprise, neutral]

    score = pd.DataFrame(emotions, columns = ['Human Emotions'])
    score['Emotion Value from the Video'] = emoval
    print(score)


