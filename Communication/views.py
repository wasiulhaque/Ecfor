from django.shortcuts import redirect, render
from django.http import HttpResponse

from django.contrib.auth.decorators import login_required
# import pygame
from datetime import datetime


import speech_recognition as sr

# text to speech
from gtts import gTTS

# playsound
from playsound import playsound

import json


import os

from pygame import mixer

import cv2
# from video_play import VideoPlayerPath

# Create your views here.


@login_required(login_url='Authentication:login')
def index(request):
    diction = {'sample_text': 'This is a text'}
    return render(request, 'index2.html', context=diction)


@login_required
def listen(request):

    r = sr.Recognizer()

    with sr.Microphone() as source:
        # r.adjust_for_ambient_noise(source)
        print("Speak Anything: ")
        audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            print("You said: {}".format(text))
            text_file = open("speech_to_txt", "wt")
            text_file.write(text)
            text_file.close()
        except:
            print("Couldn't recognize the voice")

    humanMessage = read_file()
    print(humanMessage)
    # print(type(humanMessage))

    # collecting video file id for the word
    mapping = make_video_dictionary()

    messageWords = humanMessage.split()

    video_id = "00335"
    default_path = str(os.getcwd())+'\\static\\dataset\\00335'+'.mp4'
    allVideos = []

    for word in messageWords:
        found = False
        if word in mapping:
            list_of_videos = mapping[word]

            for unit in list_of_videos:
                cwd = os.getcwd()
                # print(cwd)
                # print("current")
                path = cwd+'\\static\\dataset\\'+unit+'.mp4'
                # print(path)
                if(os.path.isfile(path)):
                    video_id = unit
                    allVideos.append(video_id)
                    found = True
                    break

        if not found:
            video_id = "00000"
            # print("check")
            allVideos.append(video_id)

    if not allVideos:
        allVideos.append(video_id)
    # combination of all the videos will be video_id
    print(allVideos)
    # video_process(allVideos)
    # video will contain id of a video which is the concat of all the videos, allVideos contains 1 set video of shob word

    allVideosJson = json.dumps(allVideos)
    context = {'humanMessage': humanMessage, 'videoList': allVideosJson}
    return render(request, 'index2.html', context=context)
    # return redirect('index.html')


def read_file():
    file = open('speech_to_txt', 'r')
    file_content = file.read()
    file.close()
    # file_content = "BD"
    return file_content


@login_required
def speak(request):

    # will read the file to convert into audio
    text_File = open("text_to_speech.txt", "r")
    myText = text_File.read().replace("\n", " ")
    # line ending er jaygay jate shesh na vabe instead break nibe bolar time a

    language = 'en'

    # slow means audio will played slow
    date_string = datetime.now().strftime("%d%m%Y%H%M%S")
    audiofilename = "voice"+date_string+".mp3"

    output = gTTS(text=myText, lang=language, slow=False)

    output.save(audiofilename)
    # output.save("output.mp3")

    text_File.close()

    # os.system("start output.mp3")     #stop using media player
    # playsound('output.mp3', True)
    # deleting the audio file to avoid audio overwrite permission problem
    playsound(audiofilename)
    os.remove(audiofilename)

    return redirect('Communication:index')


def text_to_sign():
    pass


def make_video_dictionary():

    file = open('dataset.json')

    load = json.load(file)

    mapping = {}

    for serial in range(len(load)):
        unit_data = load[serial]
        for key in unit_data:
            if(key == 'gloss'):
                word = unit_data[key]
                # print(unit_data[key])
            if(key == 'instances'):
                video_list = []
                for number in range(len(unit_data[key])):
                    video_id = unit_data[key][number]['video_id']
                    video_list.append(video_id)
                # print(unit_data[key][0]['video_id'])
                mapping[word] = video_list

    # for key, value in mapping.items():
    #     print("{} -> {}".format(key, value))
    return mapping


def video_process(video_path_list):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('merged_drum.avi', fourcc, 25.0, (1920, 1080))
    # print(video_path_list)
    for path in video_path_list:
        cap = cv2.VideoCapture(path)
        # frameToStart = 100
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow('frame', frame)
            ch = 0xFF & cv2.waitKey(15)
            out.write(frame)
            # if ch == 27:
            #     break
            # cap.release () # Turn off the camera
    out.release()
