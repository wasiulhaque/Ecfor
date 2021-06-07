from django.shortcuts import redirect, render
from django.http import HttpResponse

from django.contrib.auth.decorators import login_required

import speech_recognition as sr

# text to speech
from gtts import gTTS

# playsound
from playsound import playsound

import json

import os


# Create your views here.


@login_required(login_url='Authentication:login')
def index(request):
    diction = {'sample_text': 'This is a text'}
    return render(request, 'index.html', context=diction)


@login_required
def listen(request):

    r = sr.Recognizer()

    with sr.Microphone() as source:
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
    list_of_videos = mapping[humanMessage]

    video_id = "00335"
    for unit in list_of_videos:
        cwd = os.getcwd()
        # print(cwd)
        # print("current")
        path = cwd+'\\static\\dataset\\'+unit+'.mp4'
        # print(path)
        if(os.path.isfile(path)):
            video_id = unit
            break

    print(video_id)

    context = {'humanMessage': humanMessage, 'video_id': video_id}
    return render(request, 'index.html', context=context)


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
    output = gTTS(text=myText, lang=language, slow=False)

    output.save("output.mp3")

    text_File.close()

    # os.system("start output.mp3")     #stop using media player
    playsound('output.mp3')

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
