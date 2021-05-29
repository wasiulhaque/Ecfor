from django.shortcuts import redirect, render
from django.http import HttpResponse

from django.contrib.auth.decorators import login_required

# speech to text
import speech_recognition as sr

# text to speech
from gtts import gTTS
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
            text_file = open("speech_to_text", "wt")
            text_file.write(text)
            text_file.close()
        except:
            print("Couldn't recognize the voice")

    humanMessage = read_file()
    print(humanMessage)
    context = {'humanMessage': humanMessage}
    return render(request, 'index.html', context=context)


def read_file():
    file = open('speech_to_text', 'r')
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

    os.system("start output.mp3")

    return redirect('Communication:index')
