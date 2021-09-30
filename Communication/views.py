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


# Sign Language imports
# run
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


# run
import os
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# run
import cv2 
import numpy as np






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


# run
WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'


def ML(request):  


    # run
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    # run
    category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')



    # run
    # Setup capture
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



    # run
    while True: 
        ret, frame = cap.read()
        image_np = np.array(frame)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        pred = get_prediction(input_tensor)
        print(pred)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.5,
                    agnostic_mode=False)

        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break


@tf.function
def get_prediction(image):
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
#     detections = detection_model.postprocess(prediction_dict, shapes)
    return prediction_dict