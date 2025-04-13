import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model

# Title of the app
st.title('Audio Detector')

# File uploader widget
audio_file = st.file_uploader('Upload an audio file', type=['wav'])

# Check if a file has been uploaded
if audio_file is not None:
    file_name = audio_file.name
    st.audio(audio_file, format=audio_file.type, loop=False)
   
    image_file_name = file_name.replace('.wav', '.png')
    os.makedirs("./img", exist_ok=True )
    image_file = os.path.join('./img', image_file_name)
    # print(type(file_name))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)


    # img processing
    # load img
    x = image.load_img(image_file, target_size=(224, 224, 3))

    # transformed_img = image.img_to_array(image.load_img(img, target_size=(224, 224, 3)))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model = joblib.load('./model/audioModel.pkl')
    bmodel = joblib.load('./model/base_model_audioModel.pkl')

    # with open('./model/audioModel.pkl', 'rb') as f:
    #     model = pickle.load(f)



    # model = load_model('./model/audioModel.pkl')
    y = bmodel.predict(x)
    predictions = model.predict(y)
    sound_dict = {}
    class_labels = ['background', 'chainsaw', 'engine', 'storm']
    for i, label in enumerate(class_labels):
        sound_dict[label] = predictions[0][i].item()
        print(f'{label}: {predictions[0][i]}')
        

    print("sound_dict", sound_dict)
    st.write("The prediction of the model for the given audio file based on the classes is" , sound_dict)
    predictedSound = max(sound_dict, key=sound_dict.get)
    st.write("The Model predicted the audio as the sound of" , predictedSound.upper())
    # for i, label in enumerate(class_labels):
    #     print(f'{label}: {predictions[0][i]}')

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
