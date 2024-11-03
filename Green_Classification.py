import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import streamlit as st
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
import numpy as np
import streamlit as st

st.header("🍄 Green Classifier Model 🍄")

model = load_model("imageClassify.keras")

category = [
    "apple",
    "banana",
    "beetroot",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "grapes",
    "jalepeno",
    "kiwi",
    "lemon",
    "lettuce",
    "mango",
    "onion",
    "orange",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "soy beans",
    "sweetcorn",
    "sweetpotato",
    "tomato",
]

height_i = 180
width_i = 180

capture = st.toggle("Use Camera")

if capture:
    enable = st.checkbox("Enable Camera")
    image = st.camera_input("Take a Picture", disabled=not enable)  # option 1
else:
    image = st.file_uploader(
        "Choose an image...", type=["jpg", "png", "jpeg"]
    )  # option 2

# image = st.text_input("Enter for response","corn.jpg")   # option 3

try:
    if image is None:
        st.text("Take/select a Picture first!")
    else:
        image = tf.keras.utils.load_img(image, target_size=(height_i, width_i))     #loads img in PIL format
        img_arr = tf.keras.utils.array_to_img(image)    #convert PIL(img) to numpy array
        img_bat = tf.expand_dims(img_arr, 0)    #expands dimensions and returns the same datainput with additional dims

        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict)

        st.image(image, width=350)
        st.write(
            " The veggie / fruit in image is "
            + category[np.argmax(score)]
            + " with accuracy of "
            + str((np.max(score) * 100))
            + "%"
        )

except Exception as e:
    st.write("Error in Deployment!! Get a look into your program ", e)
