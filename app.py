import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title='ASL Identifier',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.header('American Sign Language Alphabet Identifier')
st.write("""
Kindly input your image for us to identify the alphabetical of American Sign Language
""")


#load model
model = load_model('modelvgg.h5')

def img_predict(img, model):
    #Image has to be an array
    img_array = np.array(img)

    #and atleast a 3D
    if img_array.shape[-1] >= 3:
        pred = img_array[:, :, :3]
    else:
        #Print ... if the image isnt RGB
        return "Gambar tidak memiliki tiga saluran warna (R, G, B)"

    pred = tf.image.resize(pred, size=(224, 224))
    pred = pred / 255.0

    predicted_probabilities = model.predict(x=tf.expand_dims(pred, axis=0))[0]

    predicted_class_index = np.argmax(predicted_probabilities)
    

    if predicted_class_index == 0:
        return "A"
    elif predicted_class_index == 1:
        return "B"
    elif predicted_class_index == 2:
        return "C"
    elif predicted_class_index == 3:
        return "D"
    elif predicted_class_index == 4:
        return "E"
    elif predicted_class_index == 5:
        return "F"
    elif predicted_class_index == 6:
        return "G"
    elif predicted_class_index == 7:
        return "H"
    elif predicted_class_index == 8:
        return "I"
    elif predicted_class_index == 9:
        return "J"
    elif predicted_class_index == 10:
        return "K"
    elif predicted_class_index == 11:
        return "L"
    elif predicted_class_index == 12:
        return "M"
    elif predicted_class_index == 13:
        return "N"
    elif predicted_class_index == 14:
        return "O"
    elif predicted_class_index == 15:
        return "P"
    elif predicted_class_index == 16:
        return "Q"
    elif predicted_class_index == 17:
        return "R"
    elif predicted_class_index == 18:
        return "S"
    elif predicted_class_index == 19:
        return "T"
    elif predicted_class_index == 20:
        return "U"
    elif predicted_class_index == 21:
        return "V"
    elif predicted_class_index == 22:
        return "W"
    elif predicted_class_index == 23:
        return "X"
    elif predicted_class_index == 24:
        return "Y"
    elif predicted_class_index == 25:
        return "Z"
    elif predicted_class_index == 26:
        return "del"
    elif predicted_class_index == 27:
        return "nothing"
    elif predicted_class_index == 28:
        return "N"
    else:
        return "space"

def run():
    # variable image
    img = None

    # Image upload and prediction
    uploaded_img = st.file_uploader("Place your image for translation", type=["jpg", "png", "jpeg"])

    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        prediction = img_predict(img, model)
        
        # Display the prediction result
        title = f"<h2 style='text-align:center'>{prediction}</h2>"
        st.markdown(title, unsafe_allow_html=True)
        st.image(img, use_column_width=True)

if __name__ == "__main__":
    run()