import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model_suit_jawa.h5")

labels = ['gajah', 'manusia', 'semut']

def prediksi(img):
    img = img.convert("RGB")
    img = img.resize((150,150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    return labels[np.argmax(pred)]

def tentukan_pemenang(p1, p2):
    if p1 == p2:
        return f"Seri ({p1})"

    if (p1 == "gajah" and p2 == "manusia") or \
       (p1 == "manusia" and p2 == "semut") or \
       (p1 == "semut" and p2 == "gajah"):
        return f"Menang = {p1}"
    else:
        return f"Menang = {p2}"

st.title("🖐️ Game Suit Jawa AI")
st.write("Upload 2 gambar tangan:")

file1 = st.file_uploader("Pemain 1", type=["jpg", "jpeg", "png"])
file2 = st.file_uploader("Pemain 2", type=["jpg", "jpeg", "png"])

if file1 and file2:
    img1 = Image.open(file1)
    img2 = Image.open(file2)

    st.image([img1, img2], caption=["Pemain 1", "Pemain 2"], width=200)

    p1 = prediksi(img1)
    p2 = prediksi(img2)

    st.write("### Hasil Prediksi")
    st.write("Pemain 1:", p1)
    st.write("Pemain 2:", p2)

    hasil = tentukan_pemenang(p1, p2)
    st.success(hasil)