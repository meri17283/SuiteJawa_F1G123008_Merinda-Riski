import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model_suit_jawa.h5")

labels = ["gajah", "manusia", "semut"]

def prediksi_gambar(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    hasil = labels[np.argmax(pred)]
    confidence = np.max(pred) * 100

    return hasil, confidence

def tentukan_pemenang(p1, p2):
    if p1 == p2:
        return f"Seri ({p1})"

    if (
        (p1 == "gajah" and p2 == "manusia") or
        (p1 == "manusia" and p2 == "semut") or
        (p1 == "semut" and p2 == "gajah")
    ):
        return f"Pemain 1 menang: {p1}"
    else:
        return f"Pemain 2 menang: {p2}"

st.title("Suit Jawa AI")
st.write("Upload gambar untuk Pemain 1 dan Pemain 2.")

file1 = st.file_uploader("Upload gambar Pemain 1", type=["jpg", "jpeg", "png"])
file2 = st.file_uploader("Upload gambar Pemain 2", type=["jpg", "jpeg", "png"])

if file1 and file2:
    img1 = Image.open(file1).convert("RGB")
    img2 = Image.open(file2).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img1, caption="Pemain 1", use_container_width=True)

    with col2:
        st.image(img2, caption="Pemain 2", use_container_width=True)

    if st.button("Prediksi"):
        p1, conf1 = prediksi_gambar(img1)
        p2, conf2 = prediksi_gambar(img2)

        st.write(f"Pemain 1: **{p1}** ({conf1:.2f}%)")
        st.write(f"Pemain 2: **{p2}** ({conf2:.2f}%)")

        hasil = tentukan_pemenang(p1, p2)
        st.success(hasil)