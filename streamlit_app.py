import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Hopfield Visual UI")

# Fungsi convert image ke matrix biner sesuai hopfield.py
def img2binarray(img, size=(100, 100), threshold=60):
    img = img.convert("L").resize(size)
    arr = np.array(img)
    bin_arr = np.where(arr > threshold, 1, -1)
    return bin_arr.reshape(-1, 1)

def train_hopfield(patterns):
    n = patterns.shape[1]
    W = np.zeros((n, n))
    for p in patterns:
        p = p.reshape(-1, 1)
        W += np.dot(p, p.T)
    np.fill_diagonal(W, 0)
    W /= patterns.shape[0]  # Normalisasi bobot
    return W

def recall(W, pattern, steps=20000, theta=0.5):
    y = pattern.copy()
    n = len(y)
    for _ in range(steps):
        i = np.random.randint(0, n)
        u = np.dot(W[i], y) - theta
        y[i] = 1 if u > 0 else -1
    return y

def hamming(a, b):
    return np.sum(a != b)

# Sidebar untuk upload gambar latih dan uji
st.sidebar.title("Upload Data")
train_imgs = st.sidebar.file_uploader("Upload Gambar Latih", type=["jpg","jpeg","png"], accept_multiple_files=True)
test_img = st.sidebar.file_uploader("Upload Gambar Uji", type=["jpg","jpeg","png"])

st.title("Implementasi dan Visualisasi Jaringan Hopfield untuk Pengenalan Pola Gambar Biner")
st.markdown("Unggah gambar pelatihan dan satu gambar uji. Sistem akan melatih langsung dari gambar latih dan melakukan recall pada gambar uji.")

if train_imgs and test_img:
    pattern_list = []
    st.subheader("Gambar Latih")
    cols = st.columns(min(len(train_imgs), 4))
    for i, file in enumerate(train_imgs):
        img = Image.open(file)
        vec = img2binarray(img)
        pattern_list.append(vec.flatten())
        cols[i % len(cols)].image(img.resize((100, 100)), caption=file.name)

    pattern_matrix = np.array(pattern_list)
    W = train_hopfield(pattern_matrix)

    st.subheader("Cuplikan Matriks Bobot")
    W_df = np.round(W[:10, :10], 3)
    st.dataframe(W_df)

    st.subheader("Gambar Uji & Hasil Recall")
    test_image = Image.open(test_img)
    test_vec = img2binarray(test_image).flatten()
    recalled_vec = recall(W, test_vec)

    col1, col2 = st.columns(2)
    col1.image(test_image.resize((100, 100)), caption="Input Uji (Asli)")
    col2.image(Image.fromarray(np.uint8((recalled_vec.reshape(100, 100) + 1) * 127.5)), caption="Hasil Recall")

    st.subheader("Evaluasi terhadap Pola Latih")
    accs, hams, labels = [], [], []
    for i, p in enumerate(pattern_list):
        h = hamming(p, recalled_vec)
        acc = 1 - h / len(p)
        accs.append(acc)
        hams.append(h)
        labels.append(train_imgs[i].name)

    eval_table = {
        "Gambar Latih": labels,
        "Hamming Distance": hams,
        "Akurasi per Bit (%)": [round(a * 100, 2) for a in accs],
    }
    st.dataframe(eval_table)

    st.subheader("Grafik Akurasi")
    sorted_indices = np.argsort(accs)[::-1]
    sorted_accs = np.array(accs)[sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=sorted_accs, y=sorted_labels, ax=ax, palette="Blues_d")
    ax.set_xlabel("Akurasi per Bit")
    ax.set_xlim(0, 1)
    for i, v in enumerate(sorted_accs):
        ax.text(v + 0.01, i, f"{v:.2%}", color='black', va='center')
    st.pyplot(fig)

else:
    st.warning("Silakan unggah gambar latih dan satu gambar uji untuk menjalankan Hopfield.")
