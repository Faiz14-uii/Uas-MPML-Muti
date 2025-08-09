import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Muat preprocessor, model, dan label encoder yang sudah disimpan
try:
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('best_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    st.error("File model (preprocessor.joblib, best_model.joblib, atau label_encoder.joblib) tidak ditemukan.")
    st.error("Pastikan Anda sudah menjalankan seluruh kode di Jupyter Notebook dari Stage 1 hingga 3 dan file-file tersebut ada di direktori yang sama dengan `app.py`.")
    st.stop()

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Prediksi Pemesanan Makanan Online", layout="centered", page_icon="🍜")

# --- Gaya CSS Kustom (Nuansa Pink) ---
st.markdown(
    """
    <style>
    /* Warna dasar pink */
    :root {
        --primary-color: #FF69B4; /* Hot Pink */
        --background-color: #FFF0F5; /* Lavender Blush */
        --secondary-background-color: #FFE4E1; /* Misty Rose, used for sidebar if active, but not in this layout */
        --text-color: #333333;
        --font: sans-serif;
    }

    /* Mengatur warna latar belakang utama dan teks */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: var(--font);
    }

    /* Header utama */
    .main-header {
        font-size: 4.5em; /* Ukuran font diperbesar agar lebih menonjol */
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 0.5em;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    /* Subheader */
    .subheader {
        font-size: 1.5em; /* Disesuaikan agar lebih jelas dan seimbang dengan header baru */
        color: var(--text-color);
        text-align: center;
        margin-bottom: 1.5em;
    }

    /* Label input */
    .stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label {
        font-weight: bold;
        color: var(--text-color);
        font-size: 1.1em; /* Sedikit lebih besar untuk keterbacaan */
    }

    /* Tombol Prediksi */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%; /* Tombol memenuhi lebar kolom */
        font-size: 1.2em; /* Ukuran font tombol */
    }
    .stButton>button:hover {
        background-color: #E0569A; /* Darker Pink on hover */
        transform: translateY(-2px);
    }

    /* Hasil Prediksi */
    .prediction-result {
        font-size: 2.2em; /* Disesuaikan agar lebih menonjol */
        font-weight: bold;
        text-align: center;
        margin-top: 1.5em;
        padding: 1em;
        border-radius: 10px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        animation: fadeIn 1s ease-out;
        border: 2px solid; /* Border untuk hasil */
    }
    .positive {
        color: #28a745; /* Hijau untuk 'Yes' */
        background-color: #e6ffe6;
        border-color: #28a745;
    }
    .negative {
        color: #dc3545; /* Merah untuk 'No' */
        background-color: #ffe6e6;
        border-color: #dc3545;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Konten Aplikasi Utama ---
st.markdown('<p class="main-header">Prediksi Pemesanan Makanan Online 🍜</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Aplikasi ini memprediksi apakah pelanggan akan memesan makanan secara online.</p>', unsafe_allow_html=True)

st.write("---")
st.subheader("Isi Data Pelanggan Baru:")
st.markdown("Isi data di bawah untuk mendapatkan prediksi:")

# Mengatur input dalam kolom untuk layout yang lebih seimbang
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Usia", min_value=18, max_value=60, value=25, help="Usia pelanggan.")
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"], help="Jenis kelamin pelanggan.")
    marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Prefer not to say"], help="Status pernikahan pelanggan.")
    occupation = st.selectbox("Pekerjaan", ["Student", "Employee", "Self Employed", "House wife"], help="Pekerjaan pelanggan.")

with col2:
    monthly_income = st.selectbox("Pendapatan Bulanan", ["No Income", "Below Rs.10000", "10001 to 25000", "25001 to 50000", "More than 50000"], help="Estimasi pendapatan bulanan pelanggan.")
    educational_qualifications = st.selectbox("Kualifikasi Pendidikan", ["School", "Graduate", "Post graduate", "Ph.D"], help="Tingkat pendidikan terakhir pelanggan.")
    family_size = st.number_input("Ukuran Keluarga", min_value=1, max_value=10, value=3, help="Jumlah anggota keluarga (termasuk pelanggan).")
    feedback = st.selectbox("Umpan Balik Sebelumnya", ["Positive", "Negative"], help="Umpan balik sebelumnya dari pelanggan (jika ada).")

# Buat DataFrame dari input pengguna.
# PENTING: Urutan kolom harus sama dengan X_train asli!
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Monthly Income': monthly_income,
    'Educational Qualifications': educational_qualifications,
    'Family size': family_size,
    'Feedback': feedback
}])

st.write("---")

# Tombol Prediksi di area utama untuk menyeimbangkan layout
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    if st.button("Prediksi Sekarang!"):
        with st.spinner('Memproses data dan membuat prediksi...'):
            try:
                # Lakukan Ordinal Encoding secara manual pada data input sebelum preprocessing
                monthly_income_mapping = {"No Income": 0, "Below Rs.10000": 1, "10001 to 25000": 2, "25001 to 50000": 3, "More than 50000": 4}
                edu_qual_mapping = {"School": 0, "Graduate": 1, "Post graduate": 2, "Ph.D": 3}
                
                input_data['Monthly Income'] = input_data['Monthly Income'].map(monthly_income_mapping)
                input_data['Educational Qualifications'] = input_data['Educational Qualifications'].map(edu_qual_mapping)

                # Lakukan preprocessing pada input pengguna menggunakan preprocessor yang sudah dilatih
                processed_input = preprocessor.transform(input_data)

                # Buat prediksi menggunakan model yang sudah dilatih
                prediction = model.predict(processed_input)
                prediction_proba = model.predict_proba(processed_input)

                # Mengembalikan prediksi ke label asli ('Yes'/'No') menggunakan LabelEncoder
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                
                st.write("---")
                if predicted_label == 'Yes':
                    st.markdown(
                        f'<div class="prediction-result positive">Pelanggan Cenderung Akan Memesan Makanan Online! 🎉<br>Probabilitas: **{prediction_proba[0][1]*100:.2f}%**</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-result negative">Pelanggan Cenderung Tidak Akan Memesan Makanan Online. 😔<br>Probabilitas: **{prediction_proba[0][0]*100:.2f}%**</div>',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membuat prediksi: {e}")
                st.info("Pastikan semua input data sudah benar dan file model telah dimuat dengan sukses.")

st.markdown("---")
st.markdown("Aplikasi dibuat oleh **Muthia Rahmatun Nisa'**")
st.markdown("Sumber data: [Kaggle Online Food Orders Dataset](https://www.kaggle.com/datasets/saurabhbagad/online-food-orders)")