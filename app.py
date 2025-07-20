import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Prediksi Peminatan IPA/IPS", layout="centered")

# === ISI UTAMA DIMULAI ===
st.title("📊 Prediksi peminatan SMA Al-Istiqomah")

@st.cache_resource
def load_model():
    return joblib.load("nilai_model.pkl")

model = load_model()
st.success("✅ Model Decision Tree berhasil dimuat.")

used_features = model.feature_names_in_

st.info(f"📌 Masukkan file excel yang berisikan data: {', '.join(used_features)}")
excel_file = st.file_uploader("Upload File Excel untuk Prediksi", type=["xlsx", "xls"])

if excel_file is not None:
    try:
        data = pd.read_excel(excel_file)

        st.subheader("📄 Data yang Diunggah")
        st.dataframe(data)

        try:
            _ = data[used_features]
        except KeyError as e:
            missing = list(set(used_features) - set(data.columns))
            st.error(f"❌ Kolom berikut tidak ditemukan di file: {', '.join(missing)}")
            st.stop()

        if st.button("🔮 Jalankan Prediksi"):
            # Mapping sikap ke angka
            mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            reverse_mapping = {v: k for k, v in mapping.items()}

            if 'sikap' in data.columns:
                data['sikap'] = data['sikap'].map(mapping)

            input_data = data[used_features]
            predictions = model.predict(input_data)

            # Mapping hasil prediksi ke IPA/IPS
            prediction_labels = {0: 'IPA', 1: 'IPS'}
            data['Prediction'] = [prediction_labels.get(p, 'Unknown') for p in predictions]

            # Balikkan kembali kolom sikap ke huruf
            if 'sikap' in data.columns:
                data['sikap'] = data['sikap'].map(reverse_mapping)

            st.session_state.predicted_data = data

            st.subheader("📈 Hasil Prediksi")
            st.dataframe(data)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")

# Tombol download muncul setelah prediksi
if 'predicted_data' in st.session_state:
    output_file = "hasil_prediksi.xlsx"
    st.session_state.predicted_data.to_excel(output_file, index=False)
    with open(output_file, 'rb') as f:
        st.download_button("📥 Download Hasil Prediksi", f, file_name=output_file, mime="text/csv")


