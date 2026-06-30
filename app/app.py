"""
Aplikasi Streamlit untuk prediksi Customer Churn.
Model terbaik dimuat dari folder model/.
Jalankan: streamlit run app/app.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

# ============================================================
# Konfigurasi halaman
# ============================================================
st.set_page_config(
    page_title='Customer Churn Prediction',
    page_icon='📊',
    layout='wide'
)

# ============================================================
# Styling CSS
# ============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #212121;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .result-churn {
        background-color: #EF5350;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }
    .result-no-churn {
        background-color: #1E88E5;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }
    .metric-card {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .section-title {
        color: #212121;
        font-weight: 600;
        font-size: 1.1rem;
        border-bottom: 2px solid #90CAF9;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Load model dan artefak
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')


@st.cache_resource
def load_artifacts():
    model           = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    scaler          = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    top_features    = joblib.load(os.path.join(MODEL_DIR, 'top_feature_names.pkl'))
    impute_values   = joblib.load(os.path.join(MODEL_DIR, 'impute_values.pkl'))
    return model, scaler, feature_columns, top_features, impute_values


def preprocess_input(data, scaler, feature_columns, top_feature_names):
    """
    Menerapkan preprocessing yang sama seperti di notebook:
    encoding -> align kolom -> scaling -> pilih top features.
    """
    df_in = pd.DataFrame([data])

    # Encoding gender
    gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
    df_in['gender'] = gender_map.get(df_in['gender'].values[0], 0)

    # Encoding subscription_type
    df_in['subscription_type'] = 0 if df_in['subscription_type'].values[0] == 'Annual' else 1

    # One-hot encoding device_type (reference: Desktop)
    device_val = df_in['device_type'].values[0]
    df_in['device_type_Mobile'] = 1 if device_val == 'Mobile' else 0
    df_in['device_type_Tablet'] = 1 if device_val == 'Tablet' else 0
    df_in.drop(columns=['device_type'], inplace=True)

    # One-hot encoding acquisition_channel (reference: Email)
    acq_val = df_in['acquisition_channel'].values[0]
    df_in['acquisition_channel_Facebook_Ads']  = 1 if acq_val == 'Facebook Ads' else 0
    df_in['acquisition_channel_Google_Ads']    = 1 if acq_val == 'Google Ads' else 0
    df_in['acquisition_channel_Organic']       = 1 if acq_val == 'Organic' else 0
    df_in['acquisition_channel_Referral']      = 1 if acq_val == 'Referral' else 0
    df_in.drop(columns=['acquisition_channel'], inplace=True)

    # One-hot encoding payment_method (reference: BKash)
    pay_val = df_in['payment_method'].values[0]
    df_in['payment_method_Card']   = 1 if pay_val == 'Card' else 0
    df_in['payment_method_PayPal'] = 1 if pay_val == 'PayPal' else 0
    df_in['payment_method_SEPA']   = 1 if pay_val == 'SEPA' else 0
    df_in['payment_method_UPI']    = 1 if pay_val == 'UPI' else 0
    df_in.drop(columns=['payment_method'], inplace=True)

    # Sesuaikan urutan kolom dengan training data
    for col in feature_columns:
        if col not in df_in.columns:
            df_in[col] = 0
    df_in = df_in[feature_columns]

    # Scaling
    df_scaled = scaler.transform(df_in)

    # Pilih top features
    top_idx = [feature_columns.index(f) for f in top_feature_names]
    return df_scaled[:, top_idx]


# ============================================================
# Load artefak
# ============================================================
try:
    model, scaler, feature_columns, top_feature_names, impute_values = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f'Gagal memuat model: {e}. Jalankan notebook terlebih dahulu.')

# ============================================================
# Header halaman
# ============================================================
st.markdown('<div class="main-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Prediksi apakah seorang pelanggan akan berhenti berlangganan (churn) berdasarkan data perilaku dan transaksi mereka.</div>', unsafe_allow_html=True)

st.markdown('---')

# ============================================================
# Layout utama
# ============================================================
col_input, col_result = st.columns([2, 1])

with col_input:
    st.markdown('<div class="section-title">Data Pelanggan</div>', unsafe_allow_html=True)

    # -- Tab untuk mengelompokkan input --
    tab1, tab2, tab3 = st.tabs(['Profil & Akun', 'Aktivitas & Transaksi', 'Layanan & Kepuasan'])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox('Jenis Kelamin', ['Male', 'Female', 'Other'])
            age    = st.slider('Usia', min_value=10, max_value=80, value=35)
            subscription_type = st.selectbox('Tipe Langganan', ['Annual', 'Monthly'])

        with c2:
            device_type         = st.selectbox('Tipe Perangkat', ['Desktop', 'Mobile', 'Tablet'])
            acquisition_channel = st.selectbox('Channel Akuisisi', ['Email', 'Facebook Ads', 'Google Ads', 'Organic', 'Referral'])
            payment_method      = st.selectbox('Metode Pembayaran', ['BKash', 'Card', 'PayPal', 'SEPA', 'UPI'])

        is_premium_user = st.checkbox('Pengguna Premium', value=False)
    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            total_visits     = st.slider('Total Kunjungan', 1, 30, 10)
            avg_session_time = st.slider('Rata-rata Waktu Sesi (menit)', 1.0, 25.0, 8.0, step=0.5)
            pages_per_session = st.slider('Halaman per Sesi', 1.0, 10.0, 4.0, step=0.5)
            email_open_rate  = st.slider('Email Open Rate', 0.0, 1.0, 0.5, step=0.01)
            email_click_rate = st.slider('Email Click Rate', 0.0, 1.0, 0.3, step=0.01)

        with c4:
            total_spent      = st.number_input('Total Pengeluaran (USD)', min_value=0.0, value=500.0, step=10.0)
            avg_order_value  = st.number_input('Rata-rata Nilai Transaksi (USD)', min_value=0.0, value=60.0, step=5.0)
            lifetime_value   = st.number_input('Lifetime Value (USD)', min_value=0.0, value=1000.0, step=50.0)
            marketing_spend  = st.number_input('Marketing Spend per User (USD)', min_value=0.0, value=15.0, step=1.0)

        discount_used = st.checkbox('Menggunakan Diskon', value=False)
    with tab3:
        c5, c6 = st.columns(2)
        with c5:
            support_tickets  = st.slider('Jumlah Tiket Support', 0, 10, 1)
            delivery_delay   = st.slider('Keterlambatan Pengiriman (hari)', 0, 10, 2)
            last_3m_freq     = st.slider('Frekuensi Pembelian 3 Bulan Terakhir', 0, 20, 5)

        with c6:
            satisfaction     = st.slider('Skor Kepuasan (1-5)', 1.0, 5.0, 3.5, step=0.5)
            nps_score        = st.slider('NPS Score', 0, 10, 7)
            refund_requested = st.checkbox('Pernah Minta Refund', value=False)

    st.markdown('')
    predict_btn = st.button('Prediksi Churn', use_container_width=True, type='primary')
    
    st.markdown('---')
    st.markdown('**Penjelasan Fitur Input:**')
    with st.expander('Profil & Akun'):
        st.dataframe(pd.DataFrame([
            ['Jenis Kelamin', 'Jenis kelamin pelanggan (Male, Female, Other)'],
            ['Usia', 'Usia pelanggan dalam tahun'],
            ['Tipe Langganan', 'Tipe langganan (Annual = tahunan, Monthly = bulanan)'],
            ['Tipe Perangkat', 'Perangkat utama yang digunakan'],
            ['Channel Akuisisi', 'Dari mana pelanggan pertama kali mendaftar'],
            ['Metode Pembayaran', 'Cara pelanggan melakukan pembayaran'],
            ['Pengguna Premium', 'Apakah pelanggan berlangganan paket premium']
        ], columns=['Fitur', 'Penjelasan']), use_container_width=True, hide_index=True)
        
    with st.expander('Aktivitas & Transaksi'):
        st.dataframe(pd.DataFrame([
            ['Total Kunjungan', 'Total kunjungan ke platform'],
            ['Rata-rata Waktu Sesi', 'Rata-rata durasi sesi penggunaan dalam menit'],
            ['Halaman per Sesi', 'Rata-rata halaman yang dikunjungi per sesi'],
            ['Email Open Rate', 'Persentase email promosi yang dibuka (0-1)'],
            ['Email Click Rate', 'Persentase klik pada email promosi (0-1)'],
            ['Total Pengeluaran', 'Total pengeluaran pelanggan dalam USD'],
            ['Rata-rata Nilai Transaksi', 'Rata-rata nilai per transaksi dalam USD'],
            ['Lifetime Value', 'Nilai total pelanggan selama berlangganan dalam USD'],
            ['Marketing Spend', 'Biaya pemasaran per pelanggan dalam USD'],
            ['Menggunakan Diskon', 'Apakah pelanggan pernah menggunakan diskon']
        ], columns=['Fitur', 'Penjelasan']), use_container_width=True, hide_index=True)
        
    with st.expander('Layanan & Kepuasan'):
        st.dataframe(pd.DataFrame([
            ['Jumlah Tiket Support', 'Jumlah tiket dukungan yang dibuat pelanggan'],
            ['Keterlambatan Pengiriman', 'Rata-rata keterlambatan pengiriman dalam hari'],
            ['Frekuensi Pembelian', 'Frekuensi pembelian dalam 3 bulan terakhir'],
            ['Skor Kepuasan', 'Skor kepuasan pelanggan (1-5)'],
            ['NPS Score', 'Net Promoter Score (0-10)'],
            ['Pernah Minta Refund', 'Apakah pelanggan pernah mengajukan refund']
        ], columns=['Fitur', 'Penjelasan']), use_container_width=True, hide_index=True)

# ============================================================
# Panel hasil prediksi
# ============================================================
with col_result:
    st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)

    if predict_btn and model_loaded:
        input_data = {
            'gender'                   : gender,
            'age'                      : age,
            'subscription_type'        : subscription_type,
            'is_premium_user'          : int(is_premium_user),
            'total_visits'             : total_visits,
            'avg_session_time'         : avg_session_time,
            'pages_per_session'        : pages_per_session,
            'email_open_rate'          : email_open_rate,
            'email_click_rate'         : email_click_rate,
            'total_spent'              : total_spent,
            'avg_order_value'          : avg_order_value,
            'discount_used'            : int(discount_used),
            'support_tickets'          : support_tickets,
            'refund_requested'         : int(refund_requested),
            'delivery_delay_days'      : delivery_delay,
            'satisfaction_score'       : satisfaction,
            'nps_score'                : nps_score,
            'marketing_spend_per_user' : marketing_spend,
            'lifetime_value'           : lifetime_value,
            'last_3_month_purchase_freq': last_3m_freq,
            'device_type'              : device_type,
            'acquisition_channel'      : acquisition_channel,
            'payment_method'           : payment_method,
        }

        try:
            X_input  = preprocess_input(input_data, scaler, feature_columns, top_feature_names)
            pred     = model.predict(X_input)[0]
            prob     = model.predict_proba(X_input)[0]
            prob_churn = prob[1] * 100

            if pred == 1:
                st.markdown('<div class="result-churn">Churn - Pelanggan Berisiko</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-no-churn">Tidak Churn - Pelanggan Aman</div>', unsafe_allow_html=True)

            st.markdown('')
            st.write(f'**Probabilitas Churn:** {prob_churn:.1f}%')
            st.progress(int(prob_churn))

            st.markdown('')
            st.markdown('**Detail Probabilitas:**')
            st.write(f'- Tidak Churn : {prob[0]*100:.1f}%')
            st.write(f'- Churn       : {prob[1]*100:.1f}%')

            if pred == 1:
                st.markdown('---')
                st.warning('Pelanggan ini memiliki risiko churn. Pertimbangkan untuk memberikan penawaran retensi atau menghubungi tim customer service.')
            else:
                st.markdown('---')
                st.success('Pelanggan ini cenderung loyal. Tetap jaga kualitas layanan untuk mempertahankan kepuasan.')

        except Exception as e:
            st.error(f'Error saat prediksi: {e}')

    elif not model_loaded:
        st.info('Model belum dimuat. Jalankan notebook terlebih dahulu untuk menghasilkan model.')
    else:
        st.info('Isi data pelanggan di panel kiri, lalu klik tombol "Prediksi Churn".')

    # Info model
    if model_loaded:
        st.markdown('---')
        st.markdown('<div class="section-title">Informasi Model</div>', unsafe_allow_html=True)
        st.write(f'**Tipe Model:** {type(model).__name__}')
        st.write(f'**Jumlah Fitur:** {len(top_feature_names)}')
        st.write('**Skenario:** Hyperparameter Tuning + Feature Selection')

