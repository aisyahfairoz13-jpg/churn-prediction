import streamlit as st
import joblib
import numpy as np

model = joblib.load('churn_model.pkl')

st.title('🔮 Telco Customer Churn Prediction')
st.write('**Model:** Random Forest | **Accuracy:** 79%')
st.markdown('---')

st.header('Input Data Pelanggan')

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', [0, 1], format_func=lambda x: ['Female', 'Male'][x])
    senior = st.selectbox('Senior Citizen', [0, 1], format_func=lambda x: ['No', 'Yes'][x])
    partner = st.selectbox('Partner', [0, 1], format_func=lambda x: ['No', 'Yes'][x])
    dependents = st.selectbox('Dependents', [0, 1], format_func=lambda x: ['No', 'Yes'][x])
    tenure = st.slider('Tenure (bulan)', 0, 72, 12)
    phone_service = st.selectbox('Phone Service', [0, 1], format_func=lambda x: ['No', 'Yes'][x])
    multiple_lines = st.selectbox('Multiple Lines', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No phone'][x])
    internet_service = st.selectbox('Internet Service', [0, 1, 2], format_func=lambda x: ['DSL', 'Fiber optic', 'No'][x])
    online_security = st.selectbox('Online Security', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No internet'][x])
    online_backup = st.selectbox('Online Backup', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No internet'][x])

with col2:
    device_protection = st.selectbox('Device Protection', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No internet'][x])
    tech_support = st.selectbox('Tech Support', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No internet'][x])
    streaming_tv = st.selectbox('Streaming TV', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No internet'][x])
    streaming_movies = st.selectbox('Streaming Movies', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No internet'][x])
    contract = st.selectbox('Contract', [0, 1, 2], format_func=lambda x: ['Month-to-month', 'One year', 'Two year'][x])
    paperless = st.selectbox('Paperless Billing', [0, 1], format_func=lambda x: ['No', 'Yes'][x])
    payment = st.selectbox('Payment Method', [0, 1, 2, 3], format_func=lambda x: ['Bank transfer', 'Credit card', 'Electronic check', 'Mailed check'][x])
    monthly_charges = st.number_input('Monthly Charges ($)', 0.0, 150.0, 50.0)
    total_charges = st.number_input('Total Charges ($)', 0.0, 10000.0, 500.0)

avg_monthly_spend = total_charges / (tenure + 1)

st.markdown('---')

if st.button('🔍 Prediksi Churn', use_container_width=True):
    input_data = np.array([[
        gender, senior, partner, dependents, tenure,
        phone_service, multiple_lines, internet_service,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
        contract, paperless, payment,
        monthly_charges, total_charges, avg_monthly_spend
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f'⚠️ Pelanggan kemungkinan CHURN\nProbabilitas: {probability[1]*100:.1f}%')
    else:
        st.success(f'✅ Pelanggan kemungkinan TIDAK CHURN\nProbabilitas: {probability[0]*100:.1f}%')
