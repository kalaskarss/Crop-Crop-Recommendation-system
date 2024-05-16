import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import pickle

# Load the pre-trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Define the mapping of crop prediction
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Creating a simple form
st.title('Crop Recommendation System🌱')
st.image(r"C:\\Users\\LENOVO\Downloads\\crop.jpg")


st.markdown("""
<style>
input[type="text"] {
    width: 100%;
    height: 50px;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)


with st.form("prediction_form"):
    N = st.number_input('Nitrogen', min_value=0.0, format="%.2f")
    P = st.number_input('Phosphorus', min_value=0.0, format="%.2f")
    K = st.number_input('Potassium', min_value=0.0, format="%.2f")
    temp = st.number_input('Temperature', min_value=0.0, format="%.2f")
    humidity = st.number_input('Humidity', min_value=0.0, format="%.2f")
    ph = st.number_input('pH', min_value=0.0, max_value=14.0, format="%.2f")
    rainfall = st.number_input('Rainfall', min_value=0.0, format="%.2f")

    submit_button = st.form_submit_button("Predict")

if submit_button:
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale features
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    
    # Make prediction
    prediction = model.predict(final_features)

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    st.markdown(f'<h1 style="font-size:24px;">{result}</h1>', unsafe_allow_html=True)
