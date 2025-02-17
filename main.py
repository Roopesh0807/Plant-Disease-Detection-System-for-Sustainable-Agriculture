import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Dictionary of disease prevention methods
disease_prevention = {
    "Apple___Apple_scab": "**Prevention:** Prune trees to improve air circulation. Apply dormant sprays before bud break.\n"
                          "**Treatment:** Use fungicides like Mancozeb, Captan, or Myclobutanil. Remove infected leaves.",

    "Apple___Black_rot": "**Prevention:** Maintain orchard sanitation by removing fallen leaves. Avoid overhead irrigation.\n"
                         "**Treatment:** Apply copper-based sprays during early fruit development. Use fungicides like Thiophanate-methyl.",

    "Corn_(maize)___Common_rust_": "**Prevention:** Plant resistant maize hybrids. Ensure proper row spacing for good air circulation.\n"
                                    "**Treatment:** Apply fungicides such as Azoxystrobin and Triazole-based treatments.",

    "Grape___Black_rot": "**Prevention:** Prune excess foliage and remove diseased berries.\n"
                         "**Treatment:** Use Mancozeb, Ziram, or Captan sprays every 7-10 days during the growing season.",

    "Orange___Haunglongbing_(Citrus_greening)": "**Prevention:** Control psyllid insect vectors using Imidacloprid.\n"
                                                "**Treatment:** No cure available. Remove infected trees. Apply foliar micronutrients to slow disease progression.",

    "Potato___Early_blight": "**Prevention:** Rotate crops and avoid planting potatoes in the same area for at least two years.\n"
                             "**Treatment:** Apply chlorothalonil-based fungicides. Use biofungicides like Bacillus subtilis.",

    "Tomato___Bacterial_spot": "**Prevention:** Use certified disease-free seeds and avoid overhead irrigation.\n"
                                "**Treatment:** Apply copper-based bactericides weekly. Remove infected plant parts.",

    "Tomato___Late_blight": "**Prevention:** Increase spacing between plants and avoid excessive nitrogen fertilization.\n"
                             "**Treatment:** Use Chlorothalonil, Copper-based sprays, or Dithane M-45 fungicides.",

    "Tomato___healthy": "✅ **Your plant is healthy!** Continue proper watering, fertilization, and disease monitoring."
}

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display an image
img = Image.open("Diseases.png")
st.image(img)

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction:")
        result_index = model_prediction(test_image)
        
        # Disease classes
        class_name = list(disease_prevention.keys())
        disease_name = class_name[result_index]
        
        # Display Prediction
        st.success(f"Model is predicting it's {disease_name}")
        
        # Display Prevention Tips
        prevention = disease_prevention.get(disease_name, "No prevention tips available.")
        st.warning(f"Prevention & Treatment:\n{prevention}")
