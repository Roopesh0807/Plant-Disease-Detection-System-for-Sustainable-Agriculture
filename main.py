import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    img = Image.open("Diseases.png")
    st.image(img, use_container_width=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, use_container_width=True)
    
    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        disease = class_name[result_index]
        st.success(f"Model is Predicting it's a {disease}")

        # Treatment/Solutions for all diseases
        treatments = {
            'Apple___Apple_scab': {
                'description': 'Apple scab is a fungal disease that causes dark, scaly lesions on leaves and fruit.',
                'treatment': [
                    'Apply fungicides like captan or sulfur.',
                    'Prune infected branches and dispose of them properly.',
                    'Ensure good air circulation by spacing trees appropriately.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/apple-scab',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/apple-scab/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=captan+fungicide'  # Amazon link
            },
            'Apple___Black_rot': {
                'description': 'Black rot is a fungal disease causing fruit rot and leaf spots on apple trees.',
                'treatment': [
                    'Remove and destroy infected fruit and leaves.',
                    'Apply fungicides like myclobutanil or thiophanate-methyl.',
                    'Practice good sanitation by cleaning up fallen debris.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/black-rot-apple',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/black-rot/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=myclobutanil+fungicide'  # Amazon link
            },
            'Apple___Cedar_apple_rust': {
                'description': 'Cedar apple rust is a fungal disease that affects apples and cedars.',
                'treatment': [
                    'Remove nearby cedar trees if possible.',
                    'Apply fungicides like myclobutanil or triadimefon.',
                    'Plant resistant apple varieties.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/cedar-apple-rust',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/cedar-apple-rust/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=triadimefon+fungicide'  # Amazon link
            },
            'Apple___healthy': {
                'description': 'Your apple plant is healthy! No treatment is needed.',
                'treatment': [],
                'links': [],
                'pesticide_image': '',
                'buy_link': ''
            },
            'Blueberry___healthy': {
                'description': 'Your blueberry plant is healthy! No treatment is needed.',
                'treatment': [],
                'links': [],
                'pesticide_image': '',
                'buy_link': ''
            },
            'Cherry_(including_sour)___Powdery_mildew': {
                'description': 'Powdery mildew is a fungal disease that appears as white powdery spots on leaves and fruit.',
                'treatment': [
                    'Apply fungicides like sulfur or potassium bicarbonate.',
                    'Prune infected branches to improve air circulation.',
                    'Avoid overhead watering to reduce humidity.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/powdery-mildew',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/powdery-mildew/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=sulfur+fungicide'  # Amazon link
            },
            'Cherry_(including_sour)___healthy': {
                'description': 'Your cherry plant is healthy! No treatment is needed.',
                'treatment': [],
                'links': [],
                'pesticide_image': '',
                'buy_link': ''
            },
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
                'description': 'Cercospora leaf spot and gray leaf spot are fungal diseases that cause leaf spots on corn.',
                'treatment': [
                    'Apply fungicides like chlorothalonil or azoxystrobin.',
                    'Rotate crops to prevent disease buildup in the soil.',
                    'Remove and destroy infected plant debris.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/corn-leaf-spots',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/corn-leaf-spots/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=chlorothalonil+fungicide'  # Amazon link
            },
            'Corn_(maize)___Common_rust_': {
                'description': 'Common rust is a fungal disease that causes reddish-brown pustules on corn leaves.',
                'treatment': [
                    'Apply fungicides like mancozeb or propiconazole.',
                    'Plant resistant corn varieties.',
                    'Avoid planting corn in the same area repeatedly.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/common-rust-corn',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/common-rust/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=mancozeb+fungicide'  # Amazon link
            },
            'Corn_(maize)___Northern_Leaf_Blight': {
                'description': 'Northern leaf blight is a fungal disease that causes long, grayish-green lesions on corn leaves.',
                'treatment': [
                    'Apply fungicides like chlorothalonil or pyraclostrobin.',
                    'Rotate crops to prevent disease buildup in the soil.',
                    'Remove and destroy infected plant debris.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/northern-leaf-blight-corn',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/northern-leaf-blight/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=chlorothalonil+fungicide'  # Amazon link
            },
            'Corn_(maize)___healthy': {
                'description': 'Your corn plant is healthy! No treatment is needed.',
                'treatment': [],
                'links': [],
                'pesticide_image': '',
                'buy_link': ''
            },
            'Grape___Black_rot': {
                'description': 'Black rot is a fungal disease that causes dark, sunken lesions on grape leaves and fruit.',
                'treatment': [
                    'Apply fungicides like mancozeb or myclobutanil.',
                    'Prune infected branches and dispose of them properly.',
                    'Ensure good air circulation by spacing vines appropriately.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/black-rot-grape',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/black-rot/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=mancozeb+fungicide'  # Amazon link
            },
            'Grape___Esca_(Black_Measles)': {
                'description': 'Esca (Black Measles) is a fungal disease that causes dark spots and streaks on grape leaves and fruit.',
                'treatment': [
                    'Prune infected branches and dispose of them properly.',
                    'Apply fungicides like thiophanate-methyl or tebuconazole.',
                    'Ensure good air circulation by spacing vines appropriately.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/esca-grape',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/esca/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=tebuconazole+fungicide'  # Amazon link
            },
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
                'description': 'Leaf blight (Isariopsis Leaf Spot) is a fungal disease that causes brown spots on grape leaves.',
                'treatment': [
                    'Apply fungicides like mancozeb or chlorothalonil.',
                    'Prune infected branches and dispose of them properly.',
                    'Ensure good air circulation by spacing vines appropriately.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/grape-leaf-blight',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/grape-leaf-blight/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=mancozeb+fungicide'  # Amazon link
            },
            'Grape___healthy': {
                'description': 'Your grape plant is healthy! No treatment is needed.',
                'treatment': [],
                'links': [],
                'pesticide_image': '',
                'buy_link': ''
            },
            'Orange___Haunglongbing_(Citrus_greening)': {
                'description': 'Citrus greening (Haunglongbing) is a bacterial disease that causes yellowing of citrus leaves and stunted growth.',
                'treatment': [
                    'Remove and destroy infected trees to prevent spread.',
                    'Control psyllid insects (the disease vector) using insecticides.',
                    'Plant disease-free citrus trees.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/citrus-greening',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/citrus-greening/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=citrus+greening+insecticide'  # Amazon link
            },
            'Peach___Bacterial_spot': {
                'description': 'Bacterial spot is a bacterial disease that causes dark spots on peach leaves and fruit.',
                'treatment': [
                    'Apply copper-based bactericides.',
                    'Prune infected branches and dispose of them properly.',
                    'Avoid overhead watering to reduce spread.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/bacterial-spot-peach',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/bacterial-spot/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=copper+bactericide'  # Amazon link
            },
            'Peach___healthy': {
                'description': 'Your peach plant is healthy! No treatment is needed.',
                'treatment': [],
                'links': [],
                'pesticide_image': '',
                'buy_link': ''
            },
            'Pepper,_bell___Bacterial_spot': {
                'description': 'Bacterial spot is a bacterial disease that causes dark spots on bell pepper leaves and fruit.',
                'treatment': [
                    'Apply copper-based bactericides.',
                    'Remove and destroy infected plants.',
                    'Avoid overhead watering to reduce spread.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/bacterial-spot-pepper',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/bacterial-spot/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=copper+bactericide'  # Amazon link
            },
            'Pepper,_bell___healthy': {
                'description': 'Your bell pepper plant is healthy! No treatment is needed.',
                'treatment': [],
                'links': [],
                'pesticide_image': '',
                'buy_link': ''
            },
            'Potato___Early_blight': {
                'description': 'Early blight is a fungal disease that causes dark, concentric spots on potato leaves.',
                'treatment': [
                    'Apply fungicides like chlorothalonil or mancozeb.',
                    'Rotate crops to prevent disease buildup in the soil.',
                    'Remove and destroy infected plant debris.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/early-blight-potato',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/early-blight/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=chlorothalonil+fungicide'  # Amazon link
            },
            'Potato___Late_blight': {
                'description': 'Late blight is a fungal disease that causes dark, water-soaked lesions on potato leaves and tubers.',
                'treatment': [
                    'Apply fungicides like chlorothalonil or mancozeb.',
                    'Remove and destroy infected plants immediately.',
                    'Avoid planting potatoes in the same area repeatedly.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/late-blight-potato',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/late-blight/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=chlorothalonil+fungicide'  # Amazon link
            },
            'Potato___healthy': {
                'description': 'Your potato plant is healthy! No treatment is needed.',
                'treatment': [],
                'links': [],
                'pesticide_image': '',
                'buy_link': ''
            },
            'Raspberry___healthy': {
                'description': 'Your raspberry plant is healthy! No treatment is needed.',
                'treatment': [],
                'links': [],
                'pesticide_image': '',
                'buy_link': ''
            },
            'Soybean___healthy': {
                'description': 'Your soybean plant is healthy! No treatment is needed.',
                'treatment': [],
                'links': [],
                'pesticide_image': '',
                'buy_link': ''
            },
            'Squash___Powdery_mildew': {
                'description': 'Powdery mildew is a fungal disease that appears as white powdery spots on squash leaves.',
                'treatment': [
                    'Apply fungicides like sulfur or potassium bicarbonate.',
                    'Prune infected branches to improve air circulation.',
                    'Avoid overhead watering to reduce humidity.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/powdery-mildew',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/powdery-mildew/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=sulfur+fungicide'  # Amazon link
            },
            'Strawberry___Leaf_scorch': {
                'description': 'Leaf scorch is a fungal disease that causes reddish-brown spots on strawberry leaves.',
                'treatment': [
                    'Apply fungicides like chlorothalonil or thiophanate-methyl.',
                    'Remove and destroy infected leaves.',
                    'Avoid overhead watering to reduce humidity.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/strawberry-leaf-scorch',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/leaf-scorch/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=chlorothalonil+fungicide'  # Amazon link
            },
            'Strawberry___healthy': {
                'description': 'Your strawberry plant is healthy! No treatment is needed.',
                'treatment': [],
                'links': [],
                'pesticide_image': '',
                'buy_link': ''
            },
            'Tomato___Bacterial_spot': {
                'description': 'Bacterial spot is a bacterial disease that causes dark, raised lesions on tomato leaves and fruit.',
                'treatment': [
                    'Apply copper-based bactericides.',
                    'Remove and destroy infected plants.',
                    'Avoid overhead watering to reduce spread.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/bacterial-spot-tomato',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/bacterial-spot/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=copper+bactericide'  # Amazon link
            },
            'Tomato___Early_blight': {
                'description': 'Early blight is a fungal disease that causes dark, concentric spots on tomato leaves.',
                'treatment': [
                    'Apply fungicides like chlorothalonil or mancozeb.',
                    'Rotate crops to prevent disease buildup in the soil.',
                    'Remove and destroy infected plant debris.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/early-blight-tomato',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/early-blight/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=chlorothalonil+fungicide'  # Amazon link
            },
            'Tomato___Late_blight': {
                'description': 'Late blight is a fungal disease that causes dark, water-soaked lesions on tomato leaves and fruit.',
                'treatment': [
                    'Apply fungicides like chlorothalonil or mancozeb.',
                    'Remove and destroy infected plants immediately.',
                    'Avoid planting tomatoes in the same area repeatedly.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/late-blight-tomato',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/late-blight/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=chlorothalonil+fungicide'  # Amazon link
            },
            'Tomato___Leaf_Mold': {
                'description': 'Leaf mold is a fungal disease that causes yellow spots on tomato leaves.',
                'treatment': [
                    'Apply fungicides like chlorothalonil or mancozeb.',
                    'Improve air circulation by spacing plants appropriately.',
                    'Avoid overhead watering to reduce humidity.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/tomato-leaf-mold',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/leaf-mold/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=chlorothalonil+fungicide'  # Amazon link
            },
            'Tomato___Septoria_leaf_spot': {
                'description': 'Septoria leaf spot is a fungal disease that causes small, dark spots on tomato leaves.',
                'treatment': [
                    'Apply fungicides like chlorothalonil or mancozeb.',
                    'Remove and destroy infected leaves.',
                    'Avoid overhead watering to reduce humidity.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/septoria-leaf-spot-tomato',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/septoria-leaf-spot/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=chlorothalonil+fungicide'  # Amazon link
            },
            'Tomato___Spider_mites Two-spotted_spider_mite': {
                'description': 'Spider mites are tiny pests that cause yellow stippling on tomato leaves.',
                'treatment': [
                    'Apply miticides like neem oil or insecticidal soap.',
                    'Spray plants with water to dislodge mites.',
                    'Introduce natural predators like ladybugs.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/spider-mites',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/spider-mites/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=neem+oil'  # Amazon link
            },
            'Tomato___Target_Spot': {
                'description': 'Target spot is a fungal disease that causes dark, concentric spots on tomato leaves.',
                'treatment': [
                    'Apply fungicides like chlorothalonil or mancozeb.',
                    'Remove and destroy infected leaves.',
                    'Avoid overhead watering to reduce humidity.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/target-spot-tomato',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/target-spot/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=chlorothalonil+fungicide'  # Amazon link
            },
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
                'description': 'Tomato yellow leaf curl virus is a viral disease that causes yellowing and curling of tomato leaves.',
                'treatment': [
                    'Remove and destroy infected plants to prevent spread.',
                    'Control whiteflies (the disease vector) using insecticides.',
                    'Plant resistant tomato varieties.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/tomato-yellow-leaf-curl-virus',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/tomato-yellow-leaf-curl/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=whitefly+insecticide'  # Amazon link
            },
            'Tomato___Tomato_mosaic_virus': {
                'description': 'Tomato mosaic virus is a viral disease that causes mottled leaves and stunted growth in tomatoes.',
                'treatment': [
                    'Remove and destroy infected plants to prevent spread.',
                    'Control aphids (the disease vector) using insecticides.',
                    'Plant resistant tomato varieties.'
                ],
                'links': [
                    'https://extension.umn.edu/plant-diseases/tomato-mosaic-virus',
                    'https://www.planetnatural.com/pest-problem-solver/plant-disease/tomato-mosaic-virus/'
                ],
                'pesticide_image': 'https://via.placeholder.com/300',  # Placeholder image
                'buy_link': 'https://www.amazon.com/s?k=aphid+insecticide'  # Amazon link
            },
            'Tomato___healthy': {
                'description': 'Your tomato plant is healthy! No treatment is needed.',
                'treatment': [],
                'links': [],
                'pesticide_image': '',
                'buy_link': ''
            }
        }

        # Display Treatment/Solutions
        if disease in treatments:
            st.subheader("Treatment/Solutions")
            st.write(treatments[disease]['description'])
            for step in treatments[disease]['treatment']:
                st.write(f"- {step}")
            
            st.subheader("Useful Links")
            for link in treatments[disease]['links']:
                st.markdown(f"[{link}]({link})")
            
            if treatments[disease]['pesticide_image']:
                st.subheader("Recommended Pesticide")
                st.image(treatments[disease]['pesticide_image'], caption='Recommended Pesticide', use_container_width=True)
            
            if treatments[disease]['buy_link']:
                st.subheader("Buy Pesticide")
                st.markdown(f"[Purchase this pesticide on Amazon]({treatments[disease]['buy_link']})")