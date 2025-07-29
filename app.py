import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO & WARNING logs

import warnings
warnings.filterwarnings("ignore")         # Suppress Python warnings

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# Load model
model = load_model(r'C:\Learning\plant-disease-app\backend\models\plant_disease_model.h5', compile=False)

# Load class names
with open(r'C:\Learning\plant-disease-app\backend\models\class_names.json') as f:
    class_names = json.load(f)

# Preventive tips dictionary
preventive_tips = {
    "Apple___Apple_scab": "Prune trees to improve air circulation and apply fungicides early in the season. Remove fallen leaves.",
    "Apple___Black_rot": "Remove infected fruit and prune cankers. Apply fungicides and avoid wounding trees.",
    "Apple___Cedar_apple_rust": "Plant resistant varieties. Remove nearby cedar trees if possible or apply fungicides during early growth.",
    "Apple___healthy": "Your plant appears healthy. Keep monitoring and maintain good watering and fertilizing practices.",
    "Blueberry___healthy": "Your plant is healthy. Ensure well-drained, acidic soil and regular watering.",
    "Cherry_(including_sour)___Powdery_mildew": "Prune for air flow, avoid overhead watering, and apply sulfur-based fungicides if needed.",
    "Cherry_(including_sour)___healthy": "Your plant is healthy. Regular pruning and care will keep it that way.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops, use resistant hybrids, and apply foliar fungicides as needed.",
    "Corn_(maize)___Common_rust_": "Use resistant varieties and apply fungicides when disease pressure is high.",
    "Corn_(maize)___Northern_Leaf_Blight": "Rotate crops and use hybrid varieties with resistance. Apply fungicide during tasseling if needed.",
    "Corn_(maize)___healthy": "No issues detected. Ensure balanced fertilization and pest monitoring.",
    "Grape___Black_rot": "Prune infected vines and apply fungicide early in the season. Remove mummified berries.",
    "Grape___Esca_(Black_Measles)": "Avoid pruning wounds during wet conditions. Remove infected vines and maintain good vine health.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Remove infected leaves and apply appropriate fungicides. Ensure air circulation.",
    "Grape___healthy": "Grapevine appears healthy. Continue regular pruning and monitoring for pests.",
    "Orange___Haunglongbing_(Citrus_greening)": "No cure exists. Remove infected trees and control psyllid insects with insecticides.",
    "Peach___Bacterial_spot": "Plant resistant cultivars, avoid overhead watering, and apply copper-based bactericides.",
    "Peach___healthy": "Peach tree is healthy. Maintain regular pruning and fertilization.",
    "Pepper,_bell___Bacterial_spot": "Use certified disease-free seeds, apply copper sprays, and practice crop rotation.",
    "Pepper,_bell___healthy": "No disease detected. Maintain regular watering and monitor for pests.",
    "Potato___Early_blight": "Use crop rotation, apply fungicides like chlorothalonil, and remove infected debris.",
    "Potato___Late_blight": "Remove infected plants and apply copper or systemic fungicides early. Avoid overhead watering.",
    "Potato___healthy": "Potato crop looks healthy. Ensure well-drained soil and disease-free seed potatoes.",
    "Raspberry___healthy": "Healthy plant detected. Regularly prune and mulch to retain soil moisture.",
    "Soybean___healthy": "Crop appears healthy. Rotate crops and monitor for pests and pathogens.",
    "Squash___Powdery_mildew": "Apply neem oil or sulfur sprays. Water at soil level and ensure good air flow.",
    "Strawberry___Leaf_scorch": "Avoid overhead watering, remove infected leaves, and apply appropriate fungicides.",
    "Strawberry___healthy": "Strawberry plant is healthy. Keep beds weed-free and irrigate properly.",
    "Tomato___Bacterial_spot": "Use disease-free seeds, apply copper sprays, and avoid handling plants when wet.",
    "Tomato___Early_blight": "Remove lower leaves, mulch base of plants, and use fungicides containing chlorothalonil.",
    "Tomato___Late_blight": "Destroy infected plants, avoid overhead irrigation, and apply fungicides like mancozeb.",
    "Tomato___Leaf_Mold": "Increase ventilation in greenhouses. Apply fungicides and avoid overhead watering.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves and apply fungicides. Avoid splashing water onto leaves.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spray with miticides or insecticidal soap. Increase humidity and prune heavily infested areas.",
    "Tomato___Target_Spot": "Use fungicides and remove infected leaves. Practice crop rotation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies, remove infected plants, and use resistant varieties.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants, disinfect tools, and avoid smoking near plants.",
    "Tomato___healthy": "Healthy tomato plant. Maintain consistent watering and monitor regularly."
}

# Streamlit UI
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose a plant leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((128, 128))
    img_array = keras_image.img_to_array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]  # Remove alpha channel if any

    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    # Show prediction
    st.markdown(f"### 🩺 Prediction: `{predicted_label}`")
    st.markdown(f"### 📊 Confidence: `{confidence:.2f}%`")

    tip = preventive_tips.get(predicted_label, "General care recommended. Consider consulting a local agriculture expert.")
    st.markdown("### 🌱 Preventive Tips or Remedies:")
    st.info(tip)



# import streamlit as st

# # ✅ Must be the first Streamlit command
# st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# # Other imports
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import json
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image as keras_image

# # Load model
# model = load_model(r'C:\Learning\plant-disease-app\backend\models\plant_disease_model.h5')

# # Load class names (list of labels)
# with open(r'C:\Learning\plant-disease-app\backend\models\class_names.json') as f:
#     class_names = json.load(f)  # This should be a list like ["Apple___Scab", "Tomato___Bacterial_spot", ...]

# # Define preventive tips/remedies (add more as needed)
# preventive_tips = {
#     "Apple___Apple_scab": "Prune trees to improve air circulation and apply fungicides early in the season. Remove fallen leaves.",
#     "Apple___Black_rot": "Remove infected fruit and prune cankers. Apply fungicides and avoid wounding trees.",
#     "Apple___Cedar_apple_rust": "Plant resistant varieties. Remove nearby cedar trees if possible or apply fungicides during early growth.",
#     "Apple___healthy": "Your plant appears healthy. Keep monitoring and maintain good watering and fertilizing practices.",
    
#     "Blueberry___healthy": "Your plant is healthy. Ensure well-drained, acidic soil and regular watering.",
    
#     "Cherry_(including_sour)___Powdery_mildew": "Prune for air flow, avoid overhead watering, and apply sulfur-based fungicides if needed.",
#     "Cherry_(including_sour)___healthy": "Your plant is healthy. Regular pruning and care will keep it that way.",
    
#     "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops, use resistant hybrids, and apply foliar fungicides as needed.",
#     "Corn_(maize)___Common_rust_": "Use resistant varieties and apply fungicides when disease pressure is high.",
#     "Corn_(maize)___Northern_Leaf_Blight": "Rotate crops and use hybrid varieties with resistance. Apply fungicide during tasseling if needed.",
#     "Corn_(maize)___healthy": "No issues detected. Ensure balanced fertilization and pest monitoring.",
    
#     "Grape___Black_rot": "Prune infected vines and apply fungicide early in the season. Remove mummified berries.",
#     "Grape___Esca_(Black_Measles)": "Avoid pruning wounds during wet conditions. Remove infected vines and maintain good vine health.",
#     "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Remove infected leaves and apply appropriate fungicides. Ensure air circulation.",
#     "Grape___healthy": "Grapevine appears healthy. Continue regular pruning and monitoring for pests.",
    
#     "Orange___Haunglongbing_(Citrus_greening)": "No cure exists. Remove infected trees and control psyllid insects with insecticides.",
    
#     "Peach___Bacterial_spot": "Plant resistant cultivars, avoid overhead watering, and apply copper-based bactericides.",
#     "Peach___healthy": "Peach tree is healthy. Maintain regular pruning and fertilization.",
    
#     "Pepper,_bell___Bacterial_spot": "Use certified disease-free seeds, apply copper sprays, and practice crop rotation.",
#     "Pepper,_bell___healthy": "No disease detected. Maintain regular watering and monitor for pests.",
    
#     "Potato___Early_blight": "Use crop rotation, apply fungicides like chlorothalonil, and remove infected debris.",
#     "Potato___Late_blight": "Remove infected plants and apply copper or systemic fungicides early. Avoid overhead watering.",
#     "Potato___healthy": "Potato crop looks healthy. Ensure well-drained soil and disease-free seed potatoes.",
    
#     "Raspberry___healthy": "Healthy plant detected. Regularly prune and mulch to retain soil moisture.",
    
#     "Soybean___healthy": "Crop appears healthy. Rotate crops and monitor for pests and pathogens.",
    
#     "Squash___Powdery_mildew": "Apply neem oil or sulfur sprays. Water at soil level and ensure good air flow.",
    
#     "Strawberry___Leaf_scorch": "Avoid overhead watering, remove infected leaves, and apply appropriate fungicides.",
#     "Strawberry___healthy": "Strawberry plant is healthy. Keep beds weed-free and irrigate properly.",
    
#     "Tomato___Bacterial_spot": "Use disease-free seeds, apply copper sprays, and avoid handling plants when wet.",
#     "Tomato___Early_blight": "Remove lower leaves, mulch base of plants, and use fungicides containing chlorothalonil.",
#     "Tomato___Late_blight": "Destroy infected plants, avoid overhead irrigation, and apply fungicides like mancozeb.",
#     "Tomato___Leaf_Mold": "Increase ventilation in greenhouses. Apply fungicides and avoid overhead watering.",
#     "Tomato___Septoria_leaf_spot": "Remove infected leaves and apply fungicides. Avoid splashing water onto leaves.",
#     "Tomato___Spider_mites Two-spotted_spider_mite": "Spray with miticides or insecticidal soap. Increase humidity and prune heavily infested areas.",
#     "Tomato___Target_Spot": "Use fungicides and remove infected leaves. Practice crop rotation.",
#     "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies, remove infected plants, and use resistant varieties.",
#     "Tomato___Tomato_mosaic_virus": "Remove infected plants, disinfect tools, and avoid smoking near plants.",
#     "Tomato___healthy": "Healthy tomato plant. Maintain consistent watering and monitor regularly."
# }


# # Streamlit UI
# st.title("🌿 Plant Disease Detection")
# st.write("Upload a leaf image and the model will predict the disease.")

# uploaded_file = st.file_uploader("Choose a plant leaf image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Show the uploaded image
#     img = Image.open(uploaded_file)
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     # Resize and preprocess
#     img = img.resize((128, 128))  # Adjust based on model input
#     img_array = keras_image.img_to_array(img)
    
#     if img_array.shape[-1] == 4:
#         img_array = img_array[:, :, :3]  # Remove alpha channel if present

#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     # Predict
#     predictions = model.predict(img_array)[0]
#     predicted_index = np.argmax(predictions)
#     predicted_label = class_names[predicted_index]
#     confidence = predictions[predicted_index] * 100

#     # Show prediction and confidence
#     st.markdown(f"### 🩺 Prediction: `{predicted_label}`")
#     st.markdown(f"### 📊 Confidence: `{confidence:.2f}%`")

#     # Show preventive tips if available
#     tip = preventive_tips.get(predicted_label, "General care recommended. Consider consulting a local agriculture expert.")
#     st.markdown("### 🌱 Preventive Tips or Remedies:")
#     st.info(tip)

















# import streamlit as st

# st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import json
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image as keras_image

# model = load_model(r'C:\Learning\plant-disease-app\backend\models\plant_disease_model.h5')

# with open(r'C:\Learning\plant-disease-app\backend\models\class_names.json') as f:
#     label_map = json.load(f)  # This is a list, not a dict

# st.title("🌿 Plant Disease Detection")
# st.write("Upload a leaf image and the model will predict the disease.")

# uploaded_file = st.file_uploader("Choose a plant leaf image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:

#     img = Image.open(uploaded_file)
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     img = img.resize((128, 128))
#     img_array = keras_image.img_to_array(img)

#     if img_array.shape[-1] == 4:
#         img_array = img_array[:, :, :3]

#     img_array = np.expand_dims(img_array, axis=0) / 255.0  

#     predictions = model.predict(img_array)[0]
#     predicted_index = np.argmax(predictions)

#     predicted_label = label_map[predicted_index]
#     confidence = predictions[predicted_index] * 100

#     st.markdown(f"### 🩺 Prediction: `{predicted_label}`")
#     st.markdown(f"### 📊 Confidence: `{confidence:.2f}%`")
