import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import random

# Set page configuration at the top
st.set_page_config(page_title="Plant Classification", page_icon="🌿", layout="wide")

# Custom CSS to set a background image
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

st.markdown(
    """
    <style>
    .reportview-container {
        background-image: url("C:\\Users\\LENOVO\\Downloads\\bg.jpg");  /* Replace with your image URL */
        background-size: cover;  /* Cover the entire background */
        background-position: center;  /* Center the image */
        background-repeat: no-repeat;  /* Prevent repeating the image */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
@st.cache_resource  # Cache the model to avoid reloading on every prediction
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Specify the path to the model
model = load_model(r"C:\Users\LENOVO\Downloads\plant_classifier_model.h5")  # Update with the correct path

# Get the class names from the model's training data
class_names =  ['Amaranthus Green', 'Balloon vine', 'Betel Leaves', 'Celery', 'Chinese Spinach', 'Coriander Leaves', 'Curry Leaf', 'Dwarf Copperleaf (Green)']

# Dictionary mapping each class to plant info
plant_info = {
    'Amaranthus Green': {
        'Kingdom': 'Plantae',
        'Uses': 'Used as a leafy vegetable and in salads.',
        'Care Tips': 'Needs full sun and regular watering. Grows best in well-drained soil.',
        'Nutritional Value': 'Rich in vitamins A, C, and K, and iron.'
    },
    'Balloon vine': {
        'Kingdom': 'Plantae',
        'Uses': 'Used in traditional medicine and as an ornamental plant.',
        'Care Tips': 'Prefers full sun to partial shade and moderate watering.',
        'Nutritional Value': 'Not typically consumed for nutritional purposes.'
    },
    'Betel Leaves': {
        'Kingdom': 'Plantae',
        'Uses': 'Used in traditional medicines and as a stimulant in various cultures.',
        'Care Tips': 'Requires high humidity and indirect sunlight. Grows well in loamy soil.',
        'Nutritional Value': 'Contains antioxidants and is known for its medicinal properties.'
    },
    'Celery': {
        'Kingdom': 'Plantae',
        'Uses': 'Commonly used as a vegetable, in soups and salads.',
        'Care Tips': 'Needs plenty of water and cool temperatures. Prefers rich, well-drained soil.',
        'Nutritional Value': 'Low in calories, high in fiber, and a good source of vitamins K and C.'
    },
    'Chinese Spinach': {
        'Kingdom': 'Plantae',
        'Uses': 'Used as a leafy green vegetable in Asian cuisines.',
        'Care Tips': 'Grows best in warm climates with regular watering and rich soil.',
        'Nutritional Value': 'Rich in iron, calcium, and vitamins A and C.'
    },
    'Coriander Leaves': {
        'Kingdom': 'Plantae',
        'Uses': 'Widely used in cooking as an herb for flavoring dishes.',
        'Care Tips': 'Requires full sun and well-drained soil. Water moderately.',
        'Nutritional Value': 'High in vitamin C, potassium, and antioxidants.'
    },
    'Curry Leaf': {
        'Kingdom': 'Plantae',
        'Uses': 'Used in cooking for its aromatic leaves, common in Indian cuisine.',
        'Care Tips': 'Needs full sun and well-drained soil. Water regularly.',
        'Nutritional Value': 'Rich in vitamin A, calcium, and folic acid.'
    },
    'Dwarf Copperleaf (Green)': {
        'Kingdom': 'Plantae',
        'Uses': 'Used in traditional medicine and as a decorative plant.',
        'Care Tips': 'Thrives in full sun or partial shade and needs moderate watering.',
        'Nutritional Value': 'Not typically consumed for nutritional purposes.'
    }
}

# Simulate plant growth stage estimation
growth_stages = ['Seedling', 'Vegetative', 'Flowering', 'Mature']
def estimate_growth_stage():
    return random.choice(growth_stages)

# Function to preprocess image for model prediction
def preprocess_image(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to predict the plant class
def predict(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Streamlit UI
st.title("🌱 Plant Classification and Comparison")
st.write("Upload images of plants, and the model will predict their class, estimate growth stage, and provide care tips and nutritional values.")

# Image upload widget for comparison
st.subheader("Upload two images for comparison:")
uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "jpeg", "png"], key="file1")
uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "jpeg", "png"], key="file2")

if uploaded_file1 and uploaded_file2:
    # Display both uploaded images
    col1, col2 = st.columns(2)
    with col1:
        image1 = Image.open(uploaded_file1)
        st.image(image1, caption="Uploaded Image 1", use_column_width=True)
    with col2:
        image2 = Image.open(uploaded_file2)
        st.image(image2, caption="Uploaded Image 2", use_column_width=True)

    # Predict button for comparison
    if st.button("Predict and Compare"):
        with st.spinner('Classifying...'):
            # Preprocess the images for the model
            img_array1 = preprocess_image(image1)
            img_array2 = preprocess_image(image2)

            # Make the predictions
            predicted_index1 = predict(model, img_array1)
            predicted_index2 = predict(model, img_array2)

            predicted_class_name1 = class_names[predicted_index1]
            predicted_class_name2 = class_names[predicted_index2]

            # Get plant details and growth stage
            plant_details1 = plant_info[predicted_class_name1]
            plant_details2 = plant_info[predicted_class_name2]
            growth_stage1 = estimate_growth_stage()
            growth_stage2 = estimate_growth_stage()

            # Display the predictions side by side using st.columns
            col1, col2 = st.columns(2)

            # Classification and details for Image 1
            with col1:
                st.markdown(f"<h3 style='color: #2e7d32;'>Classification for Image 1:</h3>", unsafe_allow_html=True)
                st.write(f"**Prediction**: {predicted_class_name1}")
                st.write(f"**Kingdom**: {plant_details1['Kingdom']}")
                st.write(f"**Care Tips**: {plant_details1['Care Tips']}")
                st.write(f"**Nutritional Value**: {plant_details1['Nutritional Value']}")
                st.write(f"**Estimated Growth Stage**: {growth_stage1}")

            # Classification and details for Image 2
            with col2:
                st.markdown(f"<h3 style='color: #2e7d32;'>Classification for Image 2:</h3>", unsafe_allow_html=True)
                st.write(f"**Prediction**: {predicted_class_name2}")
                st.write(f"**Kingdom**: {plant_details2['Kingdom']}")
                st.write(f"**Care Tips**: {plant_details2['Care Tips']}")
                st.write(f"**Nutritional Value**: {plant_details2['Nutritional Value']}")
                st.write(f"**Estimated Growth Stage**: {growth_stage2}")

            # Compare predictions
            if predicted_class_name1 == predicted_class_name2:
                st.write("<h4 style='color: #388e3c;'>Both images belong to the same plant class.</h4>", unsafe_allow_html=True)
            else:
                st.write("<h4 style='color: #d32f2f;'>The two images belong to different plant classes.</h4>", unsafe_allow_html=True)
