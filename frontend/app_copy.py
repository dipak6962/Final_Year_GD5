import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np 

# Define function to classify image using ensemble of models
def classify_image_ensemble(models, image, class_names):
    predictions = []
    for model in models:
        prediction = model.predict(image)
        predictions.append(prediction)
    
    # Calculate average prediction
    average_prediction = np.mean(predictions, axis=0)
    
    # Get index of maximum value in average prediction
    predicted_class_index = np.argmax(average_prediction)
    
    # Get corresponding class label
    predicted_class = class_names[predicted_class_index]
    
    return predicted_class

# Define function to load each model
def load_resnet50_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading ResNet50 model: {str(e)}")
        return None
    
def load_inception_resnet_v2_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading Inception ResNet V2 model: {str(e)}")
        return None

def load_xception_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading Xception model: {str(e)}")
        return None

# Define Streamlit app layout
st.title("Deep Learning Model Ensemble")

# Add a brief description of the app
st.write("Upload an image and select models for ensemble prediction:")

# Define buttons for each model
model_names = ["ResNet50", "Inception ResNet V2", "Xception"]
selected_models = st.multiselect("Select Models:", model_names)

# Add file uploader for image input
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Define class names
class_names = ['Healthy-KL0', 'Doubtful-KL1', 'Minimal-KL2', 'Moderate-KL3', 'Severe-KL4']

# Load all models
models = []
if st.button("Load Models"):
    for model_name in selected_models:
        if model_name == "ResNet50":
            model_path = "../src/models/model_ResNet50_ft.hdf5"
            model = load_resnet50_model(model_path)
            print("model 1 loaded ")
        elif model_name == "Inception ResNet V2":
            model_path = "/home/dipak/Documents/Knee-OA-detection/src/models/xception_n(1).hdf5"
            model = load_inception_resnet_v2_model(model_path)
        elif model_name == "Xception":
            model_path = "/home/dipak/Documents/Knee-OA-detection/src/models/xception_n(2).hdf5"
            model = load_xception_model(model_path)
        if model is not None:
            models.append(model)
            st.success(f"{model_name} model loaded successfully.")

# Process the uploaded image and make a prediction
if st.button("Make Prediction"):
    if uploaded_file is not None:
        try:
            # Open the uploaded image
            image = Image.open(uploaded_file)
            if image is not None:
                # Convert image to numpy array
                image_array = np.array(image)
                
                # Check if the file is an image
                if image.format not in ["JPEG", "PNG"]:
                    st.error("Uploaded file is not a valid image.")
                else:
                    # Ensure image has three color channels
                    if len(image_array.shape) == 2:
                        image_array = np.stack((image_array,) * 3, axis=-1)
                    
                    # Resize image to match model input shape (e.g., 224x224 for ResNet50)
                    image_array = tf.image.resize(image_array, (224, 224))
                    
                    # Expand dimensions to match model input shape
                    image_array = np.expand_dims(image_array, axis=0)
                    
                    # Perform image classification using ensemble of models
                    prediction = classify_image_ensemble(models, image_array, class_names)
                    
                    # Display the prediction result
                    st.write("Ensemble Prediction:", prediction)
            else:
                st.error("Error: Uploaded file is empty or not a valid image.")
        except Exception as e:
            st.error(f"Error processing the uploaded image: {str(e)}")
