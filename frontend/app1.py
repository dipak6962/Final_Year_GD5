import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np 

# Define function for image classification
def classify_image(model, image, class_names):
    # Perform image classification using the loaded model
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
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

# Define class names
class_names = ['Healthy-KL0', 'Doubtful-KL1', 'Minimal-KL2', 'Moderate-KL3', 'Severe-KL4']

# Define Streamlit app layout
st.title("Knee OA Detection by using X-ray Images")
with st.sidebar:
    # st.image(icon)
    st.subheader("Final Project Knee OA Detection")
    st.caption("Guided by : Prof. Ganesh Kadam")

    st.subheader(":arrow_up: Upload image")
    uploaded_file = st.file_uploader("Choose x-ray image", type=["jpg", "png", "jpeg"])


# Add file uploader for image input
# uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Display the selected image
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Add radio button group for selecting model
    st.write("Select a model for image classification:")
    model_names = ["ResNet50", "InceptionResNet V2", "Xception"]
    selected_model = st.radio("Select Model:", model_names)

    # Add button to trigger model loading
    if st.button("Load Model"):
        if selected_model == "ResNet50":
            model_path = "../src/models/model_ResNet50_ft.hdf5"
            model = load_resnet50_model(model_path)
            if model is not None:
                st.success("ResNet50 model loaded successfully.")
        elif selected_model == "InceptionResNet V2":
            model_path = "/home/dipak/Documents/Knee-OA-detection/src/models/model_InceptionResnetV2_(final).hdf5"
            model = load_inception_resnet_v2_model(model_path)
            if model is not None:
                st.success("Inception ResNet V2 model loaded successfully.")
        elif selected_model == "Xception":
            model_path = "/home/dipak/Documents/Knee-OA-detection/src/models/xception_n(2).hdf5"
            model = load_xception_model(model_path)
            if model is not None:
                st.success("Xception model loaded successfully.")
        else:
            st.error("Invalid model selected.")

        # Process the uploaded image and make a prediction
        if model is not None:
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
                        
                        # Perform image classification
                        prediction = classify_image(model, image_array, class_names)
                        
                        # Display the prediction result
                        st.write("Prediction:", prediction)
                else:
                    st.error("Error: Uploaded file is empty or not a valid image.")
            except Exception as e:
                st.error(f"Error processing the uploaded image: {str(e)}")
