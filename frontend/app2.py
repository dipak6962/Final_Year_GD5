import streamlit as st
import tensorflow as tf
import keras
# from keras import models
# from keras.models import load_model
# from PIL import Image
import numpy as np 

# Define function for image classification
def classify_image(model, image, class_names):
    # Preprocess the image (if needed)
    # Perform image classification using the loaded model
    prediction = model.predict(image)
    # Convert prediction to class label using class_names
    predicted_class = np.argmax(prediction[0])
    class_label = class_names[predicted_class]
    return class_label

# Define function to load each model

    load_model(model_path)
def load_inception_resnet_v2_model(model_path):
    # model = tf.keras.models.load_model(model_path)
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Loading mOdel error: {e}")
        # st.error(f"Error loading Inception ResNet V2 model: {str(e)}")
        return None


    
# Desired class names
class_names = ['Healthy-KL0', 'Doubtful-KL1', 'Minimal-KL2', 'Moderate-KL3', 'Severe-KL4']

def main():
    # Define Streamlit app layout
    st.title("Deep Learning Model Selection")

    # Add a brief description of the app
    st.write("Select a deep learning model to use for classification:")

    # Define buttons for each model
    model_names = ["Inception ResNet V2"]
    selected_model = st.radio("Select Model:", model_names)

    # Add file uploader for image input
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    # Display the selected image
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Add button to trigger model loading
    if st.button("Load Model"):
        print("LoadModle button Pressed")
    
        if selected_model == "Inception ResNet V2":
            try:
                model_path = "/home/dipak/Documents/knee_model_python/src/models/efficientnet.keras"
                try:
                    model = load_inception_resnet_v2_model(model_path)
                except Exception as e:
                    print(f"Failed to load the model here{e}")
                print("model loading inside")
                if model is not None:
                    print("Model loaded successfully ")
                    st.success("Inception ResNet V2 model loaded successfully.")
                else:
                    print ("model is null")
            except:
                print("Model failed to load here !!!!!!!")
    
        else:
            st.error("Invalid model selected.")

        # Process the uploaded image and make a prediction
        if model is not None:
            try:
                print("Inside the model try")
                if uploaded_file is not None:
                    print("inside the upload file")
                    # Open the uploaded image
                    image = Image.open(uploaded_file)
                    # image = uploaded_file
                    if image is not None:
                        print("inside Image loading ")
                        # Convert image to numpy array
                        image_array = np.array(image)
                        
                        # Check if the file is an image
                        if image.format not in ["JPEG", "PNG"]:
                            st.error("Uploaded file is not a valid image.")
                        else:
                            print("Inside Image processing !!!")
                            # Ensure image has three color channels
                            if len(image_array.shape) == 2:
                                image_array = np.stack((image_array,) * 3, axis=-1)

                            image_array = np.expand_dims(image_array, axis=-1)

                            # # Repeat the channel dimension 3 times to match RGB format
                            image_array = np.repeat(image_array, 3, axis=-1)

                            # # Reshape to (1, 224, 224, 3) as model expects batch dimension
                            image_array = np.expand_dims(image_array, axis=0)
                            # # Check the shape of the image array after preprocessing

                            print("!!!Shape of preprocessed image array:", image_array.shape)    
                            
                            # # Resize image to match model input shape (e.g., 224x224 for ResNet50)
                            # image_array = tf.image.resize(image_array, (224, 224))
                            
                            # # Expand dimensions to match model input shape
                            # image_array = np.expand_dims(image_array, axis=0)
                            
                            # Perform image classification
                            prediction = classify_image(model, image_array, class_names)
        
                            # Display the prediction result using the class label
                            st.write("Prediction:", prediction)
                            # Or, if you want to show both probabilities and class label:
                        
                    else:
                        st.error("Error: Uploaded file is empty or not a valid image.")
            except Exception as e:
                st.error(f"Error processing the uploaded image: {str(e)}")

if __name__ =="__main__":
    try:
        model_path = "/home/dipak/Documents/knee_model_python/src/models/model_ResNet50_ft.hdf5"
        model = load_inception_resnet_v2_model(model_path)
        if model is not None:
            print("Model loaded successfully ")
            # st.success("Inception ResNet V2 model loaded successfully.")
        else:
            print ("model is null")
    except Exception as e:
        print(f"Model failed to load here !!!!!!!:: {e}")