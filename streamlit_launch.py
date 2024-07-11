from PIL import Image
from predict_on_img import ModelInit
import io
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = ModelInit(path_checkpoint="lds-weights/model_fold_4.pth")

# Define the main page

st.title("Acne Severity Score")
st.write("This app predicts the severity of your acne and number of blemishes.")
st.write("Please upload photos of the left and right side of your face.")

# Upload images
left_image = st.file_uploader("Upload left image", type=["jpg", "jpeg", "png"])
right_image = st.file_uploader("Upload right image", type=["jpg", "jpeg", "png"])


if left_image and right_image:
    try:
        logger.info("Received image upload request")

        # Store images in firebase storage

        # Read and process the left image
        left_image_data = left_image.read()
        logger.info("Left image data read successfully")

        try:
            left_img = Image.open(io.BytesIO(left_image_data))
            left_img.load()  # Ensure the image is properly loaded
            logger.info("Left image processed successfully")
        except Exception as e:
            logger.error(f"Invalid left image format: {str(e)}")

        # Read and process the right image
        right_image_data = right_image.read()
        logger.info("Right image data read successfully")

        try:
            right_img = Image.open(io.BytesIO(right_image_data))
            right_img.load()  # Ensure the image is properly loaded
            logger.info("Right image processed successfully")
        except Exception as e:
            logger.error(f"Invalid right image format: {str(e)}")
        # Get predictions
        logger.info("Making predictions on images")
        try:
            left_predictions = model.predict_on_img(left_img)
            right_predictions = model.predict_on_img(right_img)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            st.write(f"Error making predictions: {str(e)}")
        # Convert predictions to list
        left_predictions = [tensor.tolist() for tensor in left_predictions]
        right_predictions = [tensor.tolist() for tensor in right_predictions]

        logger.info("Image processing and prediction successful")

        logger.info(f"Image pred type: {left_predictions[0][0]}")

        st.write("Left image predictions:")
        st.write("Left severity score:", left_predictions[0][0])
        st.write("Left number of blemishes:", left_predictions[1][0])

        st.write("Right image predictions:")
        st.write("Right severity score:", right_predictions[0][0])
        st.write("Right number of blemishes:", right_predictions[1][0])


    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        st.write(f"Error processing images: {str(e)}")

