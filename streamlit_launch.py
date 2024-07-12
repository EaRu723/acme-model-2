from PIL import Image
from predict_on_img import ModelInit
import io
import logging
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_prediction(db, user_email, side, scores):
    scores_dict = {
        "severity_score": scores[0][0],
        "num_blemishes": scores[1][0]
    }
    user = user_email.replace("@", "_").replace(".", "_")
    # Save the prediction in firebase
    user_doc_ref = db.collection("webApp").document(user)
    predictions_col_ref = user_doc_ref.collection("predictions")
    predictions_col_ref.add({
        'side': side,
        'scores': scores_dict,
        'date': datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    })

def save_image(user_email, side, img, img_name=None):
    user = user_email.replace("@", "_").replace(".", "_")
    # Left image loaded, store in firebase
    temp_file_path = f"temp_{side}.jpg"
    img.save(temp_file_path)

    bucket = storage.bucket()

    # Create a new blob in the webapp directory and upload the file to Firebase Storage
    blob = bucket.blob("webApp/" + f'{user}/' + img_name)
    try:
        blob.upload_from_filename(temp_file_path)
    except Exception as e:
        st.write("Image already uploaded")
    os.remove(temp_file_path)


# Get credentials from streamlit secrets
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate({
            "type": "service_account",
            "project_id": st.secrets["project_id"],
            "private_key_id": st.secrets["private_key_id"],
            "private_key": st.secrets["private_key"],
            "client_email": st.secrets["client_email"],
            "client_id": st.secrets["client_id"],
            "auth_uri": st.secrets["auth_uri"],
            "token_uri": st.secrets["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["client_x509_cert_url"]
        })
        firebase_admin.initialize_app(cred, {'storageBucket': st.secrets['storageBucket']})
    else:
        app = firebase_admin.get_app()
        cred = credentials.Certificate({
            "type": "service_account",
            "project_id": st.secrets["project_id"],
            "private_key_id": st.secrets["private_key_id"],
            "private_key": st.secrets["private_key"],
            "client_email": st.secrets["client_email"],
            "client_id": st.secrets["client_id"],
            "auth_uri": st.secrets["auth_uri"],
            "token_uri": st.secrets["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["client_x509_cert_url"]
        })
except Exception as e:
    logger.error(f"Error initializing firebase app in prod: {str(e)}")

database = firestore.client()

model = ModelInit(path_checkpoint="lds-weights/model_fold_4.pth")

# Define the main page

st.title("Acne Severity Score")
st.write("This app predicts the severity of your acne and number of blemishes.")
st.write("Please upload photos of the left and right side of your face.")

# Get email to store in firebase storage
email = st.text_input("Enter your email address to keep track of your progress")

# Upload images
st.write("Please upload photos of the left and right side of your face or a front facing photo.")
left_image = st.file_uploader("Upload left image", type=["jpg", "jpeg", "png"])
right_image = st.file_uploader("Upload right image", type=["jpg", "jpeg", "png"])
front_image = st.file_uploader("Upload front image", type=["jpg", "jpeg", "png"])



if (left_image and right_image):
    try:
        logger.info("Received image upload request")

        # Read and process the left image
        left_image_data = left_image.read()
        logger.info("Left image data read successfully")

        try:
            left_img = Image.open(io.BytesIO(left_image_data))
            left_img.load()  # Ensure the image is properly loaded
            logger.info("Left image processed successfully")
        except Exception as e:
            logger.error(f"Invalid left image format: {str(e)}")

        if email and left_img:
            save_image(email, "left", left_img, left_image.name)

        # Read and process the right image
        right_image_data = right_image.read()
        logger.info("Right image data read successfully")

        try:
            right_img = Image.open(io.BytesIO(right_image_data))
            right_img.load()  # Ensure the image is properly loaded
            logger.info("Right image processed successfully")
        except Exception as e:
            logger.error(f"Invalid right image format: {str(e)}")

        if email and right_img:
            save_image(email, "right", right_img, right_image.name)

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

        st.write("Left image predictions:")
        st.write("Left severity score:", left_predictions[0][0])
        st.write("Left number of blemishes:", left_predictions[1][0])

        logger.info(f"Left pred: {left_predictions}")
        logger.info(f"Right pred: {right_predictions}")


        st.write("Right image predictions:")
        st.write("Right severity score:", right_predictions[0][0])
        st.write("Right number of blemishes:", right_predictions[1][0])

        # Acne score storage in firebase db
        if email:
            save_prediction(database, email, "left", left_predictions)
            save_prediction(database, email, "right", right_predictions)

    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        st.write(f"Error processing images: {str(e)}")


if front_image:
    try:
        logger.info("Received front image upload request")

        # Read and process the front image
        front_image_data = front_image.read()
        logger.info("Front image data read successfully")

        try:
            front_img = Image.open(io.BytesIO(front_image_data))
            front_img.load()  # Ensure the image is properly loaded
            logger.info("Front image processed successfully")
        except Exception as e:
            logger.error(f"Invalid front image format: {str(e)}")

        if email and front_img:
            save_image(email, "front", front_img, front_image.name)

        # Get predictions
        logger.info("Making predictions on front image")
        try:
            front_predictions = model.predict_on_img(front_img)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            st.write(f"Error making predictions: {str(e)}")
        # Convert predictions to list
        front_predictions = [tensor.tolist() for tensor in front_predictions]

        logger.info("Front image processing and prediction successful")

        st.write("Front image predictions:")
        st.write("Front severity score:", (front_predictions[0][0]))
        st.write("Front number of blemishes:", front_predictions[1][0])

        # Score storage in firebase database
        if email:
            save_prediction(database, email, "front", front_predictions)


    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        st.write(f"Error processing images: {str(e)}")

