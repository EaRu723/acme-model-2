import logging
import datetime
import os
from PIL import Image
import io
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
from predict_on_img import ModelInit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data
def save_prediction(_db, user_email, side, scores, img_url):
    user = user_email.replace("@", "_").replace(".", "_")
    user_doc_ref = _db.collection("webApp").document(user)
    predictions_col_ref = user_doc_ref.collection("predictions")
    predictions_col_ref.add({
        'side': side,
        'scores': {
            "severity_score": scores[0][0],
            "num_blemishes": scores[1][0]
        },
        'date': datetime.datetime.now(),
        'img_url': img_url
    })


@st.cache_resource
def save_image(user_email, side, _img, img_name):
    user = user_email.replace("@", "_").replace(".", "_")
    temp_file_path = f"temp_{side}.jpg"
    _img.save(temp_file_path, optimize=True, quality=85)

    bucket = storage.bucket()
    blob = bucket.blob(f"webApp/{user}/{img_name}")

    try:
        blob.upload_from_filename(temp_file_path)
    except Exception as e:
        st.write("Image already uploaded")
    finally:
        os.remove(temp_file_path)

    return f"gs://{st.secrets['storageBucket']}/webApp/{user}/{img_name}"


@st.cache_resource
def init_firebase():
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
    return firestore.client()


@st.cache_resource
def load_model():
    return ModelInit(path_checkpoint="lds-weights/model_fold_4.pth")


def process_image(image, side, email, model, database):
    if image:
        try:
            img = Image.open(io.BytesIO(image.read()))
            img.load()

            if email:
                img_url = save_image(email, side, img, image.name)

            predictions = model.predict_on_img(img)
            predictions = [tensor.tolist() for tensor in predictions]

            st.write(f"{side.capitalize()} image predictions:")
            st.write(f"{side.capitalize()} severity score:", predictions[0][0])
            st.write(f"{side.capitalize()} number of blemishes:", predictions[1][0])

            if email:
                save_prediction(database, email, side, predictions, img_url)

            return predictions
        except Exception as e:
            logger.error(f"Error processing {side} image: {str(e)}")
            st.write(f"Error processing {side} image: {str(e)}")
    return None


def main():
    st.title("Acne Severity Score")
    st.write("This app predicts the severity of your acne and number of blemishes.")
    st.write("Please upload photos of the left and right side of your face.")

    email = st.text_input("Enter your email address to keep track of your progress")

    st.write("Please upload photos of the left and right side of your face or a front facing photo.")
    left_image = st.file_uploader("Upload left image", type=["jpg", "jpeg", "png"])
    right_image = st.file_uploader("Upload right image", type=["jpg", "jpeg", "png"])
    front_image = st.file_uploader("Upload front image", type=["jpg", "jpeg", "png"])

    if left_image or right_image or front_image:
        model = load_model()
        database = init_firebase() if email else None

        if left_image and right_image:
            process_image(left_image, "left", email, model, database)
            process_image(right_image, "right", email, model, database)

        if front_image:
            process_image(front_image, "front", email, model, database)


if __name__ == "__main__":
    main()