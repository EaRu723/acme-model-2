import logging
import datetime
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from PIL import Image
import io
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
from predict_on_img import ModelInit
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache the Firebase initialization
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate({
            "type": st.secrets["type"],
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

# Cache the model loading
@st.cache_resource
def load_model():
    bucket = storage.bucket()
    blob = bucket.blob("weights/model_fold_4.pth")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as temp_file:
        try:
            blob.download_to_filename(temp_file.name)
            model = ModelInit(path_checkpoint=temp_file.name)
            return model
        finally:
            os.unlink(temp_file.name)

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

    return f"gs://{st.secrets['firebase']['storageBucket']}/webApp/{user}/{img_name}"

def process_image(image, side, email, model, database):
    if image:
        try:
            img = Image.open(io.BytesIO(image.read()))
            img.load()

            # Resize the image to make it smaller
            max_size = (400, 400)  # Adjust this tuple to the desired size
            img.thumbnail(max_size)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                img.save(temp_file.name, optimize=True, quality=85)
                temp_file_path = temp_file.name

            if email:
                img_url = save_image(email, side, img, image.name)

            predictions = model.predict_on_img(img)
            predictions = [tensor.tolist() for tensor in predictions]

            clarity_score = str(100 - 25 * (predictions[0][0])).strip()
            num_blemishes = str(predictions[1][0])

            st.write(f"{side.capitalize()} image predictions:")
            st.write(f"{side.capitalize()} clarity score: {clarity_score} %")
            st.write(f"{side.capitalize()} number of blemishes: {num_blemishes}")

            if email:
                save_prediction(database, email, side, predictions, img_url)

            result_text = f"{side.capitalize()} image predictions:\n\n"
            result_text += f"{side.capitalize()} clarity score: {clarity_score} %\n"
            result_text += f"{side.capitalize()} number of blemishes: {num_blemishes}"

            return temp_file_path, result_text
        except Exception as e:
            logger.error(f"Error processing {side} image: {str(e)}")
            st.write(f"Error processing {side} image: {str(e)}")
    return None, None

def is_valid_email(email):
    return "@" in email and "." in email

def send_email(to_email, subject, body, image_paths, bcc_email):
    smtp_server = st.secrets["email"]["smtp_server"]
    smtp_port = int(st.secrets["email"]["smtp_port"])
    email_address = st.secrets["email"]["email_address"]
    email_password = st.secrets["email"]["email_password"]

    msg = MIMEMultipart()
    msg['From'] = email_address
    msg['To'] = to_email
    msg['Subject'] = subject
    msg['Bcc'] = bcc_email  # Add BCC recipient
    msg.attach(MIMEText(body, 'plain'))

    # Attach images
    for image_path in image_paths:
        if image_path:
            with open(image_path, 'rb') as img:
                img_data = img.read()
                image = MIMEImage(img_data, name=os.path.basename(image_path))
                msg.attach(image)

    server = smtplib.SMTP_SSL(smtp_server, smtp_port)
    server.login(email_address, email_password)
    text = msg.as_string()
    server.sendmail(email_address, [to_email, bcc_email], text)  # Include BCC recipient in the sendmail call
    server.quit()

    # Clean up temporary files
    for image_path in image_paths:
        if os.path.exists(image_path):
            os.remove(image_path)

def main():
    st.title("Y Acne’s Clear Skin Assessment")
    st.header("Use an AI model published by MIT researchers to measure if your skin is getting better or worse. [Read the paper](https://arxiv.org/abs/2403.00268)")

    email = st.text_input("Enter Your Email", help="We'll email your results. Feel free to reply with feedback.")

    # Create three columns for left, front, and right image uploads and previews
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Left")
        left_image = st.file_uploader("For the best results", type=["jpg", "jpeg", "png"], key="left")
        if left_image:
            st.image(left_image, caption='Left Image', use_column_width=True)

    with col2:
        st.subheader("Front")
        front_image = st.file_uploader("take consistent photos", type=["jpg", "jpeg", "png"], key="front")
        if front_image:
            st.image(front_image, caption='Front Image', use_column_width=True)

    with col3:
        st.subheader("Right")
        right_image = st.file_uploader("and check in daily.", type=["jpg", "jpeg", "png"], key="right")
        if right_image:
            st.image(right_image, caption='Right Image', use_column_width=True)

    if st.button("Submit"):
        if not is_valid_email(email):
            st.error("Please enter a valid email address.")
        else:
            if left_image or right_image or front_image:
                database = init_firebase()
                model = load_model()

                results = ""
                image_paths = []

                with col1:
                    if left_image:
                        img_path, left_results = process_image(left_image, "left", email, model, database)
                        if left_results:
                            results += f"{left_results}\n\n"
                            image_paths.append(img_path)

                with col2:
                    if front_image:
                        img_path, front_results = process_image(front_image, "front", email, model, database)
                        if front_results:
                            results += f"{front_results}\n\n"
                            image_paths.append(img_path)

                with col3:
                    if right_image:
                        img_path, right_results = process_image(right_image, "right", email, model, database)
                        if right_results:
                            results += f"{right_results}\n\n"
                            image_paths.append(img_path)

                bcc_email = "y.andrearusso@gmail.com"  # Replace with your email address
                send_email(email, "Your Acne Assessment Results", results, image_paths, bcc_email)
                st.success("Results have been emailed to you!")

    st.header("Coming Soon…")
    st.write("""
    - Personalized Progress Reports
    - Exclusive Acne-Clearing Tips
    - Supportive Community Access
    """)

    st.write("If you have any questions or need support, feel free to reach out to us at [team@yacne.com](mailto:team@yacne.com).")

if __name__ == "__main__":
    main()