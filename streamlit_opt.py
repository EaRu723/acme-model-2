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

    return f"gs://{st.secrets['storageBucket']}/webApp/{user}/{img_name}"

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

            clarity_score = str(get_complexion_class(predictions[0][0]))
            num_blemishes = str(predictions[1][0])

            # st.write(f"{side.capitalize()} image predictions:")
            # st.write(f"{side.capitalize()} clarity score: {clarity_score} %")
            # st.write(f"{side.capitalize()} number of blemishes: {num_blemishes}")

            if email:
                save_prediction(database, email, side, predictions, img_url)

            result_text = ""
            result_text += f"{side.capitalize()} complexion: {clarity_score} \n\n"
            result_text += f"{side.capitalize()} number of blemishes: {num_blemishes}"

            return temp_file_path, result_text
        except Exception as e:
            logger.error(f"Error processing {side} image: {str(e)}")
            st.write(f"Error processing {side} image: {str(e)}")
    return None, None

def is_valid_email(email):
    return "@" in email and "." in email

def get_complexion_class(complexion_score):
    complexion_classes = ["Clear", "Mild", "Moderate", "Severe"]
    return complexion_classes[int(complexion_score)]

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
                # msg.attach(image)

    server = smtplib.SMTP_SSL(smtp_server, smtp_port)
    server.login(email_address, email_password)
    text = msg.as_string()
    server.sendmail(email_address, [to_email, bcc_email], text)  # Include BCC recipient in the sendmail call
    server.quit()

    # Clean up temporary files
    for image_path in image_paths:
        if os.path.exists(image_path):
            os.remove(image_path)

def initialize_session_state():
    if 'visit_recorded' not in st.session_state:
        st.session_state.visit_recorded = False
def page_visit_update(db):
    if not st.session_state.visit_recorded:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        visits_ref = db.collection("webAppAnalytics").document('visits')
        today_visits = visits_ref.collection('dates').document(today)
        try:
            # Try to get the document
            doc = today_visits.get()
            if not doc.exists:
                # If the document doesn't exist, create it with an initial visit count of 1
                today_visits.set({
                    'visits': 1
                })
            else:
                # If the document exists, increment the visit count
                today_visits.update({
                    'visits': firestore.Increment(1)
                })
            st.session_state.visit_recorded = True

        except Exception as e:
            logger.error(f"Error updating page visit: {str(e)}")
            st.error(f"An error occurred while updating visit count. Please try again later.")


def user_submit_update(db, user_email=None):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    visits_ref = db.collection("webAppAnalytics").document('visits')
    today_visits = visits_ref.collection('dates').document(today)
    if not today_visits.get():
        today_visits.add({
        'visitors': user_email,
        })
    else:
        today_visits.update({
            'visitors': firestore.ArrayUnion([user_email]),
        })

def all_users(db, user_email):
    all_users_ref = db.collection("webAppAnalytics").document('all_visitors')
    all_users_ref.update({
        'users': firestore.ArrayUnion([user_email]),
    })

def main():
    initialize_session_state()
    database = init_firebase()
    page_visit_update(database)

    st.title("Y Acne’s Clear Skin Assessment")
    st.header("Get an honest assessment of your skin clarity using an AI model published by MIT researchers. [Read the paper](https://arxiv.org/abs/2403.00268)")

    st.subheader("Upload Photo(s)")
    st.write("""
    We don't store images; we just want to see if people find this model as useful as we did.
    """)

    # Create three columns for left, front, and right image uploads and previews
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Left")
        left_image = st.file_uploader("For the best results", type=["jpg", "jpeg", "png"], key="left")

    with col2:
        st.subheader("Front")
        front_image = st.file_uploader("take consistent photos", type=["jpg", "jpeg", "png"], key="front")

    with col3:
        st.subheader("Right")
        right_image = st.file_uploader("and check in daily.", type=["jpg", "jpeg", "png"], key="right")

    email = st.text_input("Enter your email (optional) if you want to help us eradicate acne.", help="We'll email your results. Feel free to reply with feedback.")



    if st.button("Submit"):
        if email and (not is_valid_email(email)):
            st.error("Please enter a valid email address.")
        elif not (left_image or front_image or right_image):
            st.error("Please upload at least one image before submitting.")
        else:
            # database = init_firebase()
            model = load_model()
            if email:
                user_submit_update(database, email)
                all_users(database, email)
            results = ""
            image_paths = []

            for image, side, col in zip([left_image, front_image, right_image], ["left", "front", "right"], [col1, col2, col3]):
                if image:
                    with col:
                        img_path, side_results = process_image(image, side, email, model, database)
                        if side_results:
                            st.write(side_results)
                            results += f"{side_results}\n\n"
                            image_paths.append(img_path)
                        st.image(image, caption=f'{side.capitalize()} Image', use_column_width=True)

            if results:
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

    st.subheader("About")
    st.write("""
            We use the gold-standard Hayashi method to provide a dermatologist grade evaluation with two outputs per image: a classification into one of four distinct complexions (clear, mild, moderate, severe) and a detailed blemish count.    
    """)

    st.write("""
    I’m Andrea, a MS in Biomedical Engineering and a self-taught software developer passionate about health. After struggling with acne for years, I found this AI model useful and decided to share it to help others. I use this in combination with a journal to identify what clears my skin and what makes it worse. 
    
    Find me on social media:
    - [LinkedIn](https://www.linkedin.com/in/earu723/)
    - [YouTube](https://www.youtube.com/@earu723)
    - [TikTok](https://www.tiktok.com/@y.earu)
    - [Instagram](https://www.instagram.com/y.earu/)
    - [Twitter](https://x.com/AndreaR91659141)

    I'm Dana, a BASc in Biomedical Engineering, specializing in AI. Andrea's story spoke to me and I wanted to help him share this model with others.
    You can find me [here](https://www.linkedin.com/in/danazarezankova/).
    """)



if __name__ == "__main__":
    main()
