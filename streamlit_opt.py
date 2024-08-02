import logging
import datetime
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from PIL import Image, ExifTags
import io
import streamlit as st
import streamlit.components.v1 as components
import firebase_admin
from firebase_admin import credentials, firestore, storage
from predict_on_img import ModelInit
import tempfile
from streamlit_modal import Modal
from andreas_journey import show_andreas_journey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(initial_sidebar_state="collapsed")


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

def navigation():
    page = st.sidebar.radio("Go to", ["Home", "Andrea's Journey"])
    return page

def save_prediction(db, user_email, side, scores, img_url):
    user = user_email.replace("@", "_").replace(".", "_")
    date_today = datetime.datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    user_doc_ref = db.collection("webApp").document(user)
    predictions_col_ref = user_doc_ref.collection("predictions").document(date_today)
    predictions_col_ref.collection("entries").document(f"{timestamp}_{side}").set({
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
    date_today = datetime.datetime.now().strftime("%Y-%m-%d")
    temp_file_path = f"temp_{side}.jpg"
    _img.save(temp_file_path, optimize=True, quality=85)

    bucket = storage.bucket()
    blob = bucket.blob(f"webApp/{user}/{date_today}/{img_name}")

    try:
        blob.upload_from_filename(temp_file_path)
    except Exception as e:
        st.write("Image already uploaded")
    finally:
        os.remove(temp_file_path)

    return f"gs://{st.secrets['storageBucket']}/webApp/{user}/{date_today}/{img_name}"

def process_image(image, side, email, model, database):
    if image:
        user_submit_update(database, email)
        try:
            img = Image.open(io.BytesIO(image.read()))
            
            # Preserve orientation
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            
            exif = dict(img._getexif().items())
            
            if orientation in exif:
                if exif[orientation] == 3:
                    img = img.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    img = img.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    img = img.rotate(90, expand=True)

            # Resize the image to make it smaller
            max_size = (400, 400)  # Adjust this tuple to the desired size
            img.thumbnail(max_size, Image.LANCZOS)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                img.save(temp_file.name, optimize=True, quality=85)
                temp_file_path = temp_file.name

            predictions = model.predict_on_img(img)
            predictions = [tensor.tolist() for tensor in predictions]

            if email:
                img_url = save_image(email, side, img, image.name)
                save_prediction(database, email, side, predictions, img_url)

            severity_label = get_complexion_class(predictions[0][0])
            num_blemishes = predictions[1][0]

            return temp_file_path, severity_label, num_blemishes
        except Exception as e:
            logger.error(f"Error processing {side} image: {str(e)}")
            st.write(f"Error processing {side} image: {str(e)}")
    return None, None, None

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
                image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                msg.attach(image)

    server = smtplib.SMTP_SSL(smtp_server, smtp_port)
    server.login(email_address, email_password)
    text = msg.as_string()
    server.sendmail(email_address, [to_email] + [bcc_email], text)  # Send to main and BCC recipients
    server.quit()

    # Clean up temporary files
    for image_path in image_paths:
        if os.path.exists(image_path):
            os.remove(image_path)

def initialize_session_state():
    if 'visit_recorded' not in st.session_state:
        st.session_state.visit_recorded = False

def page_visit_update(db):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    visits_ref = db.collection("webAppAnalytics").document('visits')
    today_visits = visits_ref.collection('dates').document(today)
    try:
        doc = today_visits.get()
        if not doc.exists:
            today_visits.set({'visits': 1, 'photoSubmissions': 0})
        else:
            today_visits.update({'visits': firestore.Increment(1)})
        st.session_state.visit_recorded = True
    except Exception as e:
        logger.error(f"Error updating page visit: {str(e)}")
        st.error(f"An error occurred while updating visit count. Please try again later.")

def user_submit_update(db, user_email=None):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    visits_ref = db.collection("webAppAnalytics").document('visits')
    today_visits = visits_ref.collection('dates').document(today)
    if not today_visits.get():
        today_visits.add({'visitors': user_email})
    else:
        today_visits.update({'visitors': firestore.ArrayUnion([user_email])})

def all_users(db, user_email):
    all_users_ref = db.collection("webAppAnalytics").document('all_visitors')
    all_users_ref.update({'users': firestore.ArrayUnion([user_email])})

def store_routine(_db, routine, user_email):
    user = user_email.replace("@", "_").replace(".", "_")
    date_today = datetime.datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    user_doc_ref = _db.collection("webApp").document(user)
    routines_col_ref = user_doc_ref.collection("routines").document(date_today)
    routines_col_ref.collection("entries").document(timestamp).set({
        'routine': routine,
        'date': datetime.datetime.now()
    })

def track_photo_submission(db):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    visits_ref = db.collection("webAppAnalytics").document('visits')
    today_visits = visits_ref.collection('dates').document(today)
    try:
        today_visits.update({'photoSubmissions': firestore.Increment(1)})
    except Exception as e:
        logger.error(f"Error updating photo submission count: {str(e)}")
        st.error(f"An error occurred while updating submission count. Please try again later.")


image_url = "https://firebasestorage.googleapis.com/v0/b/loltony.appspot.com/o/IMG_1008.jpeg?alt=media&token=0d73d2d2-3f41-4343-b367-c89d0cce8c70"

# Function to determine the message and emoji
def get_message_and_emoji(severity_label):
    messages = {
        "Clear": "Your skin is looking clear and radiant! 😁",
        "Mild": "Your skin is looking great, with just a few minor blemishes. 😃",
        "Moderate": "You're dealing with some moderate acne, which is common and manageable. 😊",
        "Severe": "We understand that dealing with severe acne can be challenging, but you're not alone. 🙂"
    }
    return messages.get(severity_label, "Unknown severity")

def get_detailed_explanation(severity_label):
    explanations = {
        "Clear": "Fantastic news! Keep up the amazing work with your skincare routine. We love seeing your progress, so check back in regularly. If you have any questions or need support, we're always here for you!",
        "Mild": "Keep checking in with us to track your progress. We're cheering you on every step of the way!",
        "Moderate": "Your journey to clear skin is a process, and we're here to support you. Remember to check back regularly to see how your skin is doing and get fresh advice. We're in this together!",
        "Severe": "Your journey is unique, and with the right care, things can improve. We're here to support you every step of the way. Keep checking in to monitor your progress. Together, we can make a difference!"
    }
    return explanations.get(severity_label, "")

def present_results(severity_labels, num_blemishes_list):
    overall_severity = max(severity_labels, key=severity_labels.count)  # Most frequent label
    total_blemishes = sum(num_blemishes_list)
    message = get_message_and_emoji(overall_severity)

    st.header(f"Great news! Your overall acne severity is {overall_severity} 😃")

    st.write(f"**Total blemishes:** {total_blemishes}")
    
    sides = ["Left", "Right"]
    for i, (severity, blemishes) in enumerate(zip(severity_labels, num_blemishes_list)):
        st.write(f"**{sides[i]} side:** {blemishes} blemishes ({severity})")

    st.markdown(message)

    if len(severity_labels) == 2 and num_blemishes_list[1] != num_blemishes_list[0]:
        more_affected_side = "right" if num_blemishes_list[1] > num_blemishes_list[0] else "left"
        st.info(f"💡 Your {more_affected_side} side shows more blemishes. You might want to pay extra attention to your {more_affected_side} side in your skincare routine.")
    
    st.success("Keep up the good work! Check in again next week to track your progress.")


    summary = f"""
    Great news! Your overall acne severity is {overall_severity} 😃
    Total blemishes: {total_blemishes}
    """

    for i, (side, severity, blemishes) in enumerate(zip(sides, severity_labels, num_blemishes_list)):
        summary += f"{side} side: {blemishes} blemishes ({severity})\n"

    summary += f"\n{message}\n\nKeep up the good work! Check in again next week to track your progress."
    
    return summary

def save_user_consent(db, user_email, consent):
    if not user_email:
        logger.info("User consent not saved: No email provided")
        return

    user = user_email.replace("@", "_").replace(".", "_")
    try:
        user_doc_ref = db.collection("webApp").document(user)
        user_doc_ref.set({'consent': consent}, merge=True)
        logger.info(f"Successfully saved consent for user: {user}")
    except Exception as e:
        logger.error(f"Error saving user consent: {str(e)}")

def scroll_to_section():
    st.session_state["show_upload_section"] = True
    st.components.v1.html(
        """
        <script>
        function scroll_to_upload() {
            var upload_section = window.parent.document.querySelector('.stButton');
            upload_section.scrollIntoView({ behavior: 'smooth' });
        }
        scroll_to_upload();
        </script>
        """,
        height=0,
    )

def routine_entry_form():
    st.subheader("Your Skincare Routine")
    
    routine = st.text_area(
        "Enter your routine",
        placeholder="Provide as much detail as you can about skincare, diet, supplementation, and we'll get back to you with personalized feedback.",
        help="Include information about your morning and evening skincare steps, diet, and any supplements you take."
    )
    
    return routine

def store_routine(_db, routine, user_email):
    user = user_email.replace("@", "_").replace(".", "_")
    date_today = datetime.datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    user_doc_ref = _db.collection("webApp").document(user)
    routines_col_ref = user_doc_ref.collection("routines").document(date_today)
    routines_col_ref.collection("entries").document(timestamp).set({
        'routine': routine,
        'date': datetime.datetime.now()
    })

def track_user_interaction(db, has_image, has_routine):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    analytics_ref = db.collection("webAppAnalytics").document('userInteractions')
    today_interactions = analytics_ref.collection('dates').document(today)
    
    try:
        doc = today_interactions.get()
        if not doc.exists:
            today_interactions.set({
                'imageSubmissions': 1 if has_image else 0,
                'routineSubmissions': 1 if has_routine else 0,
                'combinedSubmissions': 1 if (has_image and has_routine) else 0
            })
        else:
            today_interactions.update({
                'imageSubmissions': firestore.Increment(1) if has_image else firestore.Increment(0),
                'routineSubmissions': firestore.Increment(1) if has_routine else firestore.Increment(0),
                'combinedSubmissions': firestore.Increment(1) if (has_image and has_routine) else firestore.Increment(0)
            })
        logger.info(f"Successfully tracked user interaction: images={has_image}, routine={has_routine}")
    except Exception as e:
        logger.error(f"Error tracking user interaction: {str(e)}")

def main():
    initialize_session_state()
    database = init_firebase()
    page_visit_update(database)

    page = navigation()

    if page == "Home":
        show_home_page(database)
    elif page == "Andrea's Journey":
        show_andreas_journey(database)

def show_home_page(database):
    st.session_state["show_modal"] = False

    st.title("Is your acne improving?")
    st.subheader("Get a dermatologist-level assessment in seconds.")

    # Embed the Loom video
    components.html(
        """
        <div style="position: relative; padding-bottom: 64.5933014354067%; height: 0;">
            <iframe src="https://www.loom.com/embed/47fcb3ace2a24164824b8c78fb28673c?sid=de0d24db-29f7-4fc7-adcc-7fa82fcc31f3" 
                    frameborder="0" 
                    webkitallowfullscreen 
                    mozallowfullscreen 
                    allowfullscreen 
                    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
            </iframe>
        </div>
        """,
        height=500  # Adjust this value to control the height of the embedded video
    )

    modal = Modal(key="result_modal", title="Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Left")
        left_image = st.file_uploader("upload clear photos of your acne", key="left")

    with col2:
        st.subheader("Right")
        right_image = st.file_uploader("(we delete all photos within 24 hours)", key="right")

    routine = routine_entry_form()

    email = st.text_input("**Enter your email**: (optional) be the first to know about new features and get personal insights straight in your inbox", help="If you provide your email we'll keep track of your progress for you.")

    share_photos = st.checkbox(
    "Yes, store my photos to help track my progress over time."
    )
    st.write(
    "By choosing this option, your photos will be securely stored and accessible only to you, "
    "allowing you to monitor your skin's progress over time. You can request deletion of your photos at any time."
    )

    if st.button("Submit"):
        if email and (not is_valid_email(email)):
            st.error("Please enter a valid email address.")
        elif not (left_image or right_image):
            st.error("Please upload at least one image before submitting.")
        else:
            model = load_model()
            has_image = bool(left_image or right_image)
            has_routine = bool(routine.strip())
            
            track_user_interaction(database, has_image, has_routine)
            
            if email:
                all_users(database, email)
                save_user_consent(database, email, share_photos)
                if has_routine:
                    store_routine(database, routine, email)
            
            track_photo_submission(database)

            # Process images
            image_paths = []
            severity_labels = []
            num_blemishes_list = []

            for image, side in zip([left_image, right_image], ["left", "right"]):
                if image:
                    img_path, severity_label, num_blemishes = process_image(image, side, email, model, database)
                    if severity_label is not None and num_blemishes is not None:
                        severity_labels.append(severity_label)
                        num_blemishes_list.append(num_blemishes)
                        image_paths.append(img_path)

            if severity_labels and num_blemishes_list:
                summary_message = present_results(severity_labels, num_blemishes_list)

                if email:
                    email_subject = "Your Acne Assessment Results"
                    email_body = f"Dear user,\n\nThank you for using our acne assessment tool. Here are your personalized results:\n\n{summary_message}\n\nBest regards,\nThe Team"
                    send_email(to_email=email, subject=email_subject, body=email_body, image_paths=image_paths, bcc_email=st.secrets["email"]["bcc_email"])

        # Modal handling
        if st.session_state.get("show_modal", False):
            with modal.container():
                st.markdown(st.session_state["results_message"])
                # Display separate classifications and scores for each side
                st.markdown("### Detailed Scores:")
                if 'left_classification' in st.session_state and 'left_score' in st.session_state:
                    st.markdown(f"**Left Side:** {st.session_state['left_classification']} ({st.session_state['left_score']} blemishes)")
                if 'front_classification' in st.session_state and 'front_score' in st.session_state:
                    st.markdown(f"**Front/Center Side:** {st.session_state['front_classification']} ({st.session_state['front_score']} blemishes)")
                if 'right_classification' in st.session_state and 'right_score' in st.session_state:
                    st.markdown(f"**Right Side:** {st.session_state['right_classification']} ({st.session_state['right_score']} blemishes)")

            if st.button("Close"):
                st.session_state["show_modal"] = False  # Close the modal

    st.subheader("Hi, I'm Andrea 👋")
    st.write("""
    After years of struggling with acne, I developed a tool to track and improve my skin. Now, I’m sharing it to help you find what works for you.    """)
    # Add before and after images
    before_image_url = "https://firebasestorage.googleapis.com/v0/b/loltony.appspot.com/o/users%2F4WWZhb92MKMMjciIIYlr92cr1UM2%2Fentries%2F20240529%2Fimages%2F67283BDC-2C31-4D0A-BA18-2F167E05629E?alt=media&token=a8ca1ded-575a-4e77-94b6-a99b76496a32"
    after_image_url = "https://firebasestorage.googleapis.com/v0/b/loltony.appspot.com/o/users%2F4WWZhb92MKMMjciIIYlr92cr1UM2%2Fentries%2F20240728%2Fimages%2FEEFD7102-B92F-439C-A175-570BEB8F4938?alt=media&token=bdcfd61b-13f7-4f0d-bfb5-25e11affc9e4"

    col1, col2 = st.columns(2)

    with col1:
        st.image(before_image_url, caption="Before (May 29)", width = 300)

    with col2:
        st.image(after_image_url, caption="After (Jul 28)", width = 300)

    st.subheader("Why I Made This")
    st.write("""
    After trying so many of the "solutions" on the market and finding nothing but frustration, I decided to try something new. I would experiemnt on myself and objectively measuring what worked and what didn't. I found a [model](https://arxiv.org/abs/2403.00268) uses the gold-standard [Hayashi scale](https://pubmed.ncbi.nlm.nih.gov/18477223/) developed by researchers at MIT. My hope is that by sharing this with others, we can collectively find solutions that work.

    I have nothing to sell; my goal is to genuinely help others on their journey to clearer skin. If this tool gains traction and proves helpful, we could potentially crowdsource the cure for acne together. Your feedback and experiences are invaluable in this process.

    If you have any questions or need support, feel free to reach out anytime: team@yacne.com.
    """)     

    st.header("Coming Soon…")
    st.write("""
    - Personalized Progress Reports
    - Protocol Assessment
    - Supportive Community Access
    - iOS App
    """)

    st.subheader("References / Information:")
    st.write("""
    If you don't have a routine, here is what I've been doing:

    **Morning Wash:**
    - Salicylic Acid

    **Evening Wash:**
    - Wash with Salicylic Acid and then Benzoyl Peroxide (2.5%)
    
    **Diet and Lifestyle:**
    (baby steps. take small manageable steps)
    - Drink plenty of water.
    - Eat as much as you can: fruits, vegetables, and lean proteins.
    - Eat as little as you can: sugar, simple carbs, processed foods, seed oils.
    - Get at least 7-8 hours of sleep every night.
    - Exercise regularly to maintain overall health.
    - Practice meditation to reduce stress.
    """)

    st.subheader("Find me on the internet:")
    st.write("""
    - [LinkedIn](https://www.linkedin.com/in/earu723/)
    - [YouTube](https://www.youtube.com/@earu723)
    - [TikTok](https://www.tiktok.com/@y.earu)
    - [Instagram](https://www.instagram.com/y.earu/)
    - [Twitter](https://x.com/AndreaR91659141)
    """)

    st.image(image_url, caption='Me as a 4th year BME student', use_column_width=True)

    st.write("""
    Special thanks to Dana, a BASc in Biomedical Engineering, specializing in AI. Andrea's story spoke to me and I wanted to help him share this model with others.
    You can find me [here](https://www.linkedin.com/in/danazarezankova/).
    """)

if __name__ == "__main__":
    main()