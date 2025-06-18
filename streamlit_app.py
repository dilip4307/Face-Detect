import os
import tempfile
from PIL import Image
import streamlit as st
import cv2

def validate_profile_photo(filepath, allowed_formats=('JPEG', 'PNG'), min_size=(300, 300), max_filesize_mb=5):
    if not os.path.exists(filepath):
        return False, "❌ File does not exist."

    filesize_mb = os.path.getsize(filepath) / (1024 * 1024)
    if filesize_mb > max_filesize_mb:
        return False, f"❌ File is too large: {filesize_mb:.2f} MB (max {max_filesize_mb} MB)."

    try:
        with Image.open(filepath) as img:
            if img.format.upper() not in allowed_formats:
                return False, f"❌ Unsupported image format: {img.format}."
            if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                return False, f"❌ Image dimensions too small: {img.size} (min {min_size})."
    except Exception as e:
        return False, f"❌ Error reading image: {str(e)}"

    return True, "✅ Image is valid."

def contains_exactly_one_face(filepath):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = cv2.imread(filepath)
        if img is None:
            return False, "❌ Failed to load image for face detection."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return False, "❌ No face detected in the image."
        elif len(faces) > 1:
            return False, f"❌ Multiple faces detected ({len(faces)}). Only one face is allowed."
        else:
            return True, "✅ Exactly one face detected."
    except Exception as e:
        return False, f"❌ Face detection error: {str(e)}"

def validate_profile_photo_with_single_face(filepath):
    valid, msg = validate_profile_photo(filepath)
    if not valid:
        return False, msg

    face_valid, face_msg = contains_exactly_one_face(filepath)
    if not face_valid:
        return False, face_msg

    return True, "✅ Image is valid and contains exactly one face."


# ==== Streamlit App ====

st.title("📸 Profile Photo Validator")

uploaded_file = st.file_uploader("Upload a profile photo (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Show uploaded image
    st.image(tmp_path, caption="Uploaded Image", use_column_width=True)

    # Validate photo
    valid, message = validate_profile_photo_with_single_face(tmp_path)
    if valid:
        st.success(message)
    else:
        st.error(message)
