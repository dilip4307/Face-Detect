import os
from PIL import Image
import cv2

def validate_profile_photo(filepath, allowed_formats=('JPEG', 'PNG'), min_size=(300, 300), max_filesize_mb=5):
    # Check file existence
    if not os.path.exists(filepath):
        return False, "❌ File does not exist."

    # Check file size
    filesize_mb = os.path.getsize(filepath) / (1024 * 1024)
    if filesize_mb > max_filesize_mb:
        return False, f"❌ File is too large: {filesize_mb:.2f} MB (max {max_filesize_mb} MB)."

    # Validate image format and dimensions
    try:
        with Image.open(filepath) as img:
            if img.format.upper() not in allowed_formats:
                return False, f"❌ Unsupported image format: {img.format}."
            if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                return False, f"❌ Image dimensions too small: {img.size} (min {min_size})."
    except Exception as e:
        return False, f"❌ Error reading image: {str(e)}"

    return True, "✅ Image is valid."


def contains_face(filepath):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = cv2.imread(filepath)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return len(faces) > 0
    except Exception as e:
        print(f"Error in face detection: {e}")
        return False


def validate_profile_photo_with_face(filepath):
    valid, msg = validate_profile_photo(filepath)
    if not valid:
        return valid, msg

    if not contains_face(filepath):
        return False, "❌ No face detected in the image."

    return True, "✅ Image is valid and contains a face."


# ====== 🧪 Test the validator ======
if __name__ == "__main__":
    filepath = input("Enter the path to the profile photo: ").strip()

    # With face detection
    valid, message = validate_profile_photo_with_face(filepath)

    # If you only want basic validation without face detection:
    # valid, message = validate_profile_photo(filepath)

    print(message)
