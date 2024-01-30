# =============================================================================
# import cv2
# NEW = r'C:\Users\cheru\OneDrive\Pictures\Screenshots\Screenshot (107).png'
# IMG = cv2.imread(NEW, 1)
# cv2.imshow('My IMAGE', IMG)
# cv2.waitKey(0)
# =============================================================================
import cv2
from deepface import DeepFace


def detect_faces(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert BGR image to RGB (OpenCV uses BGR by default)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(rgb_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image with faces
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the bounding boxes of detected faces
    return faces


def predict_gender_and_age(image_path):
    # Load the image for gender and age prediction
    img = DeepFace.analyze(image_path)

    # Extract gender and age predictions
    gender = img['gender']
    age = img['age']

    return gender, age


if __name__ == "__main__":
    # Example usage
    image_path = r'C:\Users\cheru\OneDrive\Desktop\ian.jpg'

    # Detect faces and predict gender and age
    faces = detect_faces(image_path)
    for idx, (x, y, w, h) in enumerate(faces):
        gender, age = predict_gender_and_age(image_path)
        print(f"Face {idx + 1}: Gender - {gender}, Age - {age}")