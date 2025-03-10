import cv2
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
from deepface import DeepFace  

def enhance_image(image_path):
    image = cv2.imread(image_path)  # load image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    image = cv2.equalizeHist(image)  # improve contrast
    enhanced_path = "enhanced_" + image_path  # set new filename
    cv2.imwrite(enhanced_path, image)  # save enhanced image
    return enhanced_path  # return new path

def sharpen_image(image_path):
    image = cv2.imread(image_path)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # sharpen filter
    sharpened = cv2.filter2D(image, -1, kernel)
    sharpened_path = "sharpened_" + image_path
    cv2.imwrite(sharpened_path, sharpened)
    return sharpened_path

def extract_face(image_path, target_size=(160, 160), save_path=None):
    image = face_recognition.load_image_file(image_path)  # load image
    face_locations = face_recognition.face_locations(image, model="hog")  # detect faces

    if not face_locations:
        raise ValueError(f"no face detected in {image_path}")  # error if no faces found

    if len(face_locations) > 1:
        face_locations = sorted(face_locations, key=lambda x: x[2], reverse=True)  # sort faces by vertical position (to account for passports vs IDs)

    top, right, bottom, left = face_locations[0]  # select first (or lowest) face
    face_image = image[top:bottom, left:right]  # crop face
    face_image = cv2.resize(face_image, target_size)  # resize face

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))  # save face if path provided

    return face_image  # return cropped face

passport_path = sharpen_image(enhance_image("passport.jpg"))  # enhance and sharpen passport image
passport_face = extract_face(passport_path, save_path="passport_cropped.jpg")  # extract passport face
selfie_face = extract_face("selfie.jpg", save_path="selfie_cropped.jpg")  # extract selfie face

passport_image = face_recognition.load_image_file("passport.jpg")  # reload passport image
selfie_image = face_recognition.load_image_file("selfie.jpg")  # reload selfie image

passport_faces = face_recognition.face_locations(passport_image, model="hog")  # detect faces in passport
selfie_faces = face_recognition.face_locations(selfie_image, model="hog")  # detect faces in selfie

passport_copy = passport_image.copy()  # copy image for drawing
selfie_copy = selfie_image.copy()  # copy image for drawing

for (top,right,bottom,left) in passport_faces:
    cv2.rectangle(passport_copy,(left,top),(right,bottom),(255,0,0),2)  # draw bounding box

for (top,right,bottom,left) in selfie_faces:
    cv2.rectangle(selfie_copy,(left,top),(right,bottom),(255,0,0),2)  # draw bounding box

passport_copy = cv2.cvtColor(passport_copy,cv2.COLOR_BGR2RGB)  # convert for display
selfie_copy = cv2.cvtColor(selfie_copy,cv2.COLOR_BGR2RGB)  # convert for display

fig,ax = plt.subplots(1,2,figsize=(12,6)) 
ax[0].imshow(passport_copy)  # show passport image
ax[0].set_title("Detected Face in Passport")  
ax[1].imshow(selfie_copy)  # show selfie image
ax[1].set_title("Detected Face in Selfie")  
plt.show()  # display images

passport_face = face_recognition.load_image_file("passport_cropped.jpg")  # load cropped passport face
selfie_face = face_recognition.load_image_file("selfie_cropped.jpg")  # load cropped selfie face

passport_encoding = face_recognition.face_encodings(passport_face)  # encode passport face
selfie_encoding = face_recognition.face_encodings(selfie_face)  # encode selfie face

if not passport_encoding or not selfie_encoding:
    raise ValueError("could not generate face embeddings. make sure both images contain a clear face.")  # error if encoding fails

passport_encoding = passport_encoding[0]  # get first encoding
selfie_encoding = selfie_encoding[0]  # get first encoding

match_result = face_recognition.compare_faces([passport_encoding], selfie_encoding, tolerance=0.6)  # compare faces
similarity_score = face_recognition.face_distance([passport_encoding], selfie_encoding)[0]  # calculate similarity

print("=== face recognition results ===")
print("match result:", match_result[0])  # print match status
print(f"similarity score: {1 - similarity_score:.2f}")  # print similarity percentage

if not match_result[0] or (1 - similarity_score) < 0.55:
    print("\n running deepface for more robust verification...")  # fallback to deepface
    deepface_result = DeepFace.verify("passport_cropped.jpg", "selfie_cropped.jpg", model_name="VGG-Face", enforce_detection=False, distance_metric="cosine", align=True)

    print("\n=== deepface verification results ===")
    print("match result:", deepface_result["verified"])  # print deepface result
    print(f"distance: {deepface_result['distance']:.4f} (threshold: 0.6000)")  # print deepface distance

    if deepface_result["distance"] < 0.60:
        print("\n deepface match: true (better for aging)")  # print if deepface matches
    else:
        print("\n deepface match: false (still different)")  # print if deepface fails