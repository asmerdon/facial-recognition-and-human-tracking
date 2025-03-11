### Overview
This repository contains the code for two models: a **facial recognition model** and a **human tracking model**.

#### Facial Recognition Model
This model takes in a selfie and a document image (e.g. passport, ID card, etc.) and returns a similarity score to determine whether the faces in both images belong to the same person.

#### Human Tracking Model
This model takes video input, detects any humans in the footage, outlines them, and applies a blur effect. The blurring was added for an artistic project; however, the core tracking features function independently.

#### Requirements:
To run the facial recognition model, the following dependancies are required:
/*:
  - cv2
  - numpy
  - face_recognition
  - matplotlib
  - deepface
  - 
*/

To run the human tracking model, the following dependancies are required:
/*:
  - cv2
  - numpy
  - ultralytics (import YOLO)
  - moviepy.editor (import VideoFileClip)
*/
