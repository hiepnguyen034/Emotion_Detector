import cv2, os
import numpy as np
from PIL import Image

cascadePath = "C:\\Users\\Hiep Nguyen\\Desktop\\Emotion_Detector\\data\\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    images = []
    labels = []
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        emos = os.path.split(image_path)[1].split(".")[1]
        faces = faceCascade.detectMultiScale(image)

        for (height, weight, w, h) in faces:
            images.append(image[weight: weight + 151, height: height + 151]) #to keep images' shapes consistent i.e: each image's shape is (151,151)
            labels.append(emos)
            cv2.imshow("face images", image[weight: weight + 151, height: height + 151])
            cv2.waitKey(50)


    return images, labels


# def get_emotion(labels):
#     for i in range(len(labels)):
#         if labels[i]=='happy' or labels[i]=='wink':
#             labels[i]=2
#         elif labels[i]=='surprised':
#             labels[i]=3
#         elif labels[i]=='sad' or labels[i]=='sleepy':
#             labels[i]=0
#         else:
#             labels[i]=1
#     return labels


def get_emotion(labels):

    emotion_table={'happy':2,
               'wink':2,
               'surprised':3,
               'sad':0,
               'sleepy':0}

    return [1 if label not in emotion_table else emotion_table[label] for label in labels]
