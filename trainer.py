import cv2 as cv
import os
import numpy as np
from PIL import Image

recognizer=cv.face.LBPHFaceRecognizer_create();
path='dataset'
def getImageWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceNp)
        IDs.append(ID)
    return np.array(IDs),faces

IDs,faces=getImageWithID(path)
print(IDs)
recognizer.train(faces,IDs)
recognizer.save("recognizer/traindata.yml")
print("Successfully Trained!")
cv.destroyAllWindows()
