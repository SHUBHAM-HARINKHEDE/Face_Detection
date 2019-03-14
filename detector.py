import numpy as np
import cv2 as cv
import sqlite3
conn = sqlite3.connect('database/test.db')

def getName(id):
    query='SELECT NAME FROM PERSON WHERE rowid='+str(id)
    cursor=conn.execute(query)
    for row in cursor:
        return row[0]
    
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv.CascadeClassifier(cv.data.haarcascades +  'haarcascade_eye.xml')
#smile_cascade = cv.CascadeClassifier(cv.data.haarcascades +  'haarcascade_smile.xml')

cam = cv.VideoCapture(0)
recognizer=cv.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/traindata.yml')

fontface = cv.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)

while True:
    ret,img=cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    '''smile=smile_cascade.detectMultiScale(gray, 2.6, 5)
    for (x,y,w,h) in smile:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)'''
    
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<80):
            name=getName(id)
            cv.putText(img,name+"("+str(conf)+")", (x,y+h+25), fontface, fontscale, fontcolor)
            print(id)
        else:
            name="unknown"
            cv.putText(img,name+"("+str(conf)+")", (x,y+h+25), fontface, fontscale, fontcolor)
        
    cv.imshow('Face',img)
    if(cv.waitKey(1)==ord('q')):
        break
    
conn.close()       
cam.release()
cv.destroyAllWindows()


