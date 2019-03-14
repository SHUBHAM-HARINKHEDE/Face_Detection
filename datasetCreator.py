import numpy as np
import cv2 as cv
import sqlite3
conn = sqlite3.connect('database/test.db')
conn.execute('''CREATE TABLE IF NOT EXISTS PERSON(NAME  TEXT    NOT NULL UNIQUE);''')
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv.CascadeClassifier(cv.data.haarcascades +  'haarcascade_eye.xml')
cam = cv.VideoCapture(0)
name =input("Enter Name:")
try:
    conn.execute('INSERT INTO PERSON(NAME) VALUES (?)',(name,))
except:
    print("UPDATING...")
conn.commit()
id=None
for row in conn.execute('SELECT rowid FROM PERSON WHERE NAME=?',(name,)):
    id=row[0]
    
sample=0
while True:
    ret,img=cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        sample+=1
        cv.imwrite('dataset/User.'+str(id)+"."+str(sample)+".jpg",gray[y:y+h,x:x+w])
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
           # cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv.waitKey(100)
    cv.imshow('Face',img)
    cv.waitKey(1)
    if(sample>30):
        break
for row in conn.execute('SELECT * FROM PERSON'):
    print(row)

conn.close()  
cam.release()
cv.destroyAllWindows()
print("Successfully created")
