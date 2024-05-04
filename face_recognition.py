
import cv2
import numpy as np
import os 


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("saved_model/")


recognizer.read('saved_model/s_model.yml')

cascadePath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

while True:

    ret, im =cam.read()


    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2,5)

   
    for(x,y,w,h) in faces:


        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

   
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])  

   
        if Id == 1:
            Id = "Mayank {0:.2f}%".format(round(100 - confidence, 2))
        
        elif Id == 2 :
            Id = "Vismay {0:.2f}%".format(round(100 - confidence, 2))
        
        elif Id == 3:
            Id = "Deepanshu {0:.2f}%".format(round(100 - confidence, 2))
        elif Id == 4:
            Id = "Unknowns {0:.2f}%".format(round(100 - confidence, 2))
        else:
            pass

     
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)

  
    cv2.imshow('im',im) 

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()

cv2.destroyAllWindows()
