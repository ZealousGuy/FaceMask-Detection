import numpy as np
import cv2 as c
withMask=np.load('with_mask.npy')
withoutMask=np.load('without_mask.npy')

# RESHAPE to Important DATA
wm=withMask.reshape(len(withMask),50*50*3)
wom=withoutMask.reshape(len(withoutMask),50*50*3)
x=np.r_[wm,wom] # Appended dataset
labels=np.zeros(x.shape[0])
labels[200:]=1.0

#Machine Learning Start Algo Using SVM
# # SVM-Support Vector Machine
# SVC-Support Vector Classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Splitting Training and Test Data Sets
x_train, x_test, y_train, y_test=train_test_split(x,labels,test_size=0.05)
svm=SVC()
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)
accuracy_score(y_test,y_pred) #GENERATING ACCRUACY

#Adding this functionality to the webcam 
haar_data=c.CascadeClassifier('../Face_detection/haarcascade_frontalface_default.xml')
names={0:'Mask',1:'No Mask'}
vid = c.VideoCapture(0)
data=[]
font=c.FONT_HERSHEY_SIMPLEX
while True:
    flag,frame=vid.read()
    if flag:
        faces=haar_data.detectMultiScale(frame)
        for x,y,h,w in faces:
            c.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
            face=frame[y:y+h,x:x+w,:]
            face=c.resize(face,(50,50))
            face=face.reshape(1,-1)
            pred=svm.predict(face)
            n=names[int(pred)]
            c.putText(frame,n,(x,y),font,1,(255,245,200),2)
            print(n)
        c.imshow('res',frame)
        if c.waitKey(1)==27:
            break
vid.release()
c.destroyAllWindows()