

import cv2 as c
img = c.imread('../photos/face test.jpeg')
haar_data=c.CascadeClassifier('../Face_detection/haarcascade_frontalface_default.xml')

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

names={0:'Mask',1:'No Mask'}
data=[]
font=c.FONT_HERSHEY_SIMPLEX


while True:
    faces=haar_data.detectMultiScale(img)
    for x,y,h,w in faces:
        c.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)

        face=img[y:y+h,x:x+w,:]
        face=c.resize(face,(50,50))
        face=face.reshape(1,-1)
        pred=svm.predict(face)
        n=names[int(pred)]

        c.putText(img,n,(x,y),font,1,(255,0,0),1)
        print(n)

    c.imshow('res',img)
    if c.waitKey(1)==27:
        break
c.destroyAllWindows()