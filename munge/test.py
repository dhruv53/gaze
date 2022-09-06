"""
Author : Dhruv Rohatgi
University: Newcastle University

"""


import torch
import mobilemodel
import cv2
import numpy as np
import screeninfo
import random
import dlib
from imutils import face_utils


webcam = cv2.VideoCapture(1)

net = mobilemodel.final()
modelpath = r"./mobile.pt"
net.load_state_dict(torch.load(modelpath))
net.eval()
screen = screeninfo.get_monitors()[0]
width, height = screen.width, screen.height
start=0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    
    check, frames=webcam.read()
    
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    
    for rect in rects:
        if (abs(rect.right() - rect.left()) > (frames.shape[1]/10)) and abs(rect.top() - rect.bottom()) > (frames.shape[0]/10):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
    
    rightEye = shape[lStart:lEnd]
    leftEye = shape[rStart:rEnd]
        
    face = cv2.flip(frames[rect.top():rect.bottom(), rect.left():rect.right()],0)
    
    xmin = rect.right()
    xmax = rect.left()
    ymin = rect.top()
    ymax = rect.bottom()
    
    marginx = 20
    marginy = 20    

    left_eye = frames[leftEye.T[:][1].min()-marginy:leftEye.T[:][1].max()+marginy//2,leftEye.T[:][0].min()-marginx: leftEye.T[:][0].max()+marginx]    
    right_eye = frames[rightEye.T[:][1].min()-marginy:rightEye.T[:][1].max()+marginy//2,rightEye.T[:][0].min()-marginx: rightEye.T[:][0].max()+marginx]
    
    cv2.imshow("face",face)
    
    if left_eye.size!=0:
        
        left_eye=cv2.resize(left_eye,(224,224))
        
        cv2.imwrite('left.jpg',left_eye)
        cv2.imshow("left_eye",left_eye)
        left_eye = (left_eye.transpose(2,0,1)/255.0).astype('float16')
    
        if right_eye.size!=0:
            
            right_eye=cv2.resize(right_eye,(224,224))
            cv2.imshow("right_eye",right_eye)
            cv2.imwrite('right.jpg',right_eye)
            right_eye = (right_eye.transpose(2,0,1)/255.0).astype('float16')
            #cv2.imshow("rightEye",right_eye)

            #cv2.rectangle(resized_image, (rxmin, rymin), (rxmax, rymax) ,(255,0,0), 2)
            face = cv2.resize(face,(224,224))
            cv2.imwrite('face.jpg',face)
            face=face/255.0
            face = face.transpose(2,0,1).astype('float16')
            grid = np.zeros((frames.shape[0],frames.shape[1]))
            cv2.imshow('scraped',frames)
            grid[xmin:xmax+1,ymin:ymax+1]=1
            grid = cv2.resize(grid,(25,25))
            gray= cv2.cvtColor(np.float32(grid*255),cv2.COLOR_GRAY2RGB)
            cv2.imshow("grid_image",gray)
            img = {"left":torch.from_numpy(np.expand_dims(left_eye,axis=0)).type(torch.FloatTensor),
                   "right":torch.from_numpy(np.expand_dims(right_eye,axis=0)).type(torch.FloatTensor),
                   "face":torch.from_numpy(np.expand_dims(face,axis=0)).type(torch.FloatTensor),
                   "cam" :torch.from_numpy(np.expand_dims(np.array([15,30]),axis=0)).type(torch.FloatTensor),
                   "grid":torch.from_numpy(np.expand_dims(grid,axis=0)).type(torch.FloatTensor)}
            
            labels = net( img )
            
            labels=labels.detach().numpy()
            image = np.ones((height, width, 3), dtype=np.float32)
            
            r=random.randint(0,255)
            b=random.randint(0,255)
            g=random.randint(0,255)
            print('labels',labels)
            
            image=cv2.putText(
                               img = image,
                               text = str(labels[0,0])+"    "+str((labels[0,1])),
                               org = (200, 200),
                               fontFace = cv2.FONT_HERSHEY_DUPLEX,
                               fontScale = 1.0,
                               color = (0, 0, 0),
                               thickness = 3
                                             )
            image=cv2.putText(
                               img = image,
                               text = str(labels[0,0])+"    "+str((labels[0,1])),
                               org = (600, 600),
                               fontFace = cv2.FONT_HERSHEY_DUPLEX,
                               fontScale = 1.0,
                               color = (0, 0, 0),
                               thickness = 3
                                             )
            cv2.circle(image, center = (int(labels[0,0]*2.5),int(labels[0,1]*2)), radius =20, color =(b,0,0), thickness=10)
            #cv2.rectangle(image,(x1,y1), (x2,y2), (255,0,0), 2)
            print(width,height)
            print(int(labels[0,0]),int((labels[0,1])))
            window_name = 'projector'
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, image)
            
            
    else:
        continue
    cv2.imshow("winname",frames )
    
    
    if cv2.waitKey(1)==27:
        break
webcam.release()
cv2.destroyAllWindows()

