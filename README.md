# Customer Retention for eCommerce using Gaze Tracking
Knowing the direction in which a person is looking can shed light on their intentions and provide clues as to what they are interested in, making gaze estimate an invaluable resource. Many computer vision problems, such as gaze estimation, have benefited greatly from the recent developments in deep learning algorithms. The development of deep learning algorithms for gaze estimation, however, is hindered by a lack of standard practises. Lack of real-world application is also a factor discouraging research in this area. This repository provides an implementation of Gaze Tracking along with the research work in the same domain.
## Directory
```
│   README.md
└───assets
│   └───main
│   └───supplimentery        
│   
└───data
│   └───source.txt
└───docs
│   └───info.txt
└───graph
│   
└───munge
│   └───mobile.pt
│   └───mobilemodel.py
│   └───test.py
│
└───presentation
│   └───Gaze Tracking Presentation.pdf
│
└───report
    └───Gaze Tracking.pdf
```

## Setup and Pre-requisits

Step 1: Clone the github project.<br/>
Step 2: Download from https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat and save in project munge folder.<br/>
Step 3: Install the required libraries:<br/>
1. torch
2. mobilemodel
3. cv2
4. numpy as np
5. screeninfo
6. random
7. dlib
8. imutils<br/><br/>
Step 4: Run the script
