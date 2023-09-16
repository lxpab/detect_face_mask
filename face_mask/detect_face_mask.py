import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import random

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

face_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_mcs_mouth.xml')
eye_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_eye.xml')


# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 90

# User message
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (255, 255, 255)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing MASK"
not_weared_mask = "人"

# Read video
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#加载训练数据集文件
recogizer=cv2.face.LBPHFaceRecognizer_create()
try:
    recogizer.read('trainer/trainer.yml')
except:
    pass
names=[]
warningtime = 0
#准备识别的图片
def face_detect_demo(img):
    #print(names)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为灰度
    face_detector=cv2.CascadeClassifier('E:/anaconda/envs/huawei/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(100,100),(300,300))
    #face=face_detector.detectMultiScale(gray)

    for x,y,w,h in face:
        #cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        #cv2.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=1)
        # 人脸识别
        try:
            ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
            #print('标签id:',ids,'置信评分：', confidence)

            if confidence > 80:
                global warningtime
                warningtime += 1
                if warningtime > 100:
                   #warning()
                   warningtime = 0
                cv2.putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            else:
                cv2.putText(img,str(names[ids-1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        #print('bug:',ids)
        except:
            cv2.putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)

def name():
    path = 'data/jm/'
    #names = []
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name1 = str(os.path.split(imagePath)[1].split('.',2)[1])
       names.append(name1)


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

name()
while True:
    # Get individual frame
    ret, img = cap.read()
    img = cv2.flip(img,1)
    if not ret:
        break
    face_detect_demo(img)

    # Convert Image into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert image in black and white
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    #cv2.imshow('black_and_white', black_and_white)

    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #print(faces)

    # Face prediction for black and white
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)
    #print(faces_bw)

    # Detect lips counters
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
    #print(mouth_rects)

    eye_dect = eye_cascade.detectMultiScale(gray, 1.5, 5)
    print(eye_dect)

    '''
    if(len(faces) == 0 and len(faces_bw) == 0):
        img =cv2AddChineseText(img, "没有发现",  (123, 123), (255, 255, 255), 45)
        print('111111111111')
    '''
    if(len(faces) == 0 and len(eye_dect) == 0):
        # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
        img = cv2AddChineseText(img, "请您靠近摄像头",  (123, 123), (255, 255, 255), 45)
        #print('222222222')
    else:
        # Draw rectangle on gace
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Face detected but Lips not detected which means person is wearing mask



        if(len(mouth_rects) == 0):
            img = cv2AddChineseText(img, "感谢你佩戴口罩",  (123, 123), (0, 255, 0), 45)
        else:
            for (mx, my, mw, mh) in mouth_rects:
                try:
                    if(y < my < y + h):
                        # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                        # person is not waring mask
                        img = cv2AddChineseText(img, "请佩戴口罩",  (123, 123), (255, 0, 0), 45)

                        #cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                        break
                except:
                    pass

    # Show frame with results
    #cv2.putText(img, str(count), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    img = cv2AddChineseText(img, "您的体温为：", (0, 40), (0, 0, 0), 30)
    cv2.imshow('Mask Detection', img)
    #cv2.imshow('gray', gray)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release video
cap.release()
cv2.destroyAllWindows()