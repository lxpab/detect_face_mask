import cv2
import numpy as np
import os
# coding=utf-8
import urllib
import urllib.request
import hashlib
import time

#加载训练数据集文件
recogizer=cv2.face.LBPHFaceRecognizer_create()
recogizer.read('trainer/trainer.yml')
names=[]
warningtime = 0
#准备识别的图片
def face_detect_demo(img):
    start_time = time.time()
    #print(names)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为灰度
    face_detector=cv2.CascadeClassifier('E:/anaconda/envs/huawei/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(100,100),(300,300))
    #face=face_detector.detectMultiScale(gray)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        cv2.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=1)
        # 人脸识别
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
            cv2.putText(img, str(names[ids-1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)
    scaler = 1
    # 在图像上写FPS数值，参数依次为图片、添加的文字、左上角坐标、字体、字体大小、颜色、字体粗细
    img = cv2.putText(img, 'FPS ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler,
                      (0, 0, 255), 1)
    cv2.imshow('result', img)
    #print('bug:',ids)

def name():
    path = 'data/jm/'
    #names = []
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        name1 = str(os.path.split(imagePath)[1].split('.',2)[1])
        names.append(name1)


cap = cv2.VideoCapture(0)
name()
while True:
    flag,frame=cap.read()
    frame = cv2.flip(frame,1)   #镜像操作
    if not flag:
        break
    face_detect_demo(frame)
    key = cv2.waitKey(50)
    #print(key)
    if key  == ord('q'):  #判断是哪一个键按下
        break
cv2.destroyAllWindows()
cap.release()

