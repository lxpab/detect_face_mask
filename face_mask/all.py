import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import random
from multiprocessing import Process
import smbus
from time import sleep
import time
import RPi.GPIO as GPIO

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')

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
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

# 加载训练数据集文件
recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer.read('trainer/trainer.yml')
names = []
warningtime = 0


def facemask():
    global x, y, w, h

    # 准备识别的图片
    def face_detect_demo(img):
        # print(names)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
        # face=face_detector.detectMultiScale(gray)

        for x, y, w, h in face:
            # cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
            # cv2.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=1)
            # 人脸识别
            ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
            # print('标签id:',ids,'置信评分：', confidence)
            try:
                if confidence > 80:
                    global warningtime
                    warningtime += 1
                    if warningtime > 100:
                        # warning()
                        warningtime = 0
                    cv2.putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
                else:
                    cv2.putText(img, str(names[ids - 1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),
                                1)
                # print('bug:',ids)
            except:
                pass

    def name():
        path = 'data/jm/'
        # names = []
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            name1 = str(os.path.split(imagePath)[1].split('.', 2)[1])
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
        img = cv2.flip(img, 1)
        if not ret:
            break
        face_detect_demo(img)

        # Convert Image into gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert image in black and white
        (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow('black_and_white', black_and_white)

        # detect face
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # print(faces)

        # Face prediction for black and white
        faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)
        # print(faces_bw)

        # Detect lips counters
        mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
        # print(mouth_rects)

        if (len(faces) == 0 and len(faces_bw) == 1):
            # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
            img = cv2AddChineseText(img, "没有发现", (123, 123), (255, 255, 255), 45)

        else:
            # Draw rectangle on gace
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

            # Face detected but Lips not detected which means person is wearing mask
            if (len(mouth_rects) == 0):
                img = cv2AddChineseText(img, "感谢你佩戴口罩", (123, 123), (0, 255, 0), 45)
            else:
                for (mx, my, mw, mh) in mouth_rects:
                    try:
                        if (y < my < y + h):
                            # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                            # person is not waring mask
                            img = cv2AddChineseText(img, "请佩戴口罩", (123, 123), (255, 0, 0), 45)

                            # cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                            break
                    except:
                        pass

        try:
            cv2.imshow('Mask Detection', img)
            # cv2.imshow('gray', gray)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
        except:
            pass


def temp():
    colors = [0xFF00, 0x00FF, 0x0FF0, 0xF00F]
    makerobo_pins = (11, 12)  # PIN管脚字典
    GPIO.setmode(GPIO.BOARD)  # 采用实际的物理管脚给GPIO口
    GPIO.setwarnings(False)  # 去除GPIO口警告
    GPIO.setup(makerobo_pins, GPIO.OUT)  # 设置Pin模式为输出模式
    GPIO.output(makerobo_pins, GPIO.LOW)  # 设置Pin管脚为低电平（OV）关闭LED
    p_R = GPIO.PWM(makerobo_pins[0], 2000)  # 设置频率为2KHz
    p_G = GPIO.PWM(makerobo_pins[1], 2000)  # 设置频率为2KHz
    # 初始化占空间比为0（led关闭）
    p_R.start(0)
    p_G.start(0)

    def makerobo_pwm_map(x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def makerobo_set_Color(col):  # 例如：col = 0x1122
        R_val = col >> 8
        G_val = col & 0x00FF
        # 把0-255的范围同比缩小到0-100之间
        R_val = makerobo_pwm_map(R_val, 0, 255, 0, 100)
        G_val = makerobo_pwm_map(R_val, 0, 255, 0, 100)
        p_R.ChangeDutyCycle(R_val)  # 改变占空比
        p_G.ChangeDutyCycle(G_val)  # 改变占空比

    # 调用函数
    def makerobo_loop():
        while True:
            sensor = MLX90614()
            T = sensor.get_obj_temp()
            if T < 28 or T > 48:
                for col in colors:
                    makerobo_set_Color(col)
                    time.sleep(0.1)
                    print("tiwenyichang !!!!!!!!!!!", T)

            else:
                for col in colors:
                    print("tiwenzhengchang", T)
                    makerobo_set_Color(col)
                    time.sleep(2)

    # 释放资源
    def makerobo_destroy():
        p_G.stop()
        p_R.stop()
        GPIO.output(makerobo_pins, GPIO.LOW)
        GPIO.cleanup()

    class MLX90614():

        MLX90614_RAWIR1 = 0x04
        MLX90614_RAWIR2 = 0x05
        MLX90614_TA = 0x06
        MLX90614_TOBJ1 = 0x07
        MLX90614_TOBJ2 = 0x08

        MLX90614_TOMAX = 0x20
        MLX90614_TOMIN = 0x21
        MLX90614_PWMCTRL = 0x22
        MLX90614_TARANGE = 0x23
        MLX90614_EMISS = 0x24
        MLX90614_CONFIG = 0x25
        MLX90614_ADDR = 0x0E
        MLX90614_ID1 = 0x3C
        MLX90614_ID2 = 0x3D
        MLX90614_ID3 = 0x3E
        MLX90614_ID4 = 0x3F

        comm_retries = 5
        comm_sleep_amount = 0.1

        def __init__(self, address=0x5a, bus_num=1):
            self.bus_num = bus_num
            self.address = address
            self.bus = smbus.SMBus(bus=bus_num)

        def read_reg(self, reg_addr):
            err = None
            for i in range(self.comm_retries):
                try:
                    return self.bus.read_word_data(self.address, reg_addr)
                except IOError as e:
                    err = e
                    # "Rate limiting" - sleeping to prevent problems with sensor
                    # when requesting data too quickly
                    sleep(self.comm_sleep_amount)
            # By this time, we made a couple requests and the sensor didn't respond
            # (judging by the fact we haven't returned from this function yet)
            # So let's just re-raise the last IOError we got
            raise err

        def data_to_temp(self, data):
            temp = (data * 0.02 + 3.5) - 273.15
            return temp

        def get_amb_temp(self):
            data = self.read_reg(self.MLX90614_TA)
            return self.data_to_temp(data)

        def get_obj_temp(self):
            data = self.read_reg(self.MLX90614_TOBJ1)
            return self.data_to_temp(data)

    try:
        makerobo_loop()  # 调用循环函数
    except KeyboardInterrupt:  # 当按下Ctrl+C时，将执行destroy()子程序
        makerobo_destroy()  # 释放资源


if __name__ == '__main__':
    p = Process(target=facemask, args=())
    p2 = Process(target=temp, args=())
    p.start()
    p2.start()
