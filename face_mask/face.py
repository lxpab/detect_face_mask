import cv2
import os
import time
capture = cv2.VideoCapture(0)

name = input("请输入您的姓名首拼：")
num = int(input("请输入您的id(查看jm文件夹)："))
falg = 1
path= "E:\\liujin\\Face-Mask\\data\\jm\\"
if not os.path.exists(path):
    os.makedirs(path)
count = 0
while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame,1)   #镜像操作
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度
    face_detector = cv2.CascadeClassifier(
        'E:/anaconda/envs/huawei/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    face = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
    cv2.imshow("video", frame)
    if len(face) > 0:
        #print("1111111111")
        time.sleep(0.5)
        cv2.imwrite(path + str(num) + "." + name + ".jpg", frame)
        print("success to save" + str(num) + ".jpg")
        print("-------------------")
        num += 1
        count += 1
    #print(key)
    #if count == 20:  #判断是哪一个键按下
    #   break
    key = cv2.waitKey(50)
    #print(key)
    if (key  == ord('q')) or (count == 31):  #判断是哪一个键按下
        break

capture.release()
cv2.destroyAllWindows()

os.system('python face_train.py')
print("------------------我已训练完毕---------------------")
print("------------------开始识别---------------------")
os.system('python detect_face_mask.py')
