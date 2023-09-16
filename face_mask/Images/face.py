#导入模块
import cv2
import os
import time
#摄像头
cap=cv2.VideoCapture(1)
name = input("请输入您的姓名首拼：")
num = int(input("请输入您的id(查看jm文件夹)："))
falg = 1
path= "E:\\liujin\\mycodetest\\opencv\\data\\jm\\"
if not os.path.exists(path):
    os.makedirs(path)
starttime = time.time()
while(cap.isOpened()):#检测是否在开启状态
    ret_flag,Vshow = cap.read()#得到每帧图像
    cv2.imshow("Capture_Test",Vshow)#显示图像
#k = cv2.waitKey(1) & 0xFF#按键判断
#if k == ord('s'):#保存
    cv2.imwrite(path+str(num)+"."+ name +".jpg",Vshow)
    print("success to save"+str(num)+".jpg")
    print("-------------------")
    num += 1
    endtime = time.time()
    if (endtime-starttime) >= 1:
        break

#elif k == ord(' '):#退出
    #break
#释放摄像头
cap.release()
#释放内存
cv2.destroyAllWindows()
#训练
os.system('python face_train.py')
print("------------------我已训练完毕---------------------")
print("------------------开始识别---------------------")
os.system('python face_recognition.py')