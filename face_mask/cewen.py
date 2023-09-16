import smbus
from time import sleep
import time
import RPi.GPIO as GPIO
colors = [0xFF00,0x00FF,0x0FF0,0xF00F]
makerobo_pins = (11,12)  #PIN管脚字典
GPIO.setmode(GPIO.BOARD)   #采用实际的物理管脚给GPIO口
GPIO.setwarnings(False)    #去除GPIO口警告
GPIO.setup(makerobo_pins, GPIO.OUT)     #设置Pin模式为输出模式
GPIO.output(makerobo_pins, GPIO.LOW)     #设置Pin管脚为低电平（OV）关闭LED
p_R = GPIO.PWM(makerobo_pins[0], 2000)      #设置频率为2KHz
p_G = GPIO.PWM(makerobo_pins[1], 2000)      #设置频率为2KHz
#初始化占空间比为0（led关闭）
p_R.start(0)
p_G.start(0)
def makerobo_pwm_map(x, in_min, in_max, out_min, out_max):
    return(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
def makerobo_set_Color(col):    #例如：col = 0x1122
    R_val = col >> 8
    G_val = col & 0x00FF
    #把0-255的范围同比缩小到0-100之间
    R_val = makerobo_pwm_map(R_val, 0, 255, 0, 100)
    G_val = makerobo_pwm_map(R_val, 0, 255, 0, 100)
    p_R.ChangeDutyCycle(R_val)      #改变占空比
    p_G.ChangeDutyCycle(G_val)      #改变占空比
#调用函数
def makerobo_loop():
    while True:
        sensor = MLX90614()
        T = sensor.get_obj_temp()
        if T < 28 or T > 38:
            for col in colors:
                makerobo_set_Color(col)
                time.sleep(0.1)
                print("tiwenyichang !!!!!!!!!!!", T)

        else:
            for col in colors:
                print("tiwenzhengchang", T)
                makerobo_set_Color(col)
                time.sleep(2)
#释放资源
def makerobo_destroy():
    p_G.stop()
    p_R.stop()
    GPIO.output(makerobo_pins,GPIO.LOW)
    GPIO.cleanup()
class MLX90614():

    MLX90614_RAWIR1=0x04
    MLX90614_RAWIR2=0x05
    MLX90614_TA=0x06
    MLX90614_TOBJ1=0x07
    MLX90614_TOBJ2=0x08

    MLX90614_TOMAX=0x20
    MLX90614_TOMIN=0x21
    MLX90614_PWMCTRL=0x22
    MLX90614_TARANGE=0x23
    MLX90614_EMISS=0x24
    MLX90614_CONFIG=0x25
    MLX90614_ADDR=0x0E
    MLX90614_ID1=0x3C
    MLX90614_ID2=0x3D
    MLX90614_ID3=0x3E
    MLX90614_ID4=0x3F

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
                #"Rate limiting" - sleeping to prevent problems with sensor
                #when requesting data too quickly
                sleep(self.comm_sleep_amount)
        #By this time, we made a couple requests and the sensor didn't respond
        #(judging by the fact we haven't returned from this function yet)
        #So let's just re-raise the last IOError we got
        raise err

    def data_to_temp(self, data):
        temp = (data*0.02+3.5) - 273.15
        return temp

    def get_amb_temp(self):
        data = self.read_reg(self.MLX90614_TA)
        return self.data_to_temp(data)
    def get_obj_temp(self):
        data = self.read_reg(self.MLX90614_TOBJ1)
        return self.data_to_temp(data)

    try:
        makerobo_loop() #调用循环函数
    except KeyboardInterrupt: #当按下Ctrl+C时，将执行destroy()子程序
        makerobo_destroy() #释放资源