from maix import camera, display
import sys

# 初始化相机和显示器
cam = camera.Camera(1920, 1080)
disp = display.Display()
image_count = 0  # 图片计数器

while True:
    # 读取相机图像
    img = cam.read()
    
    # 显示图像
    disp.show(img)
