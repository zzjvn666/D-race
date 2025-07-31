from maix import touchscreen, app, time, display, image, camera
import cv2
import json
import os

# 配置文件路径
CONFIG_FILE = "./BGR.json"

# 初始化颜色变量（默认值）
COLOR_B = 32
COLOR_G = 77
COLOR_R = 77

def load_config():
    """从文件加载颜色配置"""
    global COLOR_B, COLOR_G, COLOR_R
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # 验证配置是否完整
                if all(key in config for key in ['B', 'G', 'R']):
                    COLOR_B = max(0, min(255, config['B']))
                    COLOR_G = max(0, min(255, config['G']))
                    COLOR_R = max(0, min(255, config['R']))
                    print(f"Loaded config: B={COLOR_B}, G={COLOR_G}, R={COLOR_R}")
                else:
                    print("Invalid config file, using defaults")
        else:
            print("No config file found, using defaults")
    except Exception as e:
        print(f"Error loading config: {e}, using defaults")

def save_config():
    """保存颜色配置到文件"""
    try:
        config = {
            'B': COLOR_B,
            'G': COLOR_G,
            'R': COLOR_R
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        print(f"Saved config: B={COLOR_B}, G={COLOR_G}, R={COLOR_R}")
    except Exception as e:
        print(f"Error saving config: {e}")

# 加载配置（程序启动时）
load_config()

# 初始化设备
ts = touchscreen.TouchScreen()
disp = display.Display()
cam = camera.Camera(1920, 1080, image.Format.FMT_BGR888)

# 创建图像缓冲区
ui_img = image.Image(disp.width(), disp.height())

# 按钮尺寸和布局参数
btn_width = 100
btn_height = 100
margin = 10
start_y = 10  # 顶部起始位置

# 定义按钮位置 - [x, y, width, height, 关联变量, 增减标志]
buttons = [
    # 蓝色控制按钮
    [margin, start_y, btn_width, btn_height, 'B', 1],          # 蓝色加
    [margin + btn_width + margin, start_y, btn_width, btn_height, 'B', -1],  # 蓝色减
    
    # 绿色控制按钮
    [margin, start_y + btn_height + margin, btn_width, btn_height, 'G', 1],  # 绿色加
    [margin + btn_width + margin, start_y + btn_height + margin, btn_width, btn_height, 'G', -1],  # 绿色减
    
    # 红色控制按钮
    [margin, start_y + 2*(btn_height + margin), btn_width, btn_height, 'R', 1],  # 红色加
    [margin + btn_width + margin, start_y + 2*(btn_height + margin), btn_width, btn_height, 'R', -1],  # 红色减
    
    # 退出按钮
    [disp.width() - btn_width - margin, start_y, btn_width, btn_height, 'EXIT', 0]  # 退出按钮
]

def is_in_button(x, y, btn):
    """检查触摸点是否在按钮区域内"""
    return (x > btn[0] and x < btn[0] + btn[2] and 
            y > btn[1] and y < btn[1] + btn[3])

def draw_ui(processed_img):
    """绘制简洁的用户界面"""
    # 将处理后的图像缩放到屏幕尺寸
    scaled_img = processed_img.resize(ui_img.width(), ui_img.height())
    # 绘制背景图像
    ui_img.draw_image(0, 0, scaled_img)
    
    # 绘制按钮
    for btn in buttons:
        x, y, w, h, var, delta = btn
        
        # 绘制按钮背景（使用高对比度颜色便于便于识别）
        if var == 'EXIT':
            btn_color = image.COLOR_RED
        else:
            btn_color = image.COLOR_WHITE
        
        ui_img.draw_rect(x, y, w, h, btn_color, -1)
        
        # 设置按钮文本
        if var == 'EXIT':
            text = "Exit"
        else:
            text = "+10" if delta == 1 else "-10"
        
        # 计算文本位置（居中）
        text_size = image.string_size(text)
        text_x = x + (w - text_size.width()) // 2
        text_y = y + (h - text_size.height()) // 2
        
        # 绘制按钮文本（深色色文字确保在白色按钮上可见）
        text_color = image.COLOR_BLACK if var != 'EXIT' else image.COLOR_WHITE
        ui_img.draw_string(text_x, text_y, text, text_color)
    
    # 显示当前颜色值（使用黑色背景白色文字确保可见）
    values_x = margin + btn_width * 2 + margin * 2
    values_y = start_y + 5
    
    # 绘制数值背景（小范围黑色背景确保文字清晰）
    ui_img.draw_rect(values_x - 5, start_y, 100, 3*(btn_height + margin) - 5, image.COLOR_BLACK, -1)
    
    # 绘制数值文字
    ui_img.draw_string(values_x, values_y, f"B: {COLOR_B}", image.COLOR_BLUE)
    ui_img.draw_string(values_x, values_y + btn_height + margin, f"G: {COLOR_G}", image.COLOR_GREEN)
    ui_img.draw_string(values_x, values_y + 2*(btn_height + margin), f"R: {COLOR_R}", image.COLOR_RED)

def update_value(var, delta):
    """更新颜色变量的值，确保在0-255范围内"""
    global COLOR_B, COLOR_G, COLOR_R
    
    if var == 'B':
        COLOR_B = max(0, min(255, COLOR_B + delta * 10))
    elif var == 'G':
        COLOR_G = max(0, min(255, COLOR_G + delta * 10))
    elif var == 'R':
        COLOR_R = max(0, min(255, COLOR_R + delta * 10))

# 主循环
try:
    while not app.need_exit():
        # 读取摄像头图像
        img = cam.read()
        if img is None:
            time.sleep_ms(10)
            continue
        
        # 转换为OpenCV格式进行处理
        cv_img = image.image2cv(img, ensure_bgr=False, copy=False)
        
        # 使用当前颜色值作为阈值进行过滤
        filtered = cv2.inRange(cv_img, (0, 0, 0), (COLOR_B, COLOR_G, COLOR_R))
        
        # 转换回maix图像格式
        processed_img = image.cv2image(filtered, bgr=False, copy=False)
        
        # 读取触摸信息
        x, y, pressed = ts.read()
        
        # 处理触摸事件
        if pressed:
            for btn in buttons:
                if is_in_button(x, y, btn):
                    var, delta = btn[4], btn[5]
                    if var == 'EXIT':
                        app.set_exit_flag(True)
                    else:
                        update_value(var, delta)
                    time.sleep_ms(100)  # 防抖动
        
        # 绘制界面并显示
        draw_ui(processed_img)
        disp.show(ui_img)
        
        time.sleep_ms(50)
finally:
    # 程序退出时保存配置
    save_config()
    print("Program exited, config saved")
    