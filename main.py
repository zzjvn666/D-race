from PIL import Image, ImageDraw, ImageFont

def cv2_putText_Chinese(img, text, position, font_size, color):
    """在OpenCV图像上绘制中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simsun.ttc", font_size)  # Windows系统自带宋体
    except:
        font = ImageFont.truetype("arialuni.ttf", font_size)  # 尝试其他字体
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
def _find_shape_1(gray_img, top_left, bottom_right):
    """
    输入：
        gray_img：整张灰度图（单通道，0-255）
        top_left：内轮廓左上角 (x,y)
        bottom_right：内轮廓右下角 (x,y)
    处理：
        1. 裁剪出ROI区域
        2. 识别每一个白色区域顶点的角度（内部为白色的角）
        3. 每一步显示图像
    """
    # 1. 裁剪ROI区域
    x1, y1 = top_left
    x2, y2 = bottom_right
    # roi = gray_img[y1+10:y2-10, x1+10:x2-10]
    roi = gray_img[y1 + 0:y2 - 0, x1 + 0:x2 - 0]
    # 显示ROI区域
    display_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    display_img = cv2_putText_Chinese(display_img, "1. ROI区域", (10, 30), 20, (0, 0, 255))
    # cv2.imshow('处理过程', display_img)
    # cv2.waitKey(0)

    # 2. 使用大津法自动确定阈值并提取白色区域
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 显示二值化结果
    display_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    display_img = cv2_putText_Chinese(display_img, "2. 白色区域提取", (10, 30), 20, (0, 0, 255))
    # cv2.imshow('处理过程', display_img)
    # cv2.waitKey(0)

    # 查找白色区域轮廓（注意使用RETR_EXTERNAL只检测外部轮廓）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        # 创建带轮廓显示的图像
        contour_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)

        # 多边形近似（减少顶点数量）
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 标记所有顶点
        for point in approx:
            cv2.circle(contour_img, tuple(point[0]), 5, (0, 0, 255), -1)

        # 显示轮廓和顶点
        display_img = contour_img.copy()
        display_img = cv2_putText_Chinese(display_img, f"3.{i + 1} 轮廓及顶点", (10, 30), 20, (0, 0, 255))
        # cv2.imshow('处理过程', display_img)
        # cv2.waitKey(0)

        # 计算每个顶点的内角（内部为白色的角）
        angles = []
        n = len(approx)
        for j in range(n):
            # 获取三个连续点（当前顶点和前后各一个点）
            prev_point = approx[(j - 1) % n][0]
            curr_point = approx[j][0]
            next_point = approx[(j + 1) % n][0]

            # 计算两个向量
            vec1 = prev_point - curr_point
            vec2 = next_point - curr_point

            # 计算向量夹角（角度制）
            angle = np.degrees(np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0]))
            angle = angle % 360
            if angle > 180:
                angle = 360 - angle
            angles.append(angle)

            # 在图像上标注角度
            text = f"{angle:.1f}°"
            contour_img = cv2.putText(contour_img, text, tuple(curr_point),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 显示角度计算结果
        display_img = contour_img.copy()
        display_img = cv2_putText_Chinese(display_img, f"3.{i + 1} 顶点角度", (10, 30), 20, (0, 0, 255))
        # cv2.imshow('处理过程', display_img)
        # cv2.waitKey(0)

        print(f"白色区域{i + 1}的顶点角度:", angles)

    cv2.destroyAllWindows()
def angle_between(p1, p2, p3):
    """
    计算p2为顶点，p1-p2-p3之间的夹角（单位：度）
    """
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    angle = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8), -1.0, 1.0))
    return np.degrees(angle)

import cv2
import numpy as np
from tools import *
def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    angle = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8), -1.0, 1.0))
    return np.degrees(angle)
def is_fully_filled(binary_img, contour):
    """
    判断轮廓内部是否全部为白色：
    1. 创建空掩膜
    2. 填充该轮廓
    3. 对比轮廓内区域的原图像和填充图像
    """
    mask = np.zeros_like(binary_img)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    filled_pixels = cv2.bitwise_and(binary_img, binary_img, mask=mask)

    # 白色占比接近100%（容忍1%噪点）
    white_area = cv2.countNonZero(filled_pixels)
    mask_area = cv2.countNonZero(mask)
    if mask_area == 0:
        return False
    fill_ratio = white_area / mask_area
    return fill_ratio > 0.7
def dynamic_threshold(roi, method='otsu', thresh_min=160, thresh_max=255):
    if method == 'otsu':
        otsu_val, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        val = np.clip(otsu_val, thresh_min, thresh_max)
        _, binary = cv2.threshold(roi, val, 255, cv2.THRESH_BINARY)
    elif method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=21, C=5
        )
    else:
        raise ValueError("Unknown threshold method")
    return binary


import cv2
import numpy as np

def detect_outer_black_border(image, draw_result=True, lower_threshold=0, upper_threshold=255, debug=False, 
                              COLOR_B=80, COLOR_G=70, COLOR_R=60):
    def resize_to_fit(img, target_size=(800, 600)):
        """将图像按比例缩放以适应目标大小"""
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        return resized

    # 1. 提取黑色区域（基于BGR阈值）
    gray = cv2.GaussianBlur(cv2.inRange(image, (0, 0, 0), (COLOR_B, COLOR_G, COLOR_R)), (15, 15), 3)
    gray = cv2.bitwise_not(gray)  # 反转：黑色区域变为白色，便于后续处理

    # 2. 二值化（反转颜色）
    _, binary = cv2.threshold(gray, lower_threshold, upper_threshold, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if debug:
        cv2.imshow("Step 1 - Binary Inverted", resize_to_fit(binary))
        cv2.waitKey(0)

    # 3. 定义感兴趣区域（ROI）
    x1, y1 = 952, 153
    x2, y2 = 1245, 593
    # 确保坐标有效性（左上角 < 右下角，且在图像范围内）
    h, w = binary.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    binary_roi = binary[y1:y2, x1:x2]  # 裁剪ROI

    # 形态学操作（开运算：去除小噪声）
    kernel = np.ones((6, 6), np.uint8)
    binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel, iterations=1)
    if debug:
        cv2.imshow("Step 2 - Morphology (ROI)", resize_to_fit(binary_roi))
        cv2.waitKey(0)

    # 4. 查找轮廓及层级关系
    contours, hierarchy = cv2.findContours(binary_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        print("未检测到轮廓。")
        return {
            'contour': None,
            'outer_points': None,
            'inner_points': None,
            'binary_image': binary_roi
        }
    hierarchy = hierarchy[0]  # 层级结构：[Next, Previous, First_Child, Parent]

    # 显示所有轮廓（调试用）
    if debug:
        all_contours_img = np.zeros_like(binary_roi)
        cv2.drawContours(all_contours_img, contours, -1, (255, 255, 255), 2)
        cv2.imshow("Step 3 - All Contours", resize_to_fit(all_contours_img))
        cv2.waitKey(0)

    # 5. 筛选有效外轮廓（无父轮廓，且面积符合要求）
    image_area = binary_roi.shape[0] * binary_roi.shape[1]
    candidate_outer = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # 外轮廓条件：无父轮廓（hierarchy[i][3] == -1），且面积在合理范围
        if hierarchy[i][3] == -1 and (image_area * 0.005 <= area <= image_area * 0.8):
            candidate_outer.append((i, cnt, area))

    if not candidate_outer:
        print("未检测到有效外轮廓。")
        return {
            'contour': None,
            'outer_points': None,
            'inner_points': None,
            'binary_image': binary_roi
        }

    # 选择面积最大的外轮廓（最可能的目标外边框）
    best_outer_idx, best_outer, best_outer_area = max(candidate_outer, key=lambda x: x[2])

    # 6. 筛选外轮廓的直接子轮廓作为内轮廓（核心修改）
    best_inner = None
    # 遍历所有轮廓，寻找外轮廓的直接子轮廓（父轮廓索引为外轮廓索引）
    for i, cnt in enumerate(contours):
        # 内轮廓条件：父轮廓是最佳外轮廓，且面积小于外轮廓
        if hierarchy[i][3] == best_outer_idx:
            inner_area = cv2.contourArea(cnt)
            if inner_area < best_outer_area * 0.9:  # 确保内轮廓在外侧内
                best_inner = cnt
                break  # 取第一个符合条件的直接子轮廓

    # 7. 提取外轮廓和内轮廓的角点（转换为原图坐标）
    outer_top_left = outer_bottom_right = None
    inner_top_left = inner_bottom_right = None

    # 外轮廓角点
    if best_outer is not None:
        x, y, w, h = cv2.boundingRect(best_outer)
        outer_top_left = (x + x1, y + y1)  # 转换回原图坐标（ROI偏移补偿）
        outer_bottom_right = (x + w + x1, y + h + y1)

    # 内轮廓角点（仅当存在有效内轮廓时）
    if best_inner is not None:
        x, y, w, h = cv2.boundingRect(best_inner)
        inner_top_left = (x + x1, y + y1)
        inner_bottom_right = (x + w + x1, y + h + y1)

    # 8. 调试：显示最终结果
    if debug:
        # 在ROI上绘制外轮廓（蓝色）和内轮廓（绿色）
        final_roi_img = cv2.cvtColor(binary_roi.copy(), cv2.COLOR_GRAY2BGR)
        if best_outer is not None:
            cv2.drawContours(final_roi_img, [best_outer], -1, (255, 0, 0), 3)
        if best_inner is not None:
            cv2.drawContours(final_roi_img, [best_inner], -1, (0, 255, 0), 3)
        
        # 绘制角点（红色）
        if outer_top_left and outer_bottom_right:
            cv2.circle(final_roi_img, (outer_top_left[0]-x1, outer_top_left[1]-y1), 5, (0,0,255), -1)
            cv2.circle(final_roi_img, (outer_bottom_right[0]-x1, outer_bottom_right[1]-y1), 5, (0,0,255), -1)
        if inner_top_left and inner_bottom_right:
            cv2.circle(final_roi_img, (inner_top_left[0]-x1, inner_top_left[1]-y1), 5, (0,0,255), -1)
            cv2.circle(final_roi_img, (inner_bottom_right[0]-x1, inner_bottom_right[1]-y1), 5, (0,0,255), -1)
        
        cv2.imshow("Step 4 - Final Contours (ROI)", resize_to_fit(final_roi_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        'contour': {'outer': best_outer, 'inner': best_inner},
        'outer_points': {
            'top_left': outer_top_left,
            'bottom_right': outer_bottom_right
        },
        'inner_points': {
            'top_left': inner_top_left,
            'bottom_right': inner_bottom_right
        },
        'binary_image': binary_roi
    }

def find_shape_1(gray_img, top_left, bottom_right):
    """
    输入：
        gray_img：整张灰度图（单通道，0-255）
        top_left：内轮廓左上角 (x,y)
        bottom_right：内轮廓右下角 (x,y)
    返回：
        shape: 形状类型（'triangle', 'rectangle', 'pentagon', 'hexagon', 'circle'）
        size: 边长/直径
    """
    # 1. 裁剪ROI
    gray_img = gray_img
    # cv2.imshow("Step 1: ROI", gray_img)
    # cv2.waitKey(0)

    # 2. 二值化（白色为前景）
    binary = dynamic_threshold(gray_img, method='otsu')
    # cv2.imshow("Step 2: Binary", binary)
    # cv2.waitKey(0)

    # 3. 找外轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area and is_fully_filled(binary, cnt):
            max_area = area
            valid_contour = cnt

    if valid_contour is None:
        print("没有找到满足条件的白色闭合区域")
        return None, None

    # 4. 绘制轮廓
    contour_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, [valid_contour], -1, (0, 255, 0), 2)
    # cv2.imshow("Step 3: Valid Contour", contour_img)
    # cv2.waitKey(0)

    # 5. 多边形逼近 + 顶点角度
    epsilon = 0.02 * cv2.arcLength(valid_contour, True)
    approx = cv2.approxPolyDP(valid_contour, epsilon, True)
    approx = approx.reshape(-1, 2)

    angle_list = []
    for i in range(len(approx)):
        p1 = approx[i - 1]
        p2 = approx[i]
        p3 = approx[(i + 1) % len(approx)]
        ang = angle_between(p1, p2, p3)
        angle_list.append((tuple(p2), ang))

    # 6. 标注角点与角度
    annotated = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
    for (pt, ang) in angle_list:
        cv2.circle(annotated, pt, 5, (0, 0, 255), -1)
        cv2.putText(annotated, f"{ang:.1f}", (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    # cv2.imshow("Step 4: Angles", annotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 7. 形状判断
    num_vertices = len(approx)

    # 计算轮廓的圆形度
    perimeter = cv2.arcLength(valid_contour, True)
    circularity = 4 * np.pi * max_area / (perimeter * perimeter)

    # 计算尺寸（边长/直径）
    if num_vertices >= 3:  # 多边形
        # 计算平均边长
        side_lengths = []
        for i in range(num_vertices):
            p1 = approx[i]
            p2 = approx[(i + 1) % num_vertices]
            side_lengths.append(np.linalg.norm(p1 - p2))
        size = np.mean(side_lengths)
    else:  # 圆形
        size = np.sqrt(4 * max_area / np.pi)  # 直径

    # 形状分类
    if circularity > 0.9:  # 接近圆形
        shape = 'circle'
        # print(circularity)
    else:
        if num_vertices == 3:
            shape = 'triangle'
        elif num_vertices == 4:
            # 检查是否是矩形（角度接近90度）
            angles = [ang for (pt, ang) in angle_list]
            if all(85 <= ang <= 95 for ang in angles):
                shape = 'quadrate'
            else:
                shape = 'quadrate'
        else:
            # print(num_vertices)
            shape = f'circle'

    return shape, size

    """
    输入：
        gray_img：整张灰度图（单通道，0-255）
        top_left：内轮廓左上角 (x,y)
        bottom_right：内轮廓右下角 (x,y)
    返回：
        shape: 形状类型（'triangle', 'rectangle', 'pentagon', 'hexagon', 'circle'）
        size: 边长/直径
    """
    # 1. 裁剪ROI
    gray_img = gray_img
    # cv2.imshow("Step 1: ROI", gray_img)
    # cv2.waitKey(0)

    # 2. 二值化（白色为前景）
    binary = dynamic_threshold(gray_img, method='otsu')
    # cv2.imshow("Step 2: Binary", binary)
    # cv2.waitKey(0)

    # 3. 找外轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area and is_fully_filled(binary, cnt):
            max_area = area
            valid_contour = cnt

    if valid_contour is None:
        print("没有找到满足条件的白色闭合区域")
        return None, None

    # 4. 绘制轮廓
    contour_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(contour_img, [valid_contour], -1, (0, 255, 0), 2)
    # cv2.imshow("Step 3: Valid Contour", contour_img)
    # cv2.waitKey(0)

    # 5. 多边形逼近 + 顶点角度
    epsilon = 0.02 * cv2.arcLength(valid_contour, True)
    approx = cv2.approxPolyDP(valid_contour, epsilon, True)
    approx = approx.reshape(-1, 2)

    angle_list = []
    for i in range(len(approx)):
        p1 = approx[i - 1]
        p2 = approx[i]
        p3 = approx[(i + 1) % len(approx)]
        ang = angle_between(p1, p2, p3)
        angle_list.append((tuple(p2), ang))

    # 6. 标注角点与角度
    annotated = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
    for (pt, ang) in angle_list:
        cv2.circle(annotated, pt, 5, (0, 0, 255), -1)
        cv2.putText(annotated, f"{ang:.1f}", (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    # cv2.imshow("Step 4: Angles", annotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 7. 形状判断
    num_vertices = len(approx)

    # 计算轮廓的圆形度
    perimeter = cv2.arcLength(valid_contour, True)
    circularity = 4 * np.pi * max_area / (perimeter * perimeter)

    # 计算尺寸（边长/直径）
    if num_vertices >= 3:  # 多边形
        # 计算平均边长
        side_lengths = []
        for i in range(num_vertices):
            p1 = approx[i]
            p2 = approx[(i + 1) % num_vertices]
            side_lengths.append(np.linalg.norm(p1 - p2))
        size = np.mean(side_lengths)
    else:  # 圆形
        size = np.sqrt(4 * max_area / np.pi)  # 直径

    # 形状分类
    if circularity > 0.6:  # 接近圆形
        shape = 'circle'
    else:
        if num_vertices == 3:
            shape = 'triangle'
        elif num_vertices == 4:
            # 检查是否是矩形（角度接近90度）
            angles = [ang for (pt, ang) in angle_list]
            if all(85 <= ang <= 95 for ang in angles):
                shape = 'quadrate'
            else:
                shape = 'quadrate'
        else:
            shape = f'circle'

    return shape, size
from maix import camera, display
from maix import image

# 初始化相机和显示器

disp = display.Display()
cam = camera.Camera(1920, 1080)

import cv2

while True:
    # image = cv2.imread(r"E:\D-Race\C_T\test\1\1.jpg")
    # 读取相机图像
    img = cam.read()
    disp.show(img)
    img = image.image2cv(img, ensure_bgr=False, copy=False)
    # # image = cv2.imread(r"E:\D-Race\C_T\1_s.png")
    result = detect_outer_black_border(img, lower_threshold=50, upper_threshold=200)
    # gray = cv2.inRange(img, (0, 0, 0), (32, 77, 77))
    gray = result['binary_image']
    # show by maix.display
    # img_show = image.cv2image(gray, bgr=True, copy=False)
    # disp.show(img_show)
    inner_top_left = result['inner_points']['top_left']
    inner_bottom_right = result['inner_points']['bottom_right']
    outer_top_left = result['outer_points']['top_left']
    outer_bottom_right = result['outer_points']['bottom_right']

    x = (outer_top_left[1] + outer_bottom_right[1])/2
    y = outer_bottom_right[1]
    distance = round(img_change(x,y)*100) # cm *100
    if distance <=10000:
        distance = 10000
    height = abs(outer_top_left[1]-outer_bottom_right[1])
    width = abs(outer_top_left[0]-outer_bottom_right[0])
    area = width * height

    shape, size = find_shape_1(gray, inner_top_left, inner_bottom_right)

    print(f"外轮廓宽度: {width}, 外轮廓高度: {height}")
    print(f"检测到的形状: {shape}, 边长/直径: {size}")

    import math
    if shape == "quadrate":
        print("X = ",round((math.sqrt(size*size*623.7/(area)+3.5)*100))) # cm *100
    elif shape == "triangle":
        print("X = ", round((math.sqrt(size * size * 623.7 / (area))+2.63)*100)) # cm *100
    elif shape == "circle":
        print("X = ", round((math.sqrt(size * size * 623.7 / (area))/4*11)*100)) # cm *100

    print("D = ", distance/100-1.15,"cm")
