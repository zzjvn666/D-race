import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cv2_putText_Chinese(img, text, position, font_size, color):
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 使用支持中文的字体
    try:
        font = ImageFont.truetype("simsun.ttc", font_size)  # Windows系统自带宋体
    except:
        font = ImageFont.truetype("arialuni.ttf", font_size)  # 尝试其他字体

    # 绘制中文文本
    draw.text(position, text, font=font, fill=color)

    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


import cv2
import numpy as np
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
    cv2.imshow('处理过程', display_img)
    cv2.waitKey(0)

    # 2. 使用大津法自动确定阈值并提取白色区域
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 显示二值化结果
    display_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    display_img = cv2_putText_Chinese(display_img, "2. 白色区域提取", (10, 30), 20, (0, 0, 255))
    cv2.imshow('处理过程', display_img)
    cv2.waitKey(0)

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
        cv2.imshow('处理过程', display_img)
        cv2.waitKey(0)

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
        cv2.imshow('处理过程', display_img)
        cv2.waitKey(0)

        print(f"白色区域{i + 1}的顶点角度:", angles)

    cv2.destroyAllWindows()

import cv2
import numpy as np
import math

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
import math

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
    roi = gray_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv2.imshow("Step 1: ROI", roi)
    cv2.waitKey(0)

    # 2. 二值化（白色为前景）
    binary = dynamic_threshold(roi, method='otsu')
    cv2.imshow("Step 2: Binary", binary)
    cv2.waitKey(0)

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
    cv2.imshow("Step 3: Valid Contour", contour_img)
    cv2.waitKey(0)

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
    cv2.imshow("Step 4: Angles", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    if circularity > 0.85:  # 接近圆形
        shape = 'circle'
    else:
        if num_vertices == 3:
            shape = 'triangle'
        elif num_vertices == 4:
            # 检查是否是矩形（角度接近90度）
            angles = [ang for (pt, ang) in angle_list]
            if all(85 <= ang <= 95 for ang in angles):
                shape = 'rectangle'
            else:
                shape = 'quadrilateral'
        elif num_vertices == 5:
            shape = 'pentagon'
        elif num_vertices == 6:
            shape = 'hexagon'
        else:
            shape = f'polygon({num_vertices})'

    return shape, size


def detect_outer_black_border(image, draw_result=True, lower_threshold=0, upper_threshold=255):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # 2. 二值化（反转颜色，使黑色变为白色以便识别）
    _, binary = cv2.threshold(gray, lower_threshold, upper_threshold, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 查找轮廓（使用层级信息）
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]  # 层级结构

    best_cnt = None
    max_area = 0

    for idx, (cnt, hier) in enumerate(zip(contours, hierarchy)):
        # 条件：有子轮廓（环形结构）且面积最大
        if hier[2] != -1:  # 有子轮廓
            area = cv2.contourArea(cnt)
            if area > max_area:
                best_cnt = cnt
                max_area = area

    if best_cnt is not None:
        # 4. 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(best_cnt)
        outer_top_left = (x, y)
        outer_bottom_right = (x + w, y + h)

        # 5. 查找子轮廓（内侧边界）
        child_idx = hierarchy[contours.index(best_cnt)][2]
        if child_idx != -1:
            inner_cnt = contours[child_idx]
            x_inner, y_inner, w_inner, h_inner = cv2.boundingRect(inner_cnt)
            inner_top_left = (x_inner, y_inner)
            inner_bottom_right = (x_inner + w_inner, y_inner + h_inner)
        else:
            inner_top_left = inner_bottom_right = None

        # 6. 绘制结果
        if draw_result:
            result = image.copy()
            cv2.drawContours(result, [best_cnt], -1, (0, 0, 255), 3)

            # 绘制外侧边界框
            cv2.rectangle(result, outer_top_left, outer_bottom_right, (0, 255, 0), 2)
            # 绘制内侧边界框
            if inner_top_left and inner_bottom_right:
                cv2.rectangle(result, inner_top_left, inner_bottom_right, (255, 0, 0), 2)

            # 标记关键点
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, 'Outer TL', outer_top_left, font, 0.5, (0, 255, 0), 2)
            cv2.putText(result, 'Outer BR', outer_bottom_right, font, 0.5, (0, 255, 0), 2)
            if inner_top_left:
                cv2.putText(result, 'Inner TL', inner_top_left, font, 0.5, (255, 0, 0), 2)
            if inner_bottom_right:
                cv2.putText(result, 'Inner BR', inner_bottom_right, font, 0.5, (255, 0, 0), 2)


        # 返回关键点坐标和二值化图像
        return {
            'contour': best_cnt,
            'outer_points': {
                'top_left': outer_top_left,
                'bottom_right': outer_bottom_right
            },
            'inner_points': {
                'top_left': inner_top_left,
                'bottom_right': inner_bottom_right
            },
            'binary_image': binary
        }
    else:
        print("未检测到目标黑色外框。")
        return {
            'contour': None,
            'outer_points': None,
            'inner_points': None,
            'binary_image': binary  # 仍然返回二值化图像
        }


if __name__ == "__main__":
    image = cv2.imread(r"E:\D-Race\C_T\2_z_2.png")
    result = detect_outer_black_border(image, lower_threshold=50, upper_threshold=200)

    gray = result['binary_image']
    inner_top_left = result['inner_points']['top_left']
    inner_bottom_right = result['inner_points']['bottom_right']
    shape, size = find_shape_1(gray, inner_top_left, inner_bottom_right)
    print(f"检测到的形状: {shape}, 边长/直径: {size}")