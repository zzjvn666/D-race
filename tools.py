import cv2
import numpy as np

# 导入原有功能函数
from PIL import Image, ImageDraw, ImageFont
# === 透视变换矩阵 M ===
M = np.array([
    [-1.55613833e+00, -3.10409025e+00, 1.95931565e+03],
    [-4.89434105e-02, -8.06154531e+00, 2.56000370e+03],
    [-1.17592704e-04, -1.05069141e-02, 1.00000000e+00]
], dtype=np.float32)


# === 非线性 Y 压缩函数 ===
def nonlinear_y(y):
    segments = [
        (53.00, 81.00, 0.000000e+00, 0.000000e+00, 1.073529e+00, 5.300000e+01),
        (81.00, 109.00, -2.388473e-05, 6.687726e-04, 1.073529e+00, 8.305882e+01),
        (109.00, 138.00, -5.616156e-05, 9.979052e-04, 1.054804e+00, 1.131176e+02),
        (138.00, 171.00, 9.569326e-05, -4.979490e-03, 9.709867e-01, 1.431765e+02),
        (171.00, 201.00, -5.221177e-05, 3.132707e-03, 9.549702e-01, 1.732353e+02),
        (201.00, 231.00, -1.480297e-18, 6.661338e-17, 1.001961e+00, 2.032941e+02),
        (231.00, 261.00, -1.815255e-05, 5.445764e-04, 1.001961e+00, 2.333529e+02),
        (261.00, 292.00, 1.663270e-05, -1.031227e-03, 9.856235e-01, 2.634118e+02),
        (292.00, 323.00, 2.772669e-18, -1.468360e-16, 9.696395e-01, 2.934706e+02),
        (323.00, 354.00, 1.663270e-05, -5.156136e-04, 9.696395e-01, 3.235294e+02),
        (354.00, 384.00, -3.630510e-05, 1.633729e-03, 9.856235e-01, 3.535882e+02),
        (384.00, 415.00, 3.326539e-05, -1.546841e-03, 9.856235e-01, 3.836471e+02),
        (415.00, 445.00, -1.815255e-05, 1.089153e-03, 9.856235e-01, 4.137059e+02),
        (445.00, 475.00, 0.000000e+00, 0.000000e+00, 1.001961e+00, 4.437647e+02),
        (475.00, 505.00, 9.868649e-19, -2.960595e-17, 1.001961e+00, 4.738235e+02),
        (505.00, 535.00, 1.897774e-05, -5.693323e-04, 1.001961e+00, 5.038824e+02),
        (535.00, 564.00, -5.802834e-07, 6.192559e-04, 1.019041e+00, 5.339412e+02),
    ]
    for start, end, a, b, c, d in segments:
        if start <= y < end:
            t = y - start
            return a * t ** 3 + b * t ** 2 + c * t + d
    return y

# M = np.array([
#     [-1.09199972e+00, -2.34781415e+00, 1.34204325e+03],
#     [-2.33254794e-01, -5.74609503e+00, 1.81585111e+03],
#     [-4.29184713e-04, -7.94632703e-03, 1.00000000e+00]
# ], dtype=np.float32)


# # === 非线性 Y 压缩函数 ===
# def nonlinear_y(y):
#     segments = [
#         (38.00, 62.00, -9.666785e-06, -1.994196e-03, 1.075254e+00, 3.800000e+01),
#         (62.00, 89.00, 2.378948e-04, -8.443135e-03, 9.628286e-01, 6.252381e+01),
#         (89.00, 110.00, -5.430358e-04, 1.810019e-02, 1.027175e+00, 8.704762e+01),
#         (110.00, 135.00, 1.407911e-04, -7.039557e-03, 1.068947e+00, 1.115714e+02),
#         (135.00, 160.00, 6.634351e-05, -1.658588e-03, 9.809524e-01, 1.360952e+02),
#         (160.00, 183.00, -8.286487e-05, 3.811784e-03, 1.022417e+00, 1.606190e+02),
#         (183.00, 206.00, -1.210369e-04, 2.783848e-03, 1.066253e+00, 1.851429e+02),
#         (206.00, 232.00, 1.148202e-04, -5.254581e-03, 1.002224e+00, 2.096667e+02),
#         (232.00, 257.00, 1.675723e-06, 7.225507e-04, 9.618413e-01, 2.341905e+02),
#         (257.00, 281.00, -3.596284e-05, 1.726216e-03, 1.001111e+00, 2.587143e+02),
#         (281.00, 305.00, -3.596284e-05, 8.631081e-04, 1.021825e+00, 2.832381e+02),
#         (305.00, 330.00, 3.225348e-05, -1.612674e-03, 1.001111e+00, 3.077619e+02),
#         (330.00, 355.00, 1.776357e-18, -8.881784e-17, 9.809524e-01, 3.322857e+02),
#         (355.00, 380.00, -3.552714e-19, 1.332268e-17, 9.809524e-01, 3.568095e+02),
#         (380.00, 405.00, 3.225348e-05, -8.063369e-04, 9.809524e-01, 3.813333e+02),
#         (405.00, 429.00, -1.060142e-04, 3.407448e-03, 1.001111e+00, 4.058571e+02),
#         (429.00, 455.00, 8.412757e-05, -3.658562e-03, 9.814758e-01, 4.303810e+02),
#         (455.00, 480.00, 1.675723e-06, 7.225507e-04, 9.618413e-01, 4.549048e+02),
#         (480.00, 504.00, -7.192567e-05, 2.589324e-03, 1.001111e+00, 4.794286e+02),
#         (504.00, 529.00, 6.450695e-05, -2.419011e-03, 1.001111e+00, 5.039524e+02),
#         (529.00, 553.00, -1.206871e-06, 8.920730e-04, 1.001111e+00, 5.284762e+02),
#     ]
#     for start, end, a, b, c, d in segments:
#         if start <= y < end:
#             t = y - start
#             return a * t ** 3 + b * t ** 2 + c * t + d
#     return y

# === 图像坐标变换函数 ===
def img_change(x, y):
    src = np.array([[[x, y]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, M)[0][0]
    x_new, y_new = dst[0], nonlinear_y(dst[1])
    distance = 100 * ((-y_new + 413) / 303 + 1)
    return distance
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
# 1. 裁剪ROI并放置到黑色背景中
def crop_and_place_on_black(gray_img, top_left, bottom_right, padding=10):
    """
    裁剪ROI并放置到与原始图像大小相同的黑色背景中

    参数:
        gray_img: 原始灰度图像
        top_left: 裁剪区域左上角坐标 (x, y)
        bottom_right: 裁剪区域右下角坐标 (x, y)
        padding: 裁剪区域内缩的像素数

    返回:
        放置在黑色背景中的ROI图像
    """
    # 创建与原始图像大小相同的黑色画布
    # 1. 计算扩充后的ROI坐标（向外扩充10像素）
    expanded_top_left = (max(top_left[0] - 10, 0), max(top_left[1] - 10, 0))
    expanded_bottom_right = (min(bottom_right[0] + 10, gray_img.shape[1]),
                             min(bottom_right[1] + 10, gray_img.shape[0]))

    # 2. 创建与扩充后ROI大小相同的黑色画布
    expanded_width = expanded_bottom_right[0] - expanded_top_left[0]
    expanded_height = expanded_bottom_right[1] - expanded_top_left[1]
    black_bg = np.zeros((expanded_height, expanded_width), dtype=gray_img.dtype)

    # 计算裁剪区域（考虑padding）
    y_start = max(top_left[1] + padding, 0)
    y_end = min(bottom_right[1] - padding, gray_img.shape[0])
    x_start = max(top_left[0] + padding, 0)
    x_end = min(bottom_right[0] - padding, gray_img.shape[1])

    # 确保裁剪区域有效
    if y_start >= y_end or x_start >= x_end:
        print("裁剪区域无效，返回原始图像")
        return gray_img

    # 裁剪ROI
    roi = gray_img[y_start:y_end, x_start:x_end]

    # 将ROI放置到黑色画布的正确位置
    black_bg[y_start:y_end, x_start:x_end] = roi

    return black_bg
def angle_between(p1, p2, p3):
    # 计算向量 p2->p1 和 p2->p3 的夹角（外侧转向角）
    v1 = p1 - p2
    v2 = p3 - p2
    ang = np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))
    return ang if ang >= 0 else ang + 360  # 统一到 0~360 范围
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
    return fill_ratio > 0.8
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

def find_squares_from_contour(approx=None, angle_thresh=20, side_ratio_thresh=0.2):
    """
    从逼近的轮廓顶点中判断是否是正方形，并返回点集。
    输入：
        approx: 顶点坐标列表，形如 [[x1,y1],[x2,y2],...]
        angle_thresh: 判断是否为直角的阈值（±angle_thresh 视为直角）
        side_ratio_thresh: 判断边长相似度的比例阈值
    返回：
        square_groups: 如果是正方形，返回顶点列表 [[pt1, pt2, pt3, pt4]]；否则返回 []
    """
    import numpy as np

    if approx is None or len(approx) != 4:
        return []  # 只处理四边形

    # 计算四个角
    def angle(a, b, c):
        v1 = a - b
        v2 = c - b
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle_rad)

    angles = []
    for i in range(4):
        p1 = approx[i - 1]
        p2 = approx[i]
        p3 = approx[(i + 1) % 4]
        ang = angle(p1, p2, p3)
        angles.append(ang)

    # 判断是否接近直角
    if not all(90 - angle_thresh <= a <= 90 + angle_thresh for a in angles):
        return []

    # 判断边长是否相近
    sides = [np.linalg.norm(approx[i] - approx[(i + 1) % 4]) for i in range(4)]
    max_side = max(sides)
    min_side = min(sides)
    if (max_side - min_side) / max_side > side_ratio_thresh:
        return []

    return [approx.tolist()]  # 返回一个正方形（四点列表的列表）

def _find_squares_from_contour(contour_binary, min_area=100, epsilon_ratio=0.02, debug=True):
    """从二值图像中基于轮廓识别正方形"""
    square_contours = []

    if debug:
        print("=== 开始正方形识别 ===")
        print(f"输入图像尺寸: {contour_binary.shape}")

    # 查找轮廓
    contours, _ = cv2.findContours(contour_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        print(f"找到的轮廓数量: {len(contours)}")

    for i, contour in enumerate(contours):
        if debug:
            print(f"\n处理轮廓 {i + 1}/{len(contours)}")

        # 计算轮廓面积
        area = cv2.contourArea(contour)

        # 过滤过小的轮廓
        if area < min_area:
            if debug:
                print(f"  面积过小: {area:.2f} < {min_area}，跳过")
            continue

        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)

        # 多边形逼近，获取近似轮廓
        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 正方形应近似为4个顶点
        if len(approx) != 4:
            if debug:
                print(f"  顶点数量不为4: {len(approx)}，跳过")
            continue

        # 计算轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        # 检查宽高比是否接近1（正方形特征）
        if not (0.85 <= aspect_ratio <= 1.15):
            if debug:
                print(f"  宽高比不合适: {aspect_ratio:.2f}，跳过")
            continue

        # 计算轮廓面积与边界矩形面积的比率
        rect_area = w * h
        area_ratio = area / rect_area

        # 进一步筛选：面积比应接近1
        if not (0.8 <= area_ratio <= 1.0):
            if debug:
                print(f"  面积比不合适: {area_ratio:.2f}，跳过")
            continue

        # 计算四个角点间的距离，检查是否四条边近似相等
        points = approx.reshape(-1, 2)
        distances = []

        for j in range(4):
            p1 = points[j]
            p2 = points[(j + 1) % 4]
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            distances.append(dist)

        # 计算边长变异系数（标准差/均值）
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        cv_dist = std_dist / mean_dist if mean_dist > 0 else 0

        # 检查边长是否足够均匀
        if cv_dist > 0.15:  # 允许15%的变异
            if debug:
                print(f"  边长变异系数过大: {cv_dist:.2f}，跳过")
            continue

        # 如果通过所有筛选，则认为是正方形
        square_contours.append(approx)

        if debug:
            print("  找到正方形！")
            print(f"    面积: {area:.2f}")
            print(f"    宽高比: {aspect_ratio:.2f}")
            print(f"    面积比: {area_ratio:.2f}")
            print(f"    边长变异系数: {cv_dist:.2f}")
            print(f"    顶点坐标: {points.tolist()}")

    if debug:
        print(f"\n=== 正方形识别完成 ===")
        print(f"总共识别的正方形数量: {len(square_contours)}")

    return square_contours
def find_squares_from_corners(angle_list, binary, debug=True):
    """从角点列表中识别可能的正方形"""
    square_groups = []
    used_points = set()
    # 新增：已使用点的坐标集合（用于距离判断）
    used_coordinates = []
    # 新增：距离阈值，可根据实际情况调整
    DISTANCE_THRESHOLD = 5

    if debug:
        print("=== 开始正方形识别 ===")
        print(f"输入的角点数量: {len(angle_list)}")
        print(f"图像尺寸: {binary.shape}")

    # 筛选近似90度的角点
    right_angles = [pt for pt, ang in angle_list if 70 <= ang <= 110]
    if debug:
        print(f"\n筛选后的直角点数量: {len(right_angles)}")
        print("直角点坐标:", right_angles)

    for i, pt1 in enumerate(right_angles):
        if debug:
            print(f"\n处理点 {i}: {pt1}")

        # 新增：检查是否与已使用点过近
        if is_too_close_to_used(pt1, used_coordinates, DISTANCE_THRESHOLD):
            if debug:
                print(f"该点与已使用点距离小于{DISTANCE_THRESHOLD}，跳过")
            continue

        if pt1 in used_points:
            if debug:
                print("该点已被使用，跳过")
            continue

        # 寻找相邻的直角点
        neighbors = []
        for j, pt2 in enumerate(right_angles):
            if i == j or pt2 in used_points:
                continue

            # 计算两点距离
            dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            if debug:
                print(f"  与点 {j} ({pt2}) 的距离: {dist:.2f}")

            # 如果距离在合理范围内（非零且不过大）
            if 10 < dist < min(binary.shape):
                neighbors.append(pt2)
                if debug:
                    print("  距离在范围内，添加到邻居")

        if debug:
            print(f"找到的邻居点数量: {len(neighbors)}")
            print("邻居点坐标:", neighbors)

        # 处理找到的相邻直角点
        for neighbor in neighbors:
            if debug:
                print(f"\n处理邻居点: {neighbor}")

            # 计算向量和可能的正方形顶点
            vec = np.array(neighbor) - np.array(pt1)
            vec_perp = np.array([-vec[1], vec[0]])  # 垂直向量
            if debug:
                print(f"向量: {vec}")
                print(f"垂直向量: {vec_perp}")

            # 计算可能的正方形四个顶点
            square_points = [
                pt1,
                neighbor,
                (neighbor[0] + vec_perp[0], neighbor[1] + vec_perp[1]),
                (pt1[0] + vec_perp[0], pt1[1] + vec_perp[1])
            ]
            if debug:
                print("计算的正方形顶点:")
                for idx, pt in enumerate(square_points):
                    print(f"  顶点{idx}: {pt}")

            # 检查所有点是否在图像范围内且为白色区域
            valid = True
            for x, y in square_points:
                if x < 0 or y < 0 or x >= binary.shape[1] or y >= binary.shape[0]:
                    if debug:
                        print(f"点({x},{y})超出图像范围")
                    valid = False
                    break

            if valid:
                if debug:
                    print("所有顶点都在图像范围内")

                # 新增：检查是否有任何点与已使用点过近
                if any(is_too_close_to_used(pt, used_coordinates, DISTANCE_THRESHOLD) for pt in square_points):
                    if debug:
                        print(f"某些顶点与已使用点距离小于{DISTANCE_THRESHOLD}，跳过")
                else:
                    # 检查是否有任何点已被使用
                    if any(pt in used_points for pt in square_points):
                        if debug:
                            print("某些顶点已被使用，跳过")
                    else:
                        square_groups.append(square_points)
                        used_points.update(square_points)
                        # 新增：添加到已使用坐标集合
                        used_coordinates.extend(square_points)
                        if debug:
                            print("添加正方形到结果中")
                            print(f"当前已识别正方形数量: {len(square_groups)}")
                            print(f"已使用的点数量: {len(used_points)}")

    if debug:
        print("\n=== 正方形识别完成 ===")
        print(f"总共识别的正方形数量: {len(square_groups)}")

    return square_groups


# 新增：判断点是否与已使用点过近的辅助函数
def is_too_close_to_used(point, used_coordinates, threshold):
    """检查点是否与任何已使用点的距离小于阈值"""
    for used_point in used_coordinates:
        dist = np.sqrt((point[0] - used_point[0]) ** 2 + (point[1] - used_point[1]) ** 2)
        if dist < threshold:
            return True
    return False
def _find_squares_from_corners(angle_list, binary, debug = False):
    """从角点列表中识别可能的正方形"""
    square_groups = []
    used_points = set()

    if debug:
        print("=== 开始正方形识别 ===")
        print(f"输入的角点数量: {len(angle_list)}")
        print(f"图像尺寸: {binary.shape}")

    # 筛选近似90度的角点
    right_angles = [pt for pt, ang in angle_list if 70 <= ang <= 110]
    if debug:
        print(f"\n筛选后的直角点数量: {len(right_angles)}")
        print("直角点坐标:", right_angles)

    for i, pt1 in enumerate(right_angles):
        if debug:
            print(f"\n处理点 {i}: {pt1}")

        if pt1 in used_points:
            if debug:
                print("该点已被使用，跳过")
            continue

        # 寻找相邻的直角点
        neighbors = []
        for j, pt2 in enumerate(right_angles):
            if i == j or pt2 in used_points:
                continue

            # 计算两点距离
            dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            if debug:
                print(f"  与点 {j} ({pt2}) 的距离: {dist:.2f}")

            # 如果距离在合理范围内（非零且不过大）
            if 10 < dist < min(binary.shape):
                neighbors.append(pt2)
                if debug:
                    print("  距离在范围内，添加到邻居")

        if debug:
            print(f"找到的邻居点数量: {len(neighbors)}")
            print("邻居点坐标:", neighbors)

        # 处理找到的相邻直角点
        for neighbor in neighbors:
            if debug:
                print(f"\n处理邻居点: {neighbor}")

            # 计算向量和可能的正方形顶点
            vec = np.array(neighbor) - np.array(pt1)
            vec_perp = np.array([-vec[1], vec[0]])  # 垂直向量
            if debug:
                print(f"向量: {vec}")
                print(f"垂直向量: {vec_perp}")

            # 计算可能的正方形四个顶点
            square_points = [
                pt1,
                neighbor,
                (neighbor[0] + vec_perp[0], neighbor[1] + vec_perp[1]),
                (pt1[0] + vec_perp[0], pt1[1] + vec_perp[1])
            ]
            if debug:
                print("计算的正方形顶点:")
                for idx, pt in enumerate(square_points):
                    print(f"  顶点{idx}: {pt}")

            # 检查所有点是否在图像范围内且为白色区域
            valid = True
            for x, y in square_points:
                if x < 0 or y < 0 or x >= binary.shape[1] or y >= binary.shape[0]:
                    if debug:
                        print(f"点({x},{y})超出图像范围")
                    valid = False
                    break

            if valid:
                if debug:
                    print("所有顶点都在图像范围内且为白色区域")

                # 检查是否有任何点已被使用
                if any(pt in used_points for pt in square_points):
                    if debug:
                        print("某些顶点已被使用，跳过")
                else:
                    square_groups.append(square_points)
                    used_points.update(square_points)
                    if debug:
                        print("添加正方形到结果中")
                        print(f"当前已识别正方形数量: {len(square_groups)}")
                        print(f"已使用的点数量: {len(used_points)}")

    if debug:
        print("\n=== 正方形识别完成 ===")
        print(f"总共识别的正方形数量: {len(square_groups)}")

    return square_groups
def _detect_outer_black_border(image, draw_result=True, lower_threshold=0, upper_threshold=255):
    # 1. 读取图像并灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # 2. 二值化（反转颜色，使黑色变为白色以便识别）
    _, binary = cv2.threshold(gray, lower_threshold, upper_threshold, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 查找轮廓（使用层级信息）
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]  # 层级结构

    best_cnt = None
    best_idx = -1
    max_area = 0

    for idx, (cnt, hier) in enumerate(zip(contours, hierarchy)):
        if hier[2] != -1:  # 有子轮廓
            area = cv2.contourArea(cnt)
            if area > max_area:
                best_cnt = cnt
                best_idx = idx
                max_area = area

    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        outer_top_left = (x, y)
        outer_bottom_right = (x + w, y + h)

        child_idx = hierarchy[best_idx][2]
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
def detect_outer_black_border(image, draw_result=True, lower_threshold=0, upper_threshold=255, debug=True,
                              COLOR_B=32, COLOR_G=27, COLOR_R=27):
    def resize_to_fit(img, target_size=(800, 600)):
        """将图像按比例缩放以适应目标大小"""
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        return resized

    gray = cv2.inRange(image, (0, 0, 0), (COLOR_B, COLOR_G, COLOR_R))

    # # 1. 提取黑色区域（基于BGR阈值）
    # gray = cv2.GaussianBlur(cv2.inRange(image, (0, 0, 0), (COLOR_B, COLOR_G, COLOR_R)), (3, 3), 1)

    # 2. 形态学处理：先开运算再闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)  # 开运算：去小噪点
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)  # 闭运算：填小空洞
    gray = cv2.bitwise_not(gray)  # 反转：黑色区域变为白色，便于后续处理
    # gray = cv2.inRange(image, (0, 0, 0), (COLOR_B, COLOR_G, COLOR_R))
    # gray = cv2.bitwise_not(gray)  # 反转：黑色区域变为白色，便于后续处理
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
    # kernel = np.ones((2, 2), np.uint8)
    # binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel, iterations=1)

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

    def sort_corners(corners):
        """按 左上、右上、右下、左下 顺序返回角点"""
        corners = np.array(corners)
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)

        top_left = corners[np.argmin(s)]
        bottom_right = corners[np.argmax(s)]
        top_right = corners[np.argmin(diff)]
        bottom_left = corners[np.argmax(diff)]

        return [tuple(top_left), tuple(top_right), tuple(bottom_right), tuple(bottom_left)]

    # === 外轮廓角点 ===
    outer_corners = None
    if best_outer is not None:
        epsilon = 0.02 * cv2.arcLength(best_outer, True)
        approx = cv2.approxPolyDP(best_outer, epsilon, True)
        if len(approx) >= 4:
            approx_pts = approx.reshape(-1, 2)
            outer_corners = [(pt[0] + x1, pt[1] + y1) for pt in approx_pts]
            xs = [pt[0] for pt in outer_corners]
            ys = [pt[1] for pt in outer_corners]
            outer_top_left = (min(xs), min(ys))
            outer_bottom_right = (max(xs), max(ys))

    # === 内轮廓角点 ===
    inner_corners = None
    if best_inner is not None:
        epsilon = 0.02 * cv2.arcLength(best_inner, True)
        approx = cv2.approxPolyDP(best_inner, epsilon, True)
        if len(approx) >= 4:
            approx_pts = approx.reshape(-1, 2)
            inner_corners = [(pt[0] + x1, pt[1] + y1) for pt in approx_pts]
            xs = [pt[0] for pt in inner_corners]
            ys = [pt[1] for pt in inner_corners]
            inner_top_left = (min(xs), min(ys))
            inner_bottom_right = (max(xs), max(ys))

    if debug:
        final_roi_img = cv2.cvtColor(binary_roi.copy(), cv2.COLOR_GRAY2BGR)
        if outer_corners:
            for pt in outer_corners:
                cv2.circle(final_roi_img, (pt[0] - x1, pt[1] - y1), 5, (255, 0, 255), -1)
        if inner_corners:
            for pt in inner_corners:
                cv2.circle(final_roi_img, (pt[0] - x1, pt[1] - y1), 5, (0, 255, 255), -1)
        cv2.imshow("Step 4 - Final Contours (ROI)", resize_to_fit(final_roi_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    result = {
         'contour': {'outer': best_outer, 'inner': best_inner},

            'outer': best_outer,
            'inner': best_inner,
        'binary_image': binary_roi

    }

    # 检查内外角点数量
    if outer_corners and inner_corners and len(inner_corners) == 4:
        # 情况1：内外角点都正好有4个
        if len(outer_corners) == 4:
            result['outer_points'] = {
                'top_left': outer_top_left,
                'bottom_right': outer_bottom_right
            }
            result['inner_points'] = {
                'top_left': inner_top_left,
                'bottom_right': inner_bottom_right
            }
        # 情况2：内轮廓4个角点，外轮廓4个以上角点
        elif len(outer_corners) >= 4:
            # 提取内轮廓的四个角点
            inner_tl, inner_tr, inner_br, inner_bl = inner_corners  # 假设已按顺序排列

            # 定义一个函数计算两点之间的距离
            def distance(pt1, pt2):
                return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

            # 为每个内轮廓角点找到外轮廓中最接近的对应角点
            # 左上：找比内轮廓左上更靠左上的外轮廓点
            outer_tl = min([pt for pt in outer_corners if pt[0] <= inner_tl[0] and pt[1] <= inner_tl[1]],
                           key=lambda p: distance(p, inner_tl))

            # 右上：找比内轮廓右上更靠右上的外轮廓点
            outer_tr = min([pt for pt in outer_corners if pt[0] >= inner_tr[0] and pt[1] <= inner_tr[1]],
                           key=lambda p: distance(p, inner_tr))

            # 右下：找比内轮廓右下更靠右下的外轮廓点
            outer_br = min([pt for pt in outer_corners if pt[0] >= inner_br[0] and pt[1] >= inner_br[1]],
                           key=lambda p: distance(p, inner_br))

            # 左下：找比内轮廓左下更靠左下的外轮廓点
            outer_bl = min([pt for pt in outer_corners if pt[0] <= inner_bl[0] and pt[1] >= inner_bl[1]],
                           key=lambda p: distance(p, inner_bl))

            # 检查相对的角点距离是否差不多近（这里用阈值判断）
            diag1 = distance(outer_tl, outer_br)
            diag2 = distance(outer_tr, outer_bl)
            distance_ratio = min(diag1, diag2) / max(diag1, diag2)

            # 如果对角线角线长度比例在可接受范围内（例如0.8以上）
            if distance_ratio > 0.8:
                # 重塑外轮廓，使用新的四个角点
                new_outer_corners = [outer_tl, outer_tr, outer_br, outer_bl]
                result['contour']['outer'] = new_outer_corners  # 替换旧外轮廓

                # 返回新的外轮廓角点
                result['outer_points'] = {
                    'top_left': outer_tl,
                    'bottom_right': outer_br
                }
                result['inner_points'] = {
                    'top_left': inner_tl,
                    'bottom_right': inner_br
                }
                # 更新outer_corners为新的四个角点
                outer_corners = new_outer_corners
            else:
                if debug:
                    print("外轮廓对角点距离差异过大，不进行重塑")
        return result
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

def _detect_outer_black_border(image, draw_result=True, lower_threshold=0, upper_threshold=255, debug=True, COLOR_B=80, COLOR_G=70, COLOR_R=60):
    import cv2
    import numpy as np

    def process_contour_curve(cnt, angle_threshold=20, image=None, debug=False, min_len_threshold=20,
                              deviation_threshold=3):
        import cv2
        import numpy as np

        def show_contour(title, img, contour, color=(0, 255, 0), thickness=6):
            vis = img.copy()
            if contour.ndim == 2:
                contour = contour.reshape(-1, 1, 2)
            h, w = vis.shape[:2]
            if np.any(contour < 0) or np.any(contour[..., 0] >= w) or np.any(contour[..., 1] >= h):
                print(f"[警告] {title} 轮廓越界，不显示")
                return
            cv2.drawContours(vis, [contour], -1, color, thickness)
            cv2.imshow(title, resize_to_fit(vis))
            cv2.waitKey(0)

        def angle_between(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            return angle

        def is_axis_aligned(v, tol=5):
            # 判断向量v是否近似水平或垂直，tol为角度容忍度（度）
            angle = abs(np.degrees(np.arctan2(v[1], v[0])))
            angle = angle % 180  # 限制在0-180度范围
            return (angle < tol or abs(angle - 180) < tol or abs(angle - 90) < tol)

        if debug and image is not None:
            show_contour("Step 0 - Original Contour", image, cnt, (0, 255, 255))

        # 简化轮廓点
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx = approx.reshape(-1, 2)

        # Step 1: 找到两个有效直角，且这两个直角对应点间距大于阈值，用直线连接替换中间部分
        right_angles = []
        for i in range(1, len(approx) - 1):
            p1, p2, p3 = approx[i - 1], approx[i], approx[i + 1]
            v1 = p1 - p2
            v2 = p3 - p2
            angle = angle_between(p1, p2, p3)
            len_v1 = np.linalg.norm(v1)
            len_v2 = np.linalg.norm(v2)
            if (90 - angle_threshold <= angle <= 90 + angle_threshold and
                    len_v1 > min_len_threshold and len_v2 > min_len_threshold and
                    is_axis_aligned(v1) and is_axis_aligned(v2)):
                right_angles.append(i)

        if len(right_angles) >= 2:
            for i in range(len(right_angles) - 1):
                idx1, idx2 = right_angles[i], right_angles[i + 1]
                p1, p2 = approx[idx1], approx[idx2]
                dist = np.linalg.norm(p2 - p1)
                if dist > min_len_threshold:
                    # 构造新轮廓：从头到第一个直角点 + 连接第二个直角点 + 第二个直角点之后的点
                    new_points = np.vstack([approx[:idx1 + 1], p2, approx[idx2 + 1:]])
                    new_cnt = new_points.reshape(-1, 1, 2)
                    if debug and image is not None:
                        show_contour("Step 1 - Two Right Angles Reconnected", image, new_cnt, (255, 255, 0))
                    return new_cnt

        # Step 2: 判断是否为偏移的水平或垂直线，若偏移则直接用首尾点连接
        points = cnt.reshape(-1, 2)
        dx = np.abs(points[-1][0] - points[0][0])
        dy = np.abs(points[-1][1] - points[0][1])
        length = np.hypot(dx, dy)
        orientation = "horizontal" if dx > dy else "vertical"

        def is_deviated_line(p_list, axis='x', threshold=deviation_threshold):
            if axis == 'x':
                values = [p[1] for p in p_list]  # y方向应该一致
            else:
                values = [p[0] for p in p_list]  # x方向应该一致
            return np.max(values) - np.min(values) > threshold

        if length >= min_len_threshold:
            if orientation == 'horizontal' and is_deviated_line(points, axis='x'):
                print("✅ 水平线中存在偏移，重建直线")
                new_cnt = np.array([points[0], points[-1]], dtype=np.int32).reshape(-1, 1, 2)
                if debug and image is not None:
                    show_contour("Step 2 - Horizontal Curve Removed", image, new_cnt, (255, 0, 0))
                return new_cnt
            elif orientation == 'vertical' and is_deviated_line(points, axis='y'):
                print("✅ 垂直线中存在偏移，重建直线")
                new_cnt = np.array([points[0], points[-1]], dtype=np.int32).reshape(-1, 1, 2)
                if debug and image is not None:
                    show_contour("Step 2 - Vertical Curve Removed", image, new_cnt, (0, 255, 0))
                return new_cnt

        return cnt  # 如果都不满足，则返回原轮廓

    def _process_contour_curve(cnt, image=None, debug=False, extension_threshold=40, min_length_ratio=0.3):
        import cv2
        import numpy as np

        def show_contour(title, img, contour, color=(0, 255, 0), thickness=2):
            vis = img.copy()
            if contour.ndim == 2:
                contour = contour.reshape(-1, 1, 2)
            cv2.drawContours(vis, [contour], -1, color, thickness)
            cv2.imshow(title, vis)
            cv2.waitKey(0)

        points = cnt.reshape(-1, 2)
        num_points = len(points)
        total_len = cv2.arcLength(cnt, True)

        # 存储水平或垂直段
        candidate_segments = []
        for i in range(num_points - 1):
            p1, p2 = points[i], points[i + 1]
            v = p2 - p1
            length = np.linalg.norm(v)
            if length < 1:
                continue
            angle = abs(np.degrees(np.arctan2(v[1], v[0]))) % 180
            is_horiz = (angle < 10 or abs(angle - 180) < 10)
            is_vert = abs(angle - 90) < 10
            if (is_horiz or is_vert) and (length / total_len >= min_length_ratio):
                candidate_segments.append((i, p1, p2, is_horiz, is_vert))

        modified = False
        for idx1, p1_start, p1_end, is_horiz1, is_vert1 in candidate_segments:
            dir_vec = p1_end - p1_start
            norm_dir = dir_vec / np.linalg.norm(dir_vec)
            ext_point = p1_end + norm_dir * extension_threshold  # 延长一点

            for idx2, p2_start, p2_end, is_horiz2, is_vert2 in candidate_segments:
                if idx1 == idx2:
                    continue
                if is_horiz1 and is_horiz2 and abs(p2_start[1] - p1_end[1]) < 5:
                    dist = np.linalg.norm(ext_point - p2_start)
                    if dist < extension_threshold:
                        # 替换中间段
                        new_cnt = np.vstack([points[:idx1 + 1], p2_start, points[idx2 + 1:]]).reshape(-1, 1, 2)
                        if debug and image is not None:
                            show_contour("Connected Horizontal", image, new_cnt, (255, 0, 0))
                        return new_cnt
                if is_vert1 and is_vert2 and abs(p2_start[0] - p1_end[0]) < 5:
                    dist = np.linalg.norm(ext_point - p2_start)
                    if dist < extension_threshold:
                        new_cnt = np.vstack([points[:idx1 + 1], p2_start, points[idx2 + 1:]]).reshape(-1, 1, 2)
                        if debug and image is not None:
                            show_contour("Connected Vertical", image, new_cnt, (0, 0, 255))
                        return new_cnt

        return cnt  # 没修改则返回原轮廓

    def resize_to_fit(img, target_size=(800, 600)):
        """将图像按比例缩放以适应目标大小"""
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        return resized

    gray = cv2.inRange(image, (0, 0, 0), (COLOR_B, COLOR_G, COLOR_R))
    gray = cv2.bitwise_not(gray)

    # 3. 二值化（反转颜色）
    _, binary = cv2.threshold(gray, lower_threshold, upper_threshold, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if debug:
        cv2.imshow("Step 3 - Binary Inverted", resize_to_fit(binary))
        cv2.waitKey(0)

    x1, y1 = 952, 153
    x2, y2 = 1245, 593
    # 确保坐标有效性（左上角 < 右下角）
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # 检查ROI是否在图像范围内
    height, width = binary.shape[:2]
    binary=binary[y1:y2, x1:x2]

    # 4. 查找轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        print("未检测到轮廓。")
        return {
            'contour': None,
            'outer_points': None,
            'inner_points': None,
            'binary_image': binary
        }
    hierarchy = hierarchy[0]
    # 显示所有轮廓
    all_contours_img = binary.copy()

    if debug:
        cv2.drawContours(all_contours_img, contours, -1, (0, 255, 255), 2)  # 黄色轮廓
        cv2.imshow("Step 0 - All Contours", resize_to_fit(all_contours_img))
        cv2.waitKey(0)



    # 5. 筛选内外轮廓
    valid_contours = []
    image_area = binary.shape[0] * binary.shape[1]

    # 步骤1.1：开闭操作
    # 定义闭操作核，大小可调节，越大闭合效果越明显
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # # 进行闭操作，先膨胀后腐蚀，闭合小缝隙
    # closed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    # if debug:
    #     cv2.imshow("Step 1.1 - closed", resize_to_fit(closed))
    #     cv2.waitKey(0)

    contours_closed, hierarchy_closed = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if debug:
    #     img_show = image.copy()  # 原图复制一份用于绘制
    #     cv2.drawContours(img_show, contours_closed, -1, (0, 255, 0), 2)  # 绿色轮廓，线宽3
    #
    #     cv2.imshow("Closed Contours", resize_to_fit(img_show))
    #     cv2.waitKey(0)


    # 步骤1：面积过滤
    area_filtered = []
    for i, cnt in enumerate(contours_closed):
        area = cv2.contourArea(cnt)
        if image_area * 0.8>= area >= image_area * 0.005:  # 保留面积大于图像1%的轮廓
            area_filtered.append((i, cnt))

    if debug and area_filtered:
        area_filtered_img = binary.copy()
        cnt_list = [cnt for _, cnt in area_filtered]
        cv2.drawContours(area_filtered_img, cnt_list, -1, (0, 255, 0), 2)
        cv2.imshow("Step 2 - Area Filtered", resize_to_fit(area_filtered_img))
        cv2.waitKey(0)

    #
    # # 步骤1.2：曲线处理步骤
    # processed_contours = []
    # for i, (original_idx, cnt) in enumerate(area_filtered):
    #     processed_cnt = process_contour_curve(cnt,image=image.copy())
    #     processed_contours.append((original_idx, processed_cnt))
    #
    # if debug and processed_contours:
    #     processed_img = image.copy()
    #     cnt_list = [cnt for _, cnt in processed_contours]
    #     cv2.drawContours(processed_img, cnt_list, -1, (0, 255, 0), 10)
    #     cv2.imshow("Step 2.5 - Processed Curves", resize_to_fit(processed_img))
    #     cv2.waitKey(0)
    def is_right_angle(p1, p2, p3, threshold=20):
        """判断是否为直角"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        return abs(angle - 90) < threshold

    def find_nearby_corner(image, corner, direction_vec, search_radius=15):
        """在指定方向的小范围内搜索近似直角点"""
        x0, y0 = int(corner[0]), int(corner[1])
        h, w = image.shape[:2]
        candidates = []

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                test_pt = np.array([x0 + dx, y0 + dy])
                if (0 <= test_pt[0] < w and 0 <= test_pt[1] < h):
                    vec = test_pt - corner
                    dot = np.dot(vec, direction_vec)
                    if dot > 0:  # 同方向
                        candidates.append(test_pt)

        # 简化：返回离目标方向最近的点（模拟直角判断）
        if candidates:
            # 选离中心最远者 or 中心角度最近者
            return max(candidates, key=lambda p: np.linalg.norm(p - corner))
        return None

    def get_direction_vectors(pts):
        """为左上、右上、右下、左下定义方向"""
        return [
            np.array([-1, -1]),  # 左上
            np.array([1, -1]),  # 右上
            np.array([1, 1]),  # 右下
            np.array([-1, 1])  # 左下
        ]

    def polygon_iou(p1, p2):
        """计算两个轮廓的重合度（面积 IoU）"""
        img = np.zeros((1000, 1000), dtype=np.uint8)
        cv2.fillPoly(img, [p1], 1)
        mask1 = img.copy()
        img[:] = 0
        cv2.fillPoly(img, [p2], 1)
        mask2 = img.copy()
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    # 步骤1：统计满足条件的矩形轮廓数量
    rectangle_count = 0
    valid_rectangles = []

    for i, cnt in area_filtered:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        if len(approx) != 4:
            continue  # 只保留 4 边形

        # 检查四个角是否近似直角（这里简化处理，实际应用中可能需要更精确的判断）
        is_rectangle = True
        pts = np.array([pt[0] for pt in approx], dtype=np.int32)

        for j in range(4):
            p1 = pts[j]
            p2 = pts[(j + 1) % 4]
            p3 = pts[(j + 2) % 4]

            v1 = p2 - p1
            v2 = p3 - p2
            dot_product = np.dot(v1, v2)
            magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)

            if magnitude_product == 0:
                cos_angle = 0
            else:
                cos_angle = dot_product / magnitude_product

            # 判断角度是否接近90度（允许一定误差）
            if abs(cos_angle) > 0.1:  # 90度的余弦值为0，允许±5.7度的误差
                is_rectangle = False
                break

        if is_rectangle:
            rectangle_count += 1
            valid_rectangles.append((i, cnt, approx))

    # 步骤2：如果只有一个矩形，执行角点优化和轮廓替换
    if rectangle_count == 1:
        i, cnt, approx = valid_rectangles[0]
        pts = np.array([pt[0] for pt in approx], dtype=np.int32)

        # 从原角点向四个方向寻找新角点
        dir_vectors = get_direction_vectors(pts)
        new_pts = []

        for pt, dir_vec in zip(pts, dir_vectors):
            found = find_nearby_corner(image, pt, dir_vec, search_radius=15)
            if found is None:
                break
            new_pts.append(found)

        if len(new_pts) == 4:
            new_cnt = np.array(new_pts, dtype=np.int32).reshape((-1, 1, 2))

            # 寻找重合度最高的旧轮廓进行替换
            max_iou = 0
            best_index = -1
            for j, (_, old_cnt) in enumerate(area_filtered):
                iou = polygon_iou(new_cnt, old_cnt)
                if iou > max_iou:
                    max_iou = iou
                    best_index = j

            if best_index != -1 and max_iou > 0.3:
                # 保存替换前的图像
                before_replace_img = binary.copy()
                before_cnt_list = [cnt for _, cnt in area_filtered]
                cv2.drawContours(before_replace_img, before_cnt_list, -1, (0, 255, 0), 2)

                # 替换轮廓
                area_filtered[best_index] = (i, new_cnt)
                print(f"轮廓 {best_index} 替换成功 (IoU={max_iou:.2f})")

                # 保存替换后的图像
                after_replace_img = binary.copy()
                after_cnt_list = [cnt for _, cnt in area_filtered]
                cv2.drawContours(after_replace_img, after_cnt_list, -1, (0, 255, 0), 2)

                # 在原图上标记新旧轮廓对比
                compare_img = binary.copy()
                # 用绿色绘制旧轮廓
                cv2.drawContours(compare_img, [before_cnt_list[best_index]], -1, (0, 255, 0), 2)
                # 用红色绘制新轮廓
                cv2.drawContours(compare_img, [new_cnt], -1, (0, 0, 255), 2)

                # 显示对比图像
                cv2.imshow("Before Replacement", resize_to_fit(before_replace_img))
                cv2.imshow("After Replacement", resize_to_fit(after_replace_img))
                cv2.imshow("Comparison", resize_to_fit(compare_img))
                cv2.waitKey(0)
                cv2.destroyAllWindows()


    if debug:
        # 显示最终处理后的轮廓
        area_filtered_img = binary.copy()
        cnt_list = [cnt for _, cnt in area_filtered]
        cv2.drawContours(area_filtered_img, cnt_list, -1, (0, 255, 0), 2)
        cv2.imshow("Step 2 - Area Filtered", resize_to_fit(area_filtered_img))
        cv2.waitKey(0)

    # 定义函数判断轮廓是否为近似直角矩形
    def is_approximate_rectangle(cnt, aspect_ratio_threshold=(1.414 /2.2, 1.414 * 2.2), angle_threshold=0.5):
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        # 检查是否有四个顶点
        if len(approx) != 4:
            return False, None

        # 检查宽高比
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(h) / w if w != 0 else 0
        if not (aspect_ratio_threshold[0] <= aspect_ratio <= aspect_ratio_threshold[1]):
            return False, None

        # 检查四个角是否近似直角
        pts = np.array([pt[0] for pt in approx], dtype=np.int32)
        is_rectangle = True

        for j in range(4):
            p1 = pts[j]
            p2 = pts[(j + 1) % 4]
            p3 = pts[(j + 2) % 4]

            v1 = p2 - p1
            v2 = p3 - p2
            dot_product = np.dot(v1, v2)
            magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)

            if magnitude_product == 0:
                cos_angle = 0
            else:
                cos_angle = dot_product / magnitude_product

            # 判断角度是否接近90度（允许一定误差）
            if abs(cos_angle) > angle_threshold:  # 90度的余弦值为0，允许±5.7度的误差
                is_rectangle = False
                break

        return is_rectangle, approx
    # 步骤4：层级过滤，分离内外轮廓
    valid_outer_contours = []
    valid_inner_contours = []

    # 根据轮廓面积排序并筛选近似矩形
    if area_filtered:
        # 计算每个轮廓的面积并过滤不符合条件的轮廓
        shape_filtered_with_area = []
        for i, cnt in area_filtered:
            is_rect, approx = is_approximate_rectangle(cnt)
            if is_rect:
                area = cv2.contourArea(cnt)
                shape_filtered_with_area.append((i, cnt, area))

        # 按面积降序排序
        shape_filtered_with_area.sort(key=lambda x: x[2], reverse=True)

        # 获取最大的两个矩形
        largest_rectangles = shape_filtered_with_area[:2]

        if len(largest_rectangles) >= 1:
            # 最大的矩形作为外轮廓
            valid_outer_contours.append(largest_rectangles[0])
            print("外轮廓")
            if len(largest_rectangles) >= 2:
                print("内轮廓")
                # 第二大的矩形作为内轮廓
                valid_inner_contours.append(largest_rectangles[1])


    # # 步骤3：比例过滤（接近A4比例）
    # ratio_filtered = []
    # for i, cnt in shape_filtered:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     aspect_ratio = float(w) / h if h != 0 else 0
    #     if 0.7 < aspect_ratio < 2:  # A4比例约为1:1.414，允许一定误差
    #         ratio_filtered.append((i, cnt))
    #
    # if debug and ratio_filtered:
    #     ratio_filtered_img = image.copy()
    #     cnt_list = [cnt for _, cnt in ratio_filtered]
    #     cv2.drawContours(ratio_filtered_img, cnt_list, -1, (0, 0, 255), 6)
    #     cv2.imshow("Step 4 - Ratio Filtered", resize_to_fit(ratio_filtered_img))
    #     cv2.waitKey(0)





    if debug and valid_outer_contours:
        outer_contours_img = binary.copy()
        cnt_list = [cnt for _, cnt, _ in valid_outer_contours]
        cv2.drawContours(outer_contours_img, cnt_list, -1, (0, 255, 255), 6)  # 黄色外轮廓
        cv2.imshow("Step 5 - Outer Contours", resize_to_fit(outer_contours_img))
        cv2.waitKey(0)

    if debug and valid_inner_contours:
        inner_contours_img = binary.copy()
        cnt_list = [cnt for _, cnt, _ in valid_inner_contours]
        cv2.drawContours(inner_contours_img, cnt_list, -1, (255, 255, 0), 6)  # 青色内轮廓
        cv2.imshow("Step 6 - Inner Contours", resize_to_fit(inner_contours_img))
        cv2.waitKey(0)

    # 6. 选择最佳内外轮廓
    best_outer = None
    best_inner = None

    # 按面积排序，选择最大的外轮廓
    if valid_outer_contours:
        valid_outer_contours.sort(key=lambda x: cv2.contourArea(x[1]), reverse=True)
        best_outer = valid_outer_contours[0][1]

        # 找到对应的内轮廓（如果有）
        if valid_inner_contours:
            # 假设内轮廓是外轮廓内部最大的轮廓
            valid_inner_contours.sort(key=lambda x: cv2.contourArea(x[1]), reverse=True)
            best_inner = valid_inner_contours[0][1]

    # 7. 提取角点坐标
    outer_top_left = None
    outer_bottom_right = None
    inner_top_left = None
    inner_bottom_right = None

    if best_outer is not None:
        # 计算外轮廓边界框
        x, y, w, h = cv2.boundingRect(best_outer)
        outer_top_left = (x, y)
        outer_bottom_right = (x + w, y + h)
        if debug:
            # 绘制外轮廓
            outer_contour_img = binary.copy()
            cv2.drawContours(outer_contour_img, [best_outer], -1, (0, 0, 255), 6)  # 红色外轮廓
            cv2.imshow("Step 7 - Best Outer Contour", resize_to_fit(outer_contour_img))
            cv2.waitKey(0)

    if best_inner is not None:
        # 计算内轮廓边界框
        x, y, w, h = cv2.boundingRect(best_inner)
        inner_top_left = (x, y)
        inner_bottom_right = (x + w, y + h)
        if debug:
            # 绘制内轮廓
            inner_contour_img = binary.copy()
            cv2.drawContours(inner_contour_img, [best_inner], -1, (0, 255, 0), 6)  # 绿色内轮廓
            cv2.imshow("Step 8 - Best Inner Contour", resize_to_fit(inner_contour_img))
            cv2.waitKey(0)

    # 8. 绘制最终结果
    if best_outer is not None and best_inner is not None and debug:
        final_img = binary.copy()
        cv2.drawContours(final_img, [best_outer], -1, (0, 0, 255), 6)  # 红色外轮廓
        cv2.drawContours(final_img, [best_inner], -1, (0, 255, 0), 6)  # 绿色内轮廓

        # 绘制角点
        if outer_top_left and outer_bottom_right:
            cv2.circle(final_img, outer_top_left, 5, (255, 0, 0), -1)  # 蓝色左上角
            cv2.circle(final_img, outer_bottom_right, 5, (255, 0, 0), -1)  # 蓝色右下角

        if inner_top_left and inner_bottom_right:
            cv2.circle(final_img, inner_top_left, 5, (255, 0, 0), -1)  # 蓝色左上角
            cv2.circle(final_img, inner_bottom_right, 5, (255, 0, 0), -1)  # 蓝色右下角
        if debug:
            cv2.imshow("Step 9 - Final Result", resize_to_fit(final_img))
            cv2.waitKey(0)

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
        'binary_image': binary
    }
    # best_cnt = None
    # best_idx = -1
    # max_area = 0
    #
    # min_area_threshold = 1000  # 面积下限，可根据图像大小调整
    #
    # for idx, (cnt, hier) in enumerate(zip(contours, hierarchy)):
    #     child_idx = hier[2]
    #     if child_idx == -1:
    #         continue  # 必须有子轮廓
    #
    #     # 父轮廓筛选：面积 + 矩形
    #     parent_area = cv2.contourArea(cnt)
    #     if parent_area < min_area_threshold or not is_approx_rectangular(cnt):
    #         continue
    #
    #     # 子轮廓筛选：面积 + 矩形
    #     child_cnt = contours[child_idx]
    #     child_area = cv2.contourArea(child_cnt)
    #     if child_area < min_area_threshold or not is_approx_rectangular(child_cnt):
    #         continue
    #
    #     # 如果通过上述筛选，认为它是候选外框
    #     if parent_area > max_area:
    #         best_cnt = cnt
    #         best_idx = idx
    #         max_area = parent_area
    #
    # if best_cnt is not None:
    #     x, y, w, h = cv2.boundingRect(best_cnt)
    #     outer_top_left = (x, y)
    #     outer_bottom_right = (x + w, y + h)
    #
    #     child_idx = hierarchy[best_idx][2]
    #     if child_idx != -1:
    #         inner_cnt = contours[child_idx]
    #         x_inner, y_inner, w_inner, h_inner = cv2.boundingRect(inner_cnt)
    #         inner_top_left = (x_inner, y_inner)
    #         inner_bottom_right = (x_inner + w_inner, y_inner + h_inner)
    #     else:
    #         inner_top_left = inner_bottom_right = None
    #
    #     if draw_result:
    #         result = image.copy()
    #         cv2.drawContours(result, [best_cnt], -1, (0, 0, 255), 3)
    #         cv2.rectangle(result, outer_top_left, outer_bottom_right, (0, 255, 0), 2)
    #         if inner_top_left and inner_bottom_right:
    #             cv2.rectangle(result, inner_top_left, inner_bottom_right, (255, 0, 0), 2)
    #
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(result, 'Outer TL', outer_top_left, font, 0.5, (0, 255, 0), 2)
    #         cv2.putText(result, 'Outer BR', outer_bottom_right, font, 0.5, (0, 255, 0), 2)
    #         if inner_top_left:
    #             cv2.putText(result, 'Inner TL', inner_top_left, font, 0.5, (255, 0, 0), 2)
    #         if inner_bottom_right:
    #             cv2.putText(result, 'Inner BR', inner_bottom_right, font, 0.5, (255, 0, 0), 2)
    #
    #         cv2.imshow("Step 4 - Result with Boxes", resize_to_fit(result))
    #         cv2.waitKey(0)
    #
    #     return {
    #         'contour': best_cnt,
    #         'outer_points': {
    #             'top_left': outer_top_left,
    #             'bottom_right': outer_bottom_right
    #         },
    #         'inner_points': {
    #             'top_left': inner_top_left,
    #             'bottom_right': inner_bottom_right
    #         },
    #         'binary_image': binary
    #     }
    #
    # else:
    #     print("未检测到目标黑色外框。")
    #     return {
    #         'contour': None,
    #         'outer_points': None,
    #         'inner_points': None,
    #         'binary_image': binary
    #     }