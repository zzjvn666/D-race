import cv2
import numpy as np

from tools import angle_between,is_fully_filled,dynamic_threshold,find_squares_from_corners
import cv2
import numpy as np
import cv2
import numpy as np
from tools import *

drawing = False
roi_start = (-1, -1)
roi_end = (-1, -1)
roi_selected = False

def detect_outer_black_border_(image, draw_result=True, lower_threshold=0, upper_threshold=255, debug=True):
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

    def resize_to_fit(img, target_size=(800, 600)):
        """将图像按比例缩放以适应目标大小"""
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        return resized

    gray = cv2.GaussianBlur(cv2.inRange(image, (0, 0, 0), (70, 60, 50)), (5, 5), 3)
    gray = cv2.bitwise_not(gray)
    if debug:
        cv2.imshow("Step 2 - Grayscale", resize_to_fit(gray))
        cv2.waitKey(0)

    # 3. 二值化（反转颜色）
    _, binary = cv2.threshold(gray, lower_threshold, upper_threshold, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if debug:
        cv2.imshow("Step 3 - Binary Inverted", resize_to_fit(binary))
        cv2.waitKey(0)

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
    all_contours_img = image.copy()

    if debug:
        cv2.drawContours(all_contours_img, contours, -1, (0, 255, 255), 6)  # 黄色轮廓
        cv2.imshow("Step 0 - All Contours", resize_to_fit(all_contours_img))
        cv2.waitKey(0)



    # 5. 筛选内外轮廓
    valid_contours = []
    image_area = image.shape[0] * image.shape[1]

    # 步骤1.1：开闭操作
    # 定义闭操作核，大小可调节，越大闭合效果越明显
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # 进行闭操作，先膨胀后腐蚀，闭合小缝隙
    closed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    if debug:
        cv2.imshow("Step 1.1 - closed", resize_to_fit(closed))
        cv2.waitKey(0)
    contours_closed, hierarchy_closed = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        img_show = image.copy()  # 原图复制一份用于绘制
        cv2.drawContours(img_show, contours_closed, -1, (0, 255, 0), 6)  # 绿色轮廓，线宽3

        cv2.imshow("Closed Contours", resize_to_fit(img_show))
        cv2.waitKey(0)


    # 步骤1：面积过滤
    area_filtered = []
    for i, cnt in enumerate(contours_closed):
        area = cv2.contourArea(cnt)
        if image_area * 0.8>= area >= image_area * 0.005:  # 保留面积大于图像1%的轮廓
            area_filtered.append((i, cnt))

    if debug and area_filtered:
        area_filtered_img = image.copy()
        cnt_list = [cnt for _, cnt in area_filtered]
        cv2.drawContours(area_filtered_img, cnt_list, -1, (0, 255, 0), 6)
        cv2.imshow("Step 2 - Area Filtered", resize_to_fit(area_filtered_img))
        cv2.waitKey(0)


    # 步骤1.2：曲线处理步骤
    processed_contours = []
    for i, (original_idx, cnt) in enumerate(area_filtered):
        processed_cnt = process_contour_curve(cnt,image=image.copy())
        processed_contours.append((original_idx, processed_cnt))

    if debug and processed_contours:
        processed_img = image.copy()
        cnt_list = [cnt for _, cnt in processed_contours]
        cv2.drawContours(processed_img, cnt_list, -1, (0, 255, 0), 10)
        cv2.imshow("Step 2.5 - Processed Curves", resize_to_fit(processed_img))
        cv2.waitKey(0)

    new_contours = []

    for i, cnt in processed_contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        if len(approx) == 4:
            # 提取四个点坐标
            pts = np.array([pt[0] for pt in approx])

            x_min = np.min(pts[:, 0])
            x_max = np.max(pts[:, 0])
            y_min = np.min(pts[:, 1])
            y_max = np.max(pts[:, 1])

            # 构造严格水平垂直的新四个角点，顺时针排序
            new_pts = np.array([
                [x_min, y_min],  # 左上
                [x_max, y_min],  # 右上
                [x_max, y_max],  # 右下
                [x_min, y_max],  # 左下
            ], dtype=np.int32)

            # 转成轮廓格式 (4, 1, 2)
            new_cnt = new_pts.reshape((-1, 1, 2))

            # 添加到新的轮廓列表
            new_contours.append((i, new_cnt))



    if debug :
        shape_filtered_img = image.copy()
        cnt_list = [cnt for _, cnt in new_contours]
        cv2.drawContours(shape_filtered_img, cnt_list, -1, (0, 255, 255), 2)
        cv2.imshow("Step 3 - Shape Filtered", resize_to_fit(shape_filtered_img))
        cv2.waitKey(0)

    # 步骤4：层级过滤，分离内外轮廓
    valid_outer_contours = []
    valid_inner_contours = []

    # 根据轮廓面积排序
    if new_contours:
        # 计算每个轮廓的面积和宽高比，并过滤不符合条件的轮廓
        shape_filtered_with_area = []
        for i, cnt in new_contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(h) / w if w != 0 else 0
            # 设置宽高比阈值，例如允许偏离1.414不超过30%
            if 1.414 * 0.7 <= aspect_ratio <= 1.414 * 1.3:
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
        outer_contours_img = image.copy()
        cnt_list = [cnt for _, cnt, _ in valid_outer_contours]
        cv2.drawContours(outer_contours_img, cnt_list, -1, (0, 255, 255), 6)  # 黄色外轮廓
        cv2.imshow("Step 5 - Outer Contours", resize_to_fit(outer_contours_img))
        cv2.waitKey(0)

    if debug and valid_inner_contours:
        inner_contours_img = image.copy()
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
            outer_contour_img = image.copy()
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
            inner_contour_img = image.copy()
            cv2.drawContours(inner_contour_img, [best_inner], -1, (0, 255, 0), 6)  # 绿色内轮廓
            cv2.imshow("Step 8 - Best Inner Contour", resize_to_fit(inner_contour_img))
            cv2.waitKey(0)

    # 8. 绘制最终结果
    if best_outer is not None and best_inner is not None and debug:
        final_img = image.copy()
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
import cv2
import numpy as np


def find_shape_2(gray_img, top_left, bottom_right):
    """
    输入：
        gray_img：整张灰度图（单通道，0-255）
        top_left：内轮廓左上角 (x,y)
        bottom_right：内轮廓右下角 (x,y)
    """
    # 1. 裁剪ROI
    roi = gray_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    # cv2.imshow("Step 1: ROI", roi)
    # cv2.waitKey(0)

    # 2. 二值化（白色为前景）
    binary = dynamic_threshold(roi, method='otsu')
    # cv2.imshow("Step 2: Binary", binary)
    # cv2.waitKey(0)

    # 3. 找外轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 计算最小有效面积（根据ROI大小自适应）
    min_area = roi.shape[0] * roi.shape[1] * 0.01  # 例如，小于ROI面积1%的区域将被过滤

    # 存储所有有效轮廓及其角点信息
    valid_contours = []
    all_square_groups = []

    # 绘制所有轮廓的图像
    all_contours_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area and is_fully_filled(binary, cnt):
            valid_contours.append(cnt)
            cv2.drawContours(all_contours_img, [cnt], -1, (0, 255, 0), 2)

    if not valid_contours:
        print("没有找到满足条件的白色闭合区域")
        return None, None

    # 显示所有有效轮廓
    cv2.imshow("Step 3: All Valid Contours", all_contours_img)
    cv2.waitKey(0)

    # 对每个有效轮廓进行处理
    for i, contour in enumerate(valid_contours):
        # 5. 多边形逼近 + 顶点角度
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx = approx.reshape(-1, 2)

        angle_list = []
        for i in range(len(approx)):
            p1 = approx[i - 1]
            p2 = approx[i]
            p3 = approx[(i + 1) % len(approx)]
            ang = angle_between(p1, p2, p3)
            angle_list.append((tuple(p2), ang))

        # 6. 标注角点与角度
        contour_binary = np.zeros_like(binary)
        cv2.drawContours(contour_binary, [contour], -1, 255, -1)
        annotated = cv2.cvtColor(contour_binary, cv2.COLOR_GRAY2BGR)

        for (pt, ang) in angle_list:
            cv2.circle(annotated, pt, 5, (0, 0, 255), -1)
            cv2.putText(annotated, f"{ang:.1f}", (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # cv2.imshow(f"Step 4: Angles - Contour {i + 1}", annotated)
        # cv2.waitKey(0)

        # 7. 识别正方形区域
        max_angle = 0
        for i in angle_list:
            if i[1] > max_angle:
                max_angle = i[1]
        if max_angle > 110:
            square_groups = find_squares_from_corners(angle_list, binary)
        else:
            square_groups = find_squares_from_contour(approx)
        all_square_groups.extend(square_groups)

    # 如果没有找到正方形，返回
    if not all_square_groups:
        print("没有找到符合条件的正方形")
        return None, None

    # 8. 计算所有正方形的面积并找到最小的
    min_area = float('inf')
    min_square = None
    min_side_lengths = []

    for square in all_square_groups:
        # 使用向量叉乘计算四边形面积
        area = 0
        for i in range(4):
            x1, y1 = square[i]
            x2, y2 = square[(i + 1) % 4]
            area += (x1 * y2 - x2 * y1)
        area = abs(area) / 2.0

        # 计算边长
        side_lengths = []
        for i in range(4):
            p1 = np.array(square[i])
            p2 = np.array(square[(i + 1) % 4])
            side_lengths.append(np.linalg.norm(p1 - p2))

        if area < min_area:
            min_area = area
            min_square = square
            min_side_lengths = side_lengths

    # 计算平均边长
    avg_side_length = sum(min_side_lengths) / 4 if min_square else 0

    # 9. 只绘制最小的正方形
    final_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(final_img, valid_contours, -1, (255, 0, 0), 1)  # 绘制所有轮廓边界

    # 绘制其他正方形（灰色，半透明）
    overlay = final_img.copy()
    for square in all_square_groups:
        if square == min_square:
            continue
        for i in range(4):
            cv2.line(overlay, square[i], square[(i + 1) % 4], (128, 128, 128), 1)
    final_img = cv2.addWeighted(overlay, 0.5, final_img, 0.5, 0)

    # 特殊标记最小的正方形（红色，加粗）
    for i in range(4):
        cv2.line(final_img, min_square[i], min_square[(i + 1) % 4], (0, 0, 255), 3)
        # 标注每条边的长度
        p1, p2 = min_square[i], min_square[(i + 1) % 4]
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(final_img, f"{min_side_lengths[i]:.1f}", (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 标记最小正方形的顶点（黄色）
    for i, pt in enumerate(min_square):
        cv2.circle(final_img, pt, 7, (0, 255, 255), -1)
        cv2.putText(final_img, f"P{i + 1}", (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 添加面积和平均边长文本
    cv2.putText(final_img, f"Area: {min_area:.1f} px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(final_img, f"Avg Side: {avg_side_length:.1f} px",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Step 5: Smallest Detected Square", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return min_square, avg_side_length

if __name__ == "__main__":
    image = cv2.imread(r"E:\D-Race\C_T\2_z_2_2.png")
    result = detect_outer_black_border(image, lower_threshold=100, upper_threshold=255,debug=False)



    gray = result['binary_image']
    if result['contour']['outer'] is not None:
        print("成功提取外轮廓")
    if result['contour']['inner'] is not None:
        print("成功提取内轮廓")
    else:
        print("提取轮廓失败")
    inner_top_left = result['inner_points']['top_left']
    inner_bottom_right = result['inner_points']['bottom_right']
    min_square, avg_side_length = find_shape_2(gray, inner_top_left, inner_bottom_right)
    # shape, size = find_shape_2(gray, inner_top_left, inner_bottom_right)
    print(f" 边长/直径: {avg_side_length}")