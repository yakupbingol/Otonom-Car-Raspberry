import cv2
import numpy as np
import re
from shapely.geometry import Point
from time import sleep
from shapely.geometry.polygon import Polygon
import glob


def calculate_angle(points):


    if len(points) != 6:
        raise ValueError(
            "İki vektör arasındaki açıyı hesaplamak için tam olarak altı koordinat gerekmektedir.")

    x1, y1, x2, y2, x3, y3 = points
    # Vektörleri tanımla
    AB = np.array([x2 - x1, y2 - y1])
    AC = np.array([x3 - x1, y3 - y1])

    # İki vektör arasındaki açıyı hesapla (radyan cinsinden)
    dot_product = np.dot(AB, AC)
    norm_AB = np.linalg.norm(AB)
    norm_AC = np.linalg.norm(AC)

    if norm_AB == 0 or norm_AC == 0:
        raise ValueError("Vektör büyüklükleri sıfır olamaz.")

    cos_theta = dot_product / (norm_AB * norm_AC)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # Radyanı dereceye çevir
    angle_deg = np.degrees(angle_rad)

    return angle_deg



def is_in_area(centerx, centery, target):
    x1, y1, x2, y2, x3, y3, x4, y4 = target
    polygon = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    point = Point(centerx, centery)
    return polygon.contains(point)


def getPerspectiveTransform(image, roi):

    pts1 = np.float32([roi[0], roi[1], roi[2], roi[3]])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    transformed_frame = cv2.warpPerspective(image, matrix, (640, 480))
    return transformed_frame, matrix,  inv_matrix


def drawCirles(frame, roi):
    cv2.circle(frame, roi[0], 5, (0, 0, 255), -1)
    cv2.circle(frame, roi[1], 5, (0, 0, 255), -1)
    cv2.circle(frame, roi[2], 5, (0, 0, 255), -1)
    cv2.circle(frame, roi[3], 5, (0, 0, 255), -1)

    return frame


def proces_binary_image(image, threshold_value=130):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
        image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image


def histogram_points(binary_image):

    histogram = np.sum(binary_image[binary_image.shape[0]//2:, :], axis=0)
    midpoint = np.int32(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    left_sum = np.sum(histogram[:midpoint])//len(histogram[:midpoint])
    right_sum = np.sum(histogram[midpoint:])//len(histogram[midpoint:])

    return histogram, midpoint, left_base, right_base , left_sum , right_sum


def processFrame(frame):

    current_lane = ""
    frame = cv2.resize(frame, (640, 480))

    # original
    #tl, bl, tr, br = (180, 280), (125, 470), (500, 280), (570, 470)
    tl, bl, tr, br = (0, 280), (0, 470), (640, 280), (640, 470)
    # tl, bl, tr, br = (209, 340), (90, 442), (431, 340), (600, 442)

    original_image = frame.copy()

    original_image = drawCirles(original_image, [tl, bl, tr, br])
    new_frame = frame.copy()
    ref_center_x = 325
    ref_center_y = 450

    cv2.circle(new_frame, (int(ref_center_x), int(ref_center_y)),
               20, (0, 0, 255), thickness=-1)

    frame = drawCirles(frame, [tl, bl, tr, br])
    transformed_image, matrix,  inv_matrix = getPerspectiveTransform(
        frame, [tl, bl, tr, br])

    binary_image = proces_binary_image(transformed_image)

   # Sliding windows
    histogram, midpoint, left_base, right_base  , left_sum , right_sum = histogram_points(binary_image)
    if left_sum > right_sum:
        current_lane = "left"
    else:
        current_lane = "right"

    y = 472
    lx = []
    rx = []
    center_lefts = []
    center_rights = []
    left_rects = []
    right_rects = []

    binary_image_copy = binary_image.copy()
    while y > 0:
        # Left threshold
        img = binary_image[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(left_base - 50 + cx)
                left_base = left_base - 50 + cx

        # Right threshold
        img = binary_image[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                rx.append(right_base - 50 + cx)
                right_base = right_base - 50 + cx

        point1 = (left_base - 50, y)
        point2 = (left_base + 50, y - 40)

        # Dikdörtgenin merkezi
        center_x = (point1[0] + point2[0]) // 2
        center_y = (point1[1] + point2[1]) // 2

        center_lefts.append((center_x, center_y))
        point1 = (right_base - 50, y)
        point2 = (right_base + 50, y - 40)

        # Dikdörtgenin merkezi
        center_x = (point1[0] + point2[0]) // 2
        center_y = (point1[1] + point2[1]) // 2
        center_rights.append((center_x, center_y))

        # dikdortgenler dizi icerisinde tutulmaktadir.
        left_rects.append([[left_base-50, y], [left_base+50, y],
                          [left_base+50, y-40], [left_base-50, y-40],])
        right_rects.append([[right_base-50, y], [right_base+50, y],
                           [right_base+50, y-40], [right_base-50, y-40]])

        cv2.rectangle(binary_image_copy, (left_base-50, y),
                      (left_base+50, y-40), (255, 255, 255), 2)
        cv2.rectangle(binary_image_copy, (right_base-50, y),
                      (right_base+50, y-40), (255, 255, 255), 2)
        y = y-40

    curve_left = np.array(center_lefts, dtype=np.int32).reshape((-1, 1, 2))
    curve_left = cv2.approxPolyDP(curve_left, epsilon=20, closed=False)

    curve_right = np.array(center_rights, dtype=np.int32).reshape((-1, 1, 2))
    curve_right = cv2.approxPolyDP(curve_right, epsilon=20, closed=False)

    left_mean_x = np.sum([x[0] for x in center_lefts])//len(center_lefts)
    left_mean_y = np.sum([x[1] for x in center_lefts])//len(center_lefts)
    right_mean_x = np.sum([x[0] for x in center_rights])//len(center_rights)
    right_mean_y = np.sum([x[1] for x in center_rights])//len(center_rights)

    mean_x = (right_mean_x - left_mean_x)//2 + left_mean_x
    mean_y = (left_mean_y + right_mean_y) // 2

    combined_points = np.concatenate((curve_left, curve_right[::-1]), axis=0)

    cv2.polylines(transformed_image, [
                  combined_points], isClosed=True, color=(0, 255, 0), thickness=2)
    width, height = 40, 20
    x = int(ref_center_x - width // 2)
    y = int(ref_center_y - height // 2) - 92

    # Dikdörtgenin rengi (mavi, yeşil, kırmızı)
    color = (0, 255, 0)
    ref_rect_coords = [x, y, x+width, y, x+width, y+height, x, y+height]

    # Dikdörtgeni çiz
    cv2.rectangle(new_frame, (x, y),
                  (x + width, y + height), color, thickness=2)

    # orta noktanin donme acisinin hesabi

    result1 = cv2.resize(cv2.hconcat([frame, transformed_image]), (500, 500))
    result2 = cv2.resize(cv2.hconcat(
        [binary_image, binary_image_copy]), (500, 500))

    newwarp = cv2.warpPerspective(transformed_image, inv_matrix, (640, 480))

    sample_point = np.array([[mean_x, mean_y]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(
        np.array([sample_point]), inv_matrix)[0][0]

    mean_x = int(transformed_point[0])
    mean_y = int(transformed_point[1])

    ref_coords = (ref_center_x, ref_center_y)
    calculated_center = (mean_x, mean_y)

    ref_rectangle_center = (x + (width // 2), y + (height // 2))

    cv2.circle(new_frame, (int(ref_rectangle_center[0]), int(
        ref_rectangle_center[1])), 5, (0, 255, 0), -1)
#

    angle_direction = -1 if ref_coords[0] - calculated_center[0] > 0 else 1

    in_area = is_in_area(mean_x, mean_y + 100, ref_rect_coords)

    calculated_angle = calculate_angle(
        (ref_coords[0], ref_coords[1], calculated_center[0], calculated_center[1], ref_rectangle_center[0], ref_rectangle_center[1]))

    triangle_pts = np.array([[ref_coords[0], ref_coords[1]], [calculated_center[0], calculated_center[1]], [
                            ref_rectangle_center[0], ref_rectangle_center[1]]], dtype=np.int32)

    cv2.polylines(new_frame, [triangle_pts],
                  isClosed=True, color=(0, 123, 0), thickness=2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)  # Metnin başlangıç noktası
    font_scale = 1
    font_color = (0, 0, 255)  # Beyaz renk
    font_thickness = 2

    angle = calculated_angle * angle_direction + 90
    text = f'Angle: {angle:.2f} derece'

    cv2.putText(new_frame, text, org, font,
                font_scale, font_color, font_thickness)
    cv2.putText(new_frame, "Suan " + current_lane + " seritte", (50,90), font,
                font_scale, font_color, font_thickness)



    cv2.circle(new_frame, (int(transformed_point[0]), int(
        transformed_point[1])), 5, (0, 0, 0), -1)

    cross_size = 10
    cv2.line(new_frame, (mean_x - cross_size, mean_y),
             (mean_x + cross_size, mean_y), (0, 255, 0), thickness=2)
    cv2.line(new_frame, (mean_x, mean_y - cross_size),
             (mean_x, mean_y + cross_size), (0, 255, 0), thickness=2)

    out_img = cv2.addWeighted(new_frame, 0.7, newwarp, 0.7, 1)

    # cv2.imshow("out_image", original_image)
    # cv2.imshow("vinary image", binary_image)
    # cv2.imshow("transformed_image", transformed_image)

    return result1, result2, current_lane, calculated_angle, angle_direction, in_area, out_img






