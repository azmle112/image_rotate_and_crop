import numpy as np
import requests
from json import JSONDecoder
import cv2
import math


def face_api(http_url, key, secret, mark, image_name, img):

    filepath = f'./examples/{image_name.split(".")[0]}/{image_name}'

    data = {"api_key": key, "api_secret": secret, "return_landmark": mark,
            "return_attributes": "gender,age,smiling,beauty", "beauty_score_max": 5}

    img_file = {"image_file": open(filepath, 'rb')}

    response = requests.post(http_url, data=data, files=img_file)

    req_con = response.content.decode('utf-8')
    req_dict = JSONDecoder().decode(req_con)
    print(req_dict)

    face_rectangle = req_dict['faces'][0]['face_rectangle']
    width = face_rectangle['width']
    top = face_rectangle['top']
    left = face_rectangle['left']
    height = face_rectangle['height']
    crop(img, left, top, width, height, 1)

    keyPoints = ['contour_left1',
                 'contour_left2',
                 'contour_left3',
                 'contour_left4',
                 'contour_left5',
                 'contour_left6',
                 'contour_left7',
                 'contour_left8',
                 'contour_left9',
                 'contour_left10',
                 'contour_left11',
                 'contour_left12',
                 'contour_left13',
                 'contour_left14',
                 'contour_left15',
                 'contour_left16',
                 'contour_chin',
                 'contour_right1',
                 'contour_right2',
                 'contour_right3',
                 'contour_right4',
                 'contour_right5',
                 'contour_right6',
                 'contour_right7',
                 'contour_right8',
                 'contour_right9',
                 'contour_right10',
                 'contour_right11',
                 'contour_right12',
                 'contour_right13',
                 'contour_right14',
                 'contour_right15',
                 'contour_right16',
                 'left_eyebrow_left_corner',
                 'left_eyebrow_upper_left_quarter',
                 'left_eyebrow_upper_middle',
                 'left_eyebrow_upper_right_quarter',
                 'left_eyebrow_upper_right_corner',
                 'left_eyebrow_lower_left_quarter',
                 'left_eyebrow_lower_middle',
                 'left_eyebrow_lower_right_quarter',
                 'left_eyebrow_lower_right_corner',
                 'right_eyebrow_upper_left_corner',
                 'right_eyebrow_upper_left_quarter',
                 'right_eyebrow_upper_middle',
                 'right_eyebrow_upper_right_quarter',
                 'right_eyebrow_right_corner',
                 'right_eyebrow_lower_left_corner',
                 'right_eyebrow_lower_left_quarter',
                 'right_eyebrow_lower_middle',
                 'right_eyebrow_lower_right_quarter',
                 'nose_bridge1',
                 'nose_bridge2',
                 'nose_bridge3',
                 'nose_tip',
                 'nose_left_contour1',
                 'nose_left_contour2',
                 'nose_left_contour3',
                 'nose_left_contour4',
                 'nose_left_contour5',
                 'nose_middle_contour',
                 'nose_right_contour1',
                 'nose_right_contour2',
                 'nose_right_contour3',
                 'nose_right_contour4',
                 'nose_right_contour5',
                 'left_eye_left_corner',
                 'left_eye_upper_left_quarter',
                 'left_eye_top',
                 'left_eye_upper_right_quarter',
                 'left_eye_right_corner',
                 'left_eye_lower_right_quarter',
                 'left_eye_bottom',
                 'left_eye_lower_left_quarter',
                 'left_eye_pupil',
                 'left_eye_center',
                 'right_eye_left_corner',
                 'right_eye_upper_left_quarter',
                 'right_eye_top',
                 'right_eye_upper_right_quarter',
                 'right_eye_right_corner',
                 'right_eye_lower_right_quarter',
                 'right_eye_bottom',
                 'right_eye_lower_left_quarter',
                 'right_eye_pupil',
                 'right_eye_center',
                 'mouth_left_corner',
                 'mouth_upper_lip_left_contour1',
                 'mouth_upper_lip_left_contour2',
                 'mouth_upper_lip_left_contour3',
                 'mouth_upper_lip_left_contour4',
                 'mouth_right_corner',
                 'mouth_upper_lip_right_contour1',
                 'mouth_upper_lip_right_contour2',
                 'mouth_upper_lip_right_contour3',
                 'mouth_upper_lip_right_contour4',
                 'mouth_upper_lip_top',
                 'mouth_upper_lip_bottom',
                 'mouth_lower_lip_right_contour1',
                 'mouth_lower_lip_right_contour2',
                 'mouth_lower_lip_right_contour3',
                 'mouth_lower_lip_left_contour1',
                 'mouth_lower_lip_left_contour2',
                 'mouth_lower_lip_left_contour3',
                 'mouth_lower_lip_top',
                 'mouth_lower_lip_bottom']
    # keyPointsNumber = len(keyPoints)

    landmarks = req_dict['faces'][0]['landmark']
    points = []
    for keyPoint in keyPoints:
        x, y = landmarks[keyPoint]['x'], landmarks[keyPoint]['y']
        points.append((int(x), int(y)))
    # print(points)

    return points, landmarks


def crop(img, left, top, width, height, flag):
    cropped = img[top:top + height, left:left + width]
    if flag == 1:
        cv2.imwrite(f"./examples/{image_name.split('.')[0]}/cut_{image_name}", cropped)
    else:
        cv2.imwrite(f"./examples/{image_name.split('.')[0]}/cropped_{image_name}", cropped)


def visualize_landmark(img, points, flag):
    point_size = 2
    point_color = (0, 0, 255)
    for point in points:
        cv2.circle(img, point, point_size, point_color, 4)
    if flag == 1:
        cv2.imwrite(f"./examples/{image_name.split('.')[0]}/draw_{image_name}", img)
    else:
        cv2.imwrite(f"./examples/{image_name.split('.')[0]}/draw_rotated_{image_name}", img)


def align_face(img, landmarks):
    left_eye_center_x = landmarks['left_eye_center']['x']
    left_eye_center_y = landmarks['left_eye_center']['y']
    right_eye_center_x = landmarks['right_eye_center']['x']
    right_eye_center_y = landmarks['right_eye_center']['y']
    dy = right_eye_center_y - left_eye_center_y
    dx = right_eye_center_x - left_eye_center_x
    angle = math.atan2(dy, dx) * 180. / math.pi
    eye_center = ((left_eye_center_x + right_eye_center_x) // 2,
                  (left_eye_center_y + right_eye_center_y) // 2)
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    retated_img = cv2.warpAffine(img, rotate_matrix, (img.shape[1], img.shape[0]))
    cv2.imwrite(f"./examples/{image_name.split('.')[0]}/compare_{image_name}", np.hstack((img, retated_img)))
    # img_old = cv2.cvtColor(np.hstack((img, aligned_face)), cv2.COLOR_BGR2RGB)
    # img_compare = Image.fromarray(img_old)
    # img_compare.show()
    return retated_img, eye_center, angle


def rotate_landmarks(points, eye_center, angle, row):
    x2, y2 = eye_center
    angle = math.radians(angle)
    rotated_points = []
    y2 = row - y2
    for point in points:
        x1, y1 = point
        y1 = row - y1
        x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
        y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
        y = row - y
        rotated_points.append((int(x), int(y)))
    return rotated_points


def corp_face(img, landmarks, eye_center):
    mouth_upper_lip_bottom_x = landmarks['mouth_upper_lip_bottom']['x']
    mouth_upper_lip_bottom_y = landmarks['mouth_upper_lip_bottom']['y']
    mouth_lower_lip_top_x = landmarks['mouth_lower_lip_top']['x']
    mouth_lower_lip_top_y = landmarks['mouth_lower_lip_top']['y']
    lip_center = ((mouth_lower_lip_top_x + mouth_upper_lip_bottom_x) // 2,
                  (mouth_lower_lip_top_y + mouth_upper_lip_bottom_y) // 2)
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part

    w = h = int(bottom - top)
    x_min = landmarks['contour_left2']['x']
    x_max = landmarks['contour_right2']['x']
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    crop(img, left, top, w, h, 2)




http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
key = "iIh6AxQH4fsewknXJzpOgdaqGkyzOquW"
secret = "-WjhHwxMbRPS5s25h_JTY-Fah1mJ5K9e"
mark = 2
image_name = '4.jpg'
img = cv2.imread(f'./examples/{image_name.split(".")[0]}/{image_name}')
print(img.shape)
points, landmarks = face_api(http_url,key,secret,mark,image_name, img)

aligned_face, eye_center, angle = align_face(img, landmarks)
visualize_landmark(img, points, 1)

rotated_points = rotate_landmarks(points, eye_center, angle, img.shape[0])
corp_face(aligned_face, landmarks, eye_center)
visualize_landmark(aligned_face, rotated_points, 2)
