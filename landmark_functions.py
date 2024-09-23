import csv
import numpy as np
import itertools
import copy
import cv2 as cv


# Classes

class CoordType:
    def __init__(self):
        self.norm = []
        self.world = []
        self.img = []

class CTHands(CoordType):
    def __init__(self):
        super().__init__()
        self.score = []

class HandSides:
    def __init__(self):
        self.left = CTHands()
        self.right = CTHands()

class Landmarks:
    def __init__(self):
        self.body = CoordType()
        self.face = CoordType()
        self.hands = HandSides()
        self.timestamp = []


# CSV-related functions

def read_csv(csv_path, datatype='list'):
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    match datatype:
        case 'list':
            pass
        case 'npfloat':
            data = np.array(data, dtype=float)
        case 'npstring':
            data = np.array(data)
        case _:
            pass

    return data


def write_csv(csv_path, data):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def logging_csv(letter, handedness, landmark_list, csv_path):
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([letter, handedness, *landmark_list])
    return


def array2list(arr):
    if arr.shape == (1, 42):
        arr.reshape(21, 2)
        return arr
    else:
        print("Wrong shape of landmark array")


# Landmark Preprocessing

def pre_process_landmark(landmark_list, include_z=False):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
            if include_z:
                base_z = landmark_point[2]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        if include_z:
            temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    if include_z:
        max_z = max(abs(x) for x in temp_landmark_list[2::3])
        for i in range(2, len(temp_landmark_list), 3):
            temp_landmark_list[i] /= max_z
        check = copy.deepcopy(temp_landmark_list)
        temp_landmark_list = temp_landmark_list[3:]
    else:
        temp_landmark_list = temp_landmark_list[2:]

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def conv_to_2D(landmark_list):
    landmarks_2D = []
    for row in landmark_list:
        landmarks_2D.append(row[:2])

    return landmarks_2D


def conv_to_3D_angles(landmark_list):
    coords = np.array(landmark_list, dtype="float").reshape(21, 3)
    e = {
        "vert": coords[9, :] - coords[0, :],
        "hor": coords[17, :] - coords[5, :]
    }
    x = {
        "thumb": coords[4, :] - coords[2, :],
        "index": coords[8, :] - coords[6, :],
        "middle": coords[12, :] - coords[10, :],
        "ring": coords[16, :] - coords[14, :],
        "little": coords[20, :] - coords[18, :]
    }
    n = np.cross(e["vert"], e["hor"])
    feature_list = [val/np.linalg.norm(n) for val in n]

    for finger in x:
        x_finger = x[finger]
        x_proj = x_finger - (np.dot(x_finger, n) / np.dot(n, n)) * n

        len_x = np.linalg.norm(x_finger)
        sin_phi = abs(np.dot(x_finger, n)) / (len_x * np.linalg.norm(n))
        cos_theta = np.dot(e["vert"], x_proj) / (np.linalg.norm(e["vert"]) * np.linalg.norm(x_proj))
        # phi = np.arcsin(min(1, max(-1, sin_phi)))
        # theta = np.arccos(min(1, max(-1, cos_theta)))

        feature_list.extend([sin_phi, cos_theta, len_x])

    return feature_list


def conv_to_2D_angles(landmark_list, from_3D=False):
    if from_3D:
        landmark_list = [x for i, x in enumerate(landmark_list) if (i + 1) % 3 != 0]

    feature_list = []
    reference_axis = [0, 1]
    coords = np.array(landmark_list, dtype="float").reshape(21, 2)
    x = {
        "palm": coords[9, :] - coords[0, :],
        "thumb": coords[4, :] - coords[2, :],
        "index": coords[8, :] - coords[6, :],
        "middle": coords[12, :] - coords[10, :],
        "ring": coords[16, :] - coords[14, :],
        "little": coords[20, :] - coords[18, :]
    }

    for vector in x:
        x_vector = x[vector]
        len_vector = np.linalg.norm(x_vector)
        if len_vector != 0:
            cos_theta = np.dot(x_vector, reference_axis) / (len_vector * np.linalg.norm(reference_axis))
        else:
            cos_theta = 0

        feature_list.extend([cos_theta, len_vector])

    return feature_list

# Transform to Image Coordinates

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks_norm, landmarks_world):
    image_width, image_height = image.shape[1], image.shape[0]

    landmarks_xyz = []
    landmarks_world_xyz = []
    landmarks_img = []

    for landmark_norm, landmark_world in zip(landmarks_norm, landmarks_world):
        landmark_x = min(int(landmark_norm.x * image_width), image_width - 1)
        landmark_y = min(int(landmark_norm.y * image_height), image_height - 1)

        landmarks_xyz.append([landmark_norm.x, landmark_norm.y, landmark_norm.z])
        landmarks_world_xyz.append([landmark_world.x, landmark_world.y, landmark_world.z])
        landmarks_img.append([landmark_x, landmark_y])

    return landmarks_xyz, landmarks_world_xyz, landmarks_img


def delete_wrong_classifications(landmark_array, key):
    if isinstance(key, int):
        key = str(key)
    landmark_array = np.array(landmark_array)
    mask = landmark_array[:, 1] != key
    cleaned_array = landmark_array[mask]

    return cleaned_array


def encode_handedness(landmark_array, column):
    landmark_array[landmark_array[:, 1] == 'Left', column] = '0'
    landmark_array[landmark_array[:, 1] == 'Right', column] = '1'

    return landmark_array


# Augmentation

def random_rotation(landmark_array, max_angle=180):
    lm_array = copy.deepcopy(landmark_array)
    aug_array = landmark_array[:, 2:].reshape(landmark_array.shape[0], 21, 3)
    aug_array = aug_array.astype(float)
    for i, hand in enumerate(aug_array):
        angle = np.random.uniform(np.radians(-max_angle), np.radians(max_angle))
        rot_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                          [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]])
        for j, xyz in enumerate(hand):
            xy_rot = np.dot(rot_y, xyz.T)
            aug_array[i, j] = xy_rot.T
    aug_array = aug_array.reshape(landmark_array.shape[0], 63)
    lm_array[:, 2:] = aug_array.astype('<U22')

    return lm_array


# Landmark Visualization

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text="", finger_gesture_text=""):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ': ' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture: " + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture: " + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps):
    x = 10
    y = 470
    cv.putText(image, "FPS: " + str(fps), (x, y), cv.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS: " + str(fps), (x, y), cv.FONT_HERSHEY_SIMPLEX,
               0.6, (255, 255, 255), 2, cv.LINE_AA)

    # mode_string = ['Logging Key Point', 'Logging Point History']
    # if 1 <= mode <= 2:
    #     cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
    #                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    #                cv.LINE_AA)
    #     if 0 <= number <= 9:
    #         cv.putText(image, "NUM:" + str(number), (10, 110),
    #                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    #                    cv.LINE_AA)
    return image


def draw_letters(image, letters):
    x = 10
    y = 30
    cv.putText(image, "Text: " + str(letters) + "_", (x, y), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "Text: " + str(letters) + "_", (x, y), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 255, 0), 2, cv.LINE_AA)

    return image


def draw_hand_landmarks(image, hand_landmarks, hand_colors=None):

    if hand_colors is None:
        hand_colors = [(255, 255, 255), (0, 0, 0)]

    hand_connections = [[2, 3],     # Thumb
                        [3, 4],
                        [5, 6],     # Index finger
                        [6, 7],
                        [7, 8],
                        [9, 10],    # Middle finger
                        [10, 11],
                        [11, 12],
                        [13, 14],   # Ring finger
                        [14, 15],
                        [15, 16],
                        [17, 18],   # Little finger
                        [18, 19],
                        [19, 20],
                        [0, 1],     # Palm
                        [1, 2],
                        [2, 5],
                        [5, 9],
                        [9, 13],
                        [13, 17],
                        [17, 0]]

    # Skeleton connections
    if len(hand_landmarks) > 0:
        for pair in hand_connections:
            cv.line(image, tuple(hand_landmarks[pair[0]]), tuple(hand_landmarks[pair[1]]), hand_colors[1], 6)
            cv.line(image, tuple(hand_landmarks[pair[0]]), tuple(hand_landmarks[pair[1]]), hand_colors[0], 2)

    # Key Points
    for index, landmark in enumerate(hand_landmarks):
        cv.circle(image, (landmark[0], landmark[1]), 5, hand_colors[0], -1)
        cv.circle(image, (landmark[0], landmark[1]), 5, hand_colors[1], 1)

    return image


def draw_body_landmarks(image, body_landmarks, body_colors=None):

    if body_colors is None:
        body_colors = [(255, 255, 255), (0, 0, 0)]

    body_connections = [[11, 12],   # Shoulders
                        [11, 13],   # Left upper arm
                        [13, 15],   # Left forearm
                        [12, 14],   # Right upper arm
                        [14, 16]]   # Right forearm

    # Skeleton connections
    if len(body_landmarks) > 0:
        for pair in body_connections:
            cv.line(image, tuple(body_landmarks[pair[0]]), tuple(body_landmarks[pair[1]]), body_colors[1], 6)
            cv.line(image, tuple(body_landmarks[pair[0]]), tuple(body_landmarks[pair[1]]), body_colors[0], 2)

    # Key Points
    for index, landmark in enumerate(body_landmarks):
        if 11 <= index <= 16:
            cv.circle(image, (landmark[0], landmark[1]), 5, body_colors[0], -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, body_colors[1], 1)

    return image


def draw_face_landmarks(image, face_landmarks, face_colors=None):

    if face_colors is None:
        face_colors = [(255, 255, 255), (0, 0, 0)]

    visible_landmarks = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,               # Upper lip
                         95, 88, 178, 87, 14, 317, 402, 318, 324, 306]              # Lower lip
    face_connections = [[78, 191], [191, 80], [80, 81], [81, 82], [82, 13],         # Upper lip
                        [13, 312], [312, 311], [311, 310], [310, 415], [415, 306],
                        [78, 95], [95, 88], [88, 178], [178, 87], [87, 14],         # Lower lip
                        [14, 317], [317, 402], [402, 318], [318, 324], [324, 306]]

    # Skeleton connections
    if len(face_landmarks) > 0:
        for pair in face_connections:
            cv.line(image, tuple(face_landmarks[pair[0]]), tuple(face_landmarks[pair[1]]), face_colors[1], 2)
            cv.line(image, tuple(face_landmarks[pair[0]]), tuple(face_landmarks[pair[1]]), face_colors[0], 1)

    # Key Points
    for index, landmark in enumerate(face_landmarks):
        if index in visible_landmarks or True:
            cv.circle(image, (landmark[0], landmark[1]), 1, face_colors[0], 1)     # radius 5
            # cv.circle(image, (landmark[0], landmark[1]), 1, face_colors[1], 0)      # radius 5

    return image
