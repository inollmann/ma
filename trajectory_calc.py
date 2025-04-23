import pickle
import json
import math
import random
import os
import cv2 as cv
import numpy as np
from scipy.stats import zscore

# names = ('FK-Neck', 'FK-Shoulder.L', 'FK-Shoulder.R', 'FK-UpperArm.L', 'FK-UpperArm.R',
#                              'FK-Forearm.L', 'FK-Forearm.R', 'FK-Finger_Index_Carpal.L', 'FK-Finger_Index_Carpal.R',
#                              'FK-Finger_Middle_Carpal.L', 'FK-Finger_Middle_Carpal.R', 'FK-Finger_Ring_Carpal.L',
#                              'FK-Finger_Ring_Carpal.R', 'FK-Finger_Pinky_Carpal.L', 'FK-Finger_Pinky_Carpal.R',
#                              'FK-Finger_Thumb1.L', 'FK-Finger_Thumb1.R', 'FK-Finger_Index1.L', 'FK-Finger_Index1.R',
#                              'FK-Finger_Middle1.L', 'FK-Finger_Middle1.R', 'FK-Finger_Ring1.L', 'FK-Finger_Ring1.R',
#                              'FK-Finger_Pinky1.L', 'FK-Finger_Pinky1.R', 'FK-Finger_Thumb2.L', 'FK-Finger_Thumb2.R',
#                              'FK-Finger_Index2.L', 'FK-Finger_Index2.R', 'FK-Finger_Middle2.L', 'FK-Finger_Middle2.R',
#                              'FK-Finger_Ring2.L', 'FK-Finger_Ring2.R', 'FK-Finger_Pinky2.L', 'FK-Finger_Pinky2.R',
#                              'FK-Finger_Thumb3.L', 'FK-Finger_Thumb3.R', 'FK-Finger_Index3.L', 'FK-Finger_Index3.R',
#                              'FK-Finger_Middle3.L', 'FK-Finger_Middle3.R', 'FK-Finger_Ring3.L', 'FK-Finger_Ring3.R',
#                              'FK-Finger_Pinky3.L', 'FK-Finger_Pinky3.R')

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def filter_outliers(data, threshold=2):
    data_array = np.array(data)
    z_scores = np.abs(zscore(data_array, axis=0))
    filtered_data = data_array[(z_scores < threshold).all(axis=1)]
    return filtered_data


def interpolate_pair(t0, t1, dec, remove_score=False):
    inter = []
    for i, (x0, x1) in enumerate(zip(t0, t1)):
        if not (remove_score and (i+1)%3 == 0):
            inter.append(x1*dec + x0*(1-dec))

    return inter


def timeline_projection(recordings):
    traj_lengths = [len(landmarks['front']['pose']) for landmarks in recordings.values()]
    max_len = max(traj_lengths)
    projections = []
    for traj_len in traj_lengths:
        projection = [i * ((traj_len-1) / (max_len-1)) for i in range(max_len)]
        projection[-1] = round(projection[-1])
        projections.append(projection)

    return projections


def alignment(idx_x, records):
    coords = np.array([coords[idx_x:idx_x+2] for coords in records.values()])
    means = np.mean(coords, axis=0)
    diff = coords - means

    return diff.tolist()


def normalize_vectors(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def get_bone_info():
    bone_info = {
        0: {'bone': 'FK-Neck', 'in_array': ['pose', 0], 'parent': None},
        1: {'bone': 'FK-Shoulder.L', 'in_array': ['pose', 5], 'parent': None},
        2: {'bone': 'FK-Shoulder.R', 'in_array': ['pose', 2], 'parent': None},
        3: {'bone': 'FK-UpperArm.L', 'in_array': ['pose', 6], 'parent': 1},
        4: {'bone': 'FK-UpperArm.R', 'in_array': ['pose', 3], 'parent': 2},
        5: {'bone': 'FK-Forearm.L', 'in_array': ['pose', 7], 'parent': 3},
        6: {'bone': 'FK-Forearm.R', 'in_array': ['pose', 4], 'parent': 4},
        7: {'bone': 'FK-Finger_Index_Carpal.L', 'in_array': ['hand_left', 5], 'parent': 5},
        8: {'bone': 'FK-Finger_Index_Carpal.R', 'in_array': ['hand_right', 5], 'parent': 6},
        9: {'bone': 'FK-Finger_Middle_Carpal.L', 'in_array': ['hand_left', 9], 'parent': 5},
        10: {'bone': 'FK-Finger_Middle_Carpal.R', 'in_array': ['hand_right', 9], 'parent': 6},
        11: {'bone': 'FK-Finger_Ring_Carpal.L', 'in_array': ['hand_left', 13], 'parent': 5},
        12: {'bone': 'FK-Finger_Ring_Carpal.R', 'in_array': ['hand_right', 13], 'parent': 6},
        13: {'bone': 'FK-Finger_Pinky_Carpal.L', 'in_array': ['hand_left', 17], 'parent': 5},
        14: {'bone': 'FK-Finger_Pinky_Carpal.R', 'in_array': ['hand_right', 17], 'parent': 6},
        15: {'bone': 'FK-Finger_Thumb1.L', 'in_array': ['hand_left', 2], 'parent': 5},
        16: {'bone': 'FK-Finger_Thumb1.R', 'in_array': ['hand_right', 2], 'parent': 6},
        17: {'bone': 'FK-Finger_Index1.L', 'in_array': ['hand_left', 6], 'parent': 7},
        18: {'bone': 'FK-Finger_Index1.R', 'in_array': ['hand_right', 6], 'parent': 8},
        19: {'bone': 'FK-Finger_Middle1.L', 'in_array': ['hand_left', 10], 'parent': 9},
        20: {'bone': 'FK-Finger_Middle1.R', 'in_array': ['hand_right', 10], 'parent': 10},
        21: {'bone': 'FK-Finger_Ring1.L', 'in_array': ['hand_left', 14], 'parent': 11},
        22: {'bone': 'FK-Finger_Ring1.R', 'in_array': ['hand_right', 14], 'parent': 12},
        23: {'bone': 'FK-Finger_Pinky1.L', 'in_array': ['hand_left', 18], 'parent': 13},
        24: {'bone': 'FK-Finger_Pinky1.R', 'in_array': ['hand_right', 18], 'parent': 14},
        25: {'bone': 'FK-Finger_Thumb2.L', 'in_array': ['hand_left', 3], 'parent': 15},
        26: {'bone': 'FK-Finger_Thumb2.R', 'in_array': ['hand_right', 3], 'parent': 16},
        27: {'bone': 'FK-Finger_Index2.L', 'in_array': ['hand_left', 7], 'parent': 17},
        28: {'bone': 'FK-Finger_Index2.R', 'in_array': ['hand_right', 7], 'parent': 18},
        29: {'bone': 'FK-Finger_Middle2.L', 'in_array': ['hand_left', 11], 'parent': 19},
        30: {'bone': 'FK-Finger_Middle2.R', 'in_array': ['hand_right', 11], 'parent': 20},
        31: {'bone': 'FK-Finger_Ring2.L', 'in_array': ['hand_left', 15], 'parent': 21},
        32: {'bone': 'FK-Finger_Ring2.R', 'in_array': ['hand_right', 15], 'parent': 22},
        33: {'bone': 'FK-Finger_Pinky2.L', 'in_array': ['hand_left', 19], 'parent': 23},
        34: {'bone': 'FK-Finger_Pinky2.R', 'in_array': ['hand_right', 19], 'parent': 24},
        35: {'bone': 'FK-Finger_Thumb3.L', 'in_array': ['hand_left', 4], 'parent': 25},
        36: {'bone': 'FK-Finger_Thumb3.R', 'in_array': ['hand_right', 4], 'parent': 26},
        37: {'bone': 'FK-Finger_Index3.L', 'in_array': ['hand_left', 8], 'parent': 27},
        38: {'bone': 'FK-Finger_Index3.R', 'in_array': ['hand_right', 8], 'parent': 28},
        39: {'bone': 'FK-Finger_Middle3.L', 'in_array': ['hand_left', 12], 'parent': 29},
        40: {'bone': 'FK-Finger_Middle3.R', 'in_array': ['hand_right', 12], 'parent': 30},
        41: {'bone': 'FK-Finger_Ring3.L', 'in_array': ['hand_left', 16], 'parent': 31},
        42: {'bone': 'FK-Finger_Ring3.R', 'in_array': ['hand_right', 16], 'parent': 32},
        43: {'bone': 'FK-Finger_Pinky3.L', 'in_array': ['hand_left', 20], 'parent': 33},
        44: {'bone': 'FK-Finger_Pinky3.R', 'in_array': ['hand_right', 20], 'parent': 34}}
    return bone_info


class BlenderCoords:
    def __init__(self, rec, rid):
        self.rec = rec
        self.id = rid
        self.norm_xyz = self.get_norms()
        self.bone_info = get_bone_info()
        self.to_origin = self.coords_to_origin()
        self.to_parent = self.coords_to_parent()

    def coords_to_origin(self):
        rel_coords = []
        person = self.rec['person']
        # ref_ru = np.array(self.rec['front']['pose'])[:, 3:5]
        # ref_b = np.array(self.rec['side']['pose'])[:, 3]
        for bone in self.bone_info.values():
            name = bone['bone']
            part = bone['in_array'][0]
            parent = bone['parent']
            idx = bone['in_array'][1]
            # num_keypoints = num_keypoints_dict[part]
            right_up = np.array(self.rec['front'][part])[:, idx*3:idx*3+2]
            back = np.array(self.rec['side'][part])[:, idx*3]
            # if person == 'B': back = - back
            n = self.get_norms()
            coords = np.zeros((len(self.rec['front']['pose']), 3))
            coords[:, 0] = (right_up[:, 0] - n['origin']['x']) / n['delta']['x']
            coords[:, 1] = (back - n['origin']['y']) / n['delta']['y']
            coords[:, 2] = (- right_up[:, 1] + n['origin']['z']) / n['delta']['z']
            rel_coords.append({'bone': name, 'parent': parent, 'coords': coords.tolist()})

        face = np.array(self.rec['front']['face'])
        face = face.reshape(face.shape[0], -1, 3)
        face = face[:, :, :2]
        face = np.transpose(face, (1, 0, 2))
        rel_coords.append({'bone': 'face', 'parent': None, 'coords': face.tolist()})
        return {'id': self.id, 'coords': rel_coords}

    def get_norms(self):
        origin_x, origin_y, origin_z, delta_x, delta_y, delta_z = [], [], [], [], [], []
        for frame in range(len(self.rec['front']['pose'])):
            x0 = self.rec['front']['pose'][frame][1 * 3]
            x1 = self.rec['front']['pose'][frame][5 * 3]
            y0 = self.rec['side']['pose'][frame][1 * 3 + 1]
            if self.rec['person'] == 'A':
                y1 = self.rec['side']['pose'][frame][10 * 3]
            else:
                y1 = self.rec['side']['pose'][frame][13 * 3]
            z0 = self.rec['front']['pose'][frame][1 * 3 + 1]
            z1 = self.rec['front']['pose'][frame][8 * 3 + 1]
            origin_x.append(x0)
            origin_y.append(y0)
            origin_z.append(z0)
            delta_x.append(abs(x1 - x0))
            delta_y.append(y0 - y1)
            delta_z.append(abs(z1 - z0))
        return {'origin': {'x': np.array(origin_x), 'y': np.array(origin_y), 'z': np.array(origin_z)},
                'delta': {'x': np.array(delta_x), 'y': np.array(delta_y), 'z': np.array(delta_z)}}

    def coords_to_parent(self):
        rel_coords = []
        for bone_info, coords in zip(self.bone_info.values(), self.to_origin['coords']):
            name = bone_info['bone']
            coords = coords['coords']
            if bone_info['parent'] is not None:
                parent_coords = np.array(self.to_origin['coords'][bone_info['parent']]['coords'])
            else:
                parent_coords = np.array([0, 0, 0])
            coords = coords - parent_coords
            coords = normalize_vectors(coords)
            rel_coords.append({'bone': name, 'coords': coords.tolist()})
        return rel_coords


class TrajectoryCalculator:
    def __init__(self, word_file, directory='', rec_id=None, vis_only=False, print_rid_list=False):
        self.word = word_file[:-4]
        self.recordings = load_pickle(directory + 'vocab/landmarks/dw-dgs/' + word_file)

        recording_ids = list(self.recordings.keys())
        if print_rid_list:
            print(f"List of {len(recording_ids)} Recording IDs:")
            for id in recording_ids:
                print(id)

        if rec_id is not None:
            self.choice = rec_id
            if not vis_only:
                self.blender = BlenderCoords(self.recordings[rec_id], rec_id)
        else:
            success = False
            while not success:
                choice = random.choice(recording_ids)
                if not 5 < len(self.recordings[choice]['front']['pose']) < 15:
                    recording_ids.remove(choice)
                    continue

                try:
                    if not vis_only:
                        self.blender = BlenderCoords(self.recordings[choice], choice)

                    self.choice = choice
                    success = True
                except Exception as e:
                    print(e)
                    recording_ids.remove(choice)

    def visualize(self, wait_between_frames=2000):
        front = self.recordings[self.choice]['front']
        p_lines = [[0, 1], [1, 8], [1, 2], [1, 5], [2, 3], [5, 6], [3, 4], [6, 7], [0, 15], [0, 16], [15, 17], [16, 18]]
        h_lines = [[2, 3], [3, 4], [5, 6],  [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16],
                            [17, 18], [18, 19], [19, 20], [0, 1], [1, 2], [2, 5], [5, 9], [9, 13], [13, 17], [17, 0]]

        for p, hl, hr, f in zip(front['pose'], front['hand_left'], front['hand_right'], front['face']):
            wait_key = cv.waitKey(max(wait_between_frames, 0))
            if wait_key == 27:  # ESC
                break
            # take first frame average coords as base
            img_height, img_width = 800, 2000
            img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
            x_min, x_max, y_min, y_max = img_width -1, 0, img_height - 1, 0

            for i in range(int(len(p) / 3)):
                x = p[i * 3]
                y = p[i * 3 + 1]
                if x != 0:
                    x_min, x_max, y_min, y_max = min(x_min, x), max(x_max, x), min(y_min, y), max(y_max, y)

                cv.circle(img, (int(x), int(y)), 2, (0, 0, 255), thickness=2)
                if i in [1, 4, 7]:
                    text = f"{i}: ({x}, {y})"
                    cv.putText(img, text, (int(x) - 50, int(y) + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            for pair in p_lines:
                start = tuple([int(p[pair[0] * 3]), int(p[pair[0] * 3 + 1])])
                end = tuple([int(p[pair[1] * 3]), int(p[pair[1] * 3 + 1])])
                cv.line(img, start, end, (0, 0, 255), 2)

            for i in range(int(len(hl) / 3)):
                x_l = hl[i * 3]
                y_l = hl[i * 3 + 1]
                x_r = hr[i * 3]
                y_r = hr[i * 3 + 1]

                if x_l != 0:
                    x_min, x_max, y_min, y_max = min(x_min, x_l), max(x_max, x_l), min(y_min, y_l), max(y_max, y_l)
                if x_r != 0:
                    x_min, x_max, y_min, y_max = min(x_min, x_r), max(x_max, x_r), min(y_min, y_r), max(y_max, y_r)

                cv.circle(img, (int(x_l), int(y_l)), 2, (0, 255, 0), thickness=2)
                cv.circle(img, (int(x_r), int(y_r)), 2, (255, 0, 0), thickness=2)

            for pair in h_lines:
                start_l = tuple([int(hl[pair[0] * 3]), int(hl[pair[0] * 3 + 1])])
                end_l = tuple([int(hl[pair[1] * 3]), int(hl[pair[1] * 3 + 1])])
                start_r = tuple([int(hr[pair[0] * 3]), int(hr[pair[0] * 3 + 1])])
                end_r = tuple([int(hr[pair[1] * 3]), int(hr[pair[1] * 3 + 1])])
                cv.line(img, start_l, end_l, (0, 255, 0), 2)
                cv.line(img, start_r, end_r, (255, 0, 0), 2)
            
            for i in range(int(len(f) / 3)):
                x = f[i * 3]
                y = f[i * 3 + 1]
                if x != 0:
                    x_min, x_max, y_min, y_max = min(x_min, x), max(x_max, x), min(y_min, y), max(y_max, y)

                cv.circle(img, (int(x), int(y)), 2, (255, 0, 255), thickness=2)

            x_min, x_max, y_min, y_max = (max(int(x_min) - 20, 0),
                                          min(int(x_max) + 20, img_width - 1),
                                          max(int(y_min) - 20, 0),
                                          min(int(y_max) + 20, img_height - 1))
            win_name = self.word + " " + self.choice
            cv.imshow(win_name, img[y_min:y_max, x_min:x_max])
            cv.setWindowProperty(win_name, cv.WND_PROP_TOPMOST, 1)
        cv.destroyAllWindows()


if __name__ == '__main__':

    #file_list = os.listdir('vocab/landmarks/dw-dgs')
    file_list = ['abend.pkl']

    write_to_json = False
    visualize = True
    print_rec_ids = True
    time_between_frames = 500

    rec_id = "1430396 9901"             # None for random

    for file in file_list:
        print(file)

        try:
            tc = TrajectoryCalculator(file, rec_id=rec_id, print_rid_list=print_rec_ids, vis_only=not write_to_json)

            if visualize:
                tc.visualize(wait_between_frames=time_between_frames)

            if write_to_json:
                with open('blender/vocab/' + file[:-4] + '.json', 'w') as f:
                    json.dump(tc.blender.to_origin, f)

            del tc

        except Exception as e:
            print(e)
            continue



    # tc.visualize(wait_between_frames=200)
    # print(tc.find_average())
    # tc.relative_coordinates_2D()
    