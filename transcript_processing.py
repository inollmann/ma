import json
import pysrt
import cv2 as cv
import numpy as np
import os
import os.path
import pickle
import csv
from PIL import Image, ImageDraw, ImageFont
from dataset import DgsDataset
from seq2seq import clean_token


def log_all_translations(start_idx, srt_folder, openpose_folder, translations_file):
    if not start_idx:
        with open(translations_file, 'wb') as f:
            pickle.dump({'transcript': [], 'index': [], 'de': [], 'dgs': [], 'mouth': []}, f)

    for i, srt_file_name in enumerate(os.listdir(srt_folder)[start_idx:]):
        srt_file = srt_folder + srt_file_name
        transcript_id = srt_file_name.split('_de.srt')[0]
        print(i+start_idx, transcript_id)
        openpose_file = openpose_folder + transcript_id + '_openpose.json'

        if not os.path.isfile(openpose_file):
            print("Nope")
            continue

        tm = TranscriptManager(openpose_file, srt_file, vis_mode=False)
        de, dgs, mouth = tm.get_translations()

        if len(de) == len(dgs):
            with open(translations_file, 'rb') as f:
                translations = pickle.load(f)

            num_translations = len(de)
            translations['transcript'] = translations['transcript'] + [tm.transcript_id] * num_translations
            translations['index'] = translations['index'] + list(range(num_translations))
            translations['de'] = translations['de'] + de
            translations['dgs'].extend(dgs)
            translations['mouth'].extend(mouth)

            with open(translations_file, 'wb') as f:
                pickle.dump(translations, f)


def log_all_gestures(start_idx, srt_folder, openpose_folder):
    for i, srt_file_name in enumerate(os.listdir(srt_folder)[start_idx:]):
        srt_file = srt_folder + srt_file_name
        transcript_id = srt_file_name.split('_de.srt')[0]
        print(i+start_idx, transcript_id)
        openpose_file = openpose_folder + transcript_id + '_openpose.json'

        if not os.path.isfile(openpose_file):
            print("Nope")
            continue

        tm = TranscriptManager(openpose_file, srt_file, vis_mode=False)
        tm.log_pose_data()


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def draw_keypoints(img, part, keypoints):
    format_dict = {'pose_keypoints_2d': [(0, 255, 0), 5],
                   'pose': [(0, 255, 0), 5],
                   'face_keypoints_2d': [(0, 255, 255), 1],
                   'face': [(0, 255, 255), 1],
                   'hand_left_keypoints_2d': [(0, 0, 255), 3],
                   'hand_left': [(0, 0, 255), 3],
                   'hand_right_keypoints_2d': [(255, 0, 0), 3],
                   'hand_right': [(255, 0, 0), 3]}

    color = format_dict[part][0]
    radius = format_dict[part][1]
    keypoints = np.array(keypoints).reshape(int(len(keypoints)/3), 3)

    if 'pose' in part:
        connections = [(1, 8),
                       (1, 2),
                       (1, 5),
                       (2, 3),
                       (5, 6),
                       (3, 4),
                       (6, 7)]
        for connection in connections:
            start = keypoints[connection[0], :]
            end = keypoints[connection[1], :]
            cv.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, min(radius, 1))

    for point in keypoints:
        cv.circle(img, (int(point[0]), int(point[1])), radius, color, -1)

    return img


def get_track(subtitle):
    if subtitle.islower() or '[MG' in subtitle or '[LM' in subtitle:   # DGS mouth
        return 2
    elif subtitle.isupper() or subtitle[0] == '$' or "||" in subtitle:  # DGS sign
        return 1
    elif subtitle[-1] in ".!?/:" or subtitle[-2] in ".!?/:":   # DE sentence
        return 0
    else:
        return 9
        # raise ValueError(f'Subtitle could not be classified: {subtitle}')


def is_processed(transcript_id):
    with open('vocab/landmarks/dw-dgs/_completed_transcripts.txt', 'r') as f:
        return transcript_id in f.read().splitlines()


def check_landmark_completion(check_token=None):
    ds = DgsDataset('vocab/translations/from_transcripts.pkl', 'models/word_embedding/cc.de.100.reduced.bin', simplify=True)
    folder = 'vocab/landmarks/dw-dgs/'
    landmarks_recorded = {'completed': [], 'pending': []}
    for token in ds.vocabulary:
        if os.path.isfile(folder + clean_token(token) + ".pkl"):
            landmarks_recorded['completed'].append(token)
        else:
            landmarks_recorded['pending'].append(token)
    print(f"{len(landmarks_recorded['completed']) / len(ds.vocabulary) * 100 :.0f}% of vocabulary completed.")

    if check_token is not None:
        completed = check_token in landmarks_recorded['completed']
        return landmarks_recorded, completed
    else:
        return landmarks_recorded


class TranscriptManager:
    def __init__(self, pose_file, subtitle_file, vis_mode=True):
        self.pose_data = read_json(pose_file)
        self.transcript_id = self.pose_data[0]['id']
        self.subtitle_data = pysrt.open(subtitle_file)
        self.num_frames = len(self.pose_data[0]['frames'])
        self.recording_length = self.subtitle_data[-1].end.ordinal / 1000
        self.fps = int((self.num_frames - 1) / self.recording_length)
        self.subtitle_dict = self.subtitle_dict()
        if vis_mode:
            self.script = self.frame2subtitles()
            self.img_size = (self.pose_data[0]['height'], self.pose_data[0]['width'], 3)


    def ms2frame(self, ms):
        return int(self.fps * ms / 1000)


    def subtitle_dict(self):
        return {e.index-1: [self.ms2frame(e.start.ordinal), self.ms2frame(e.end.ordinal), e.text[0].upper(), e.text[3:]] for e in self.subtitle_data.data}


    def frame2subtitles(self):
        script = {frame: {'A': ["", "", ""], 'B': ["", "", ""], 'C': ["", "", ""]} for frame in range(self.num_frames)}
        for _, (start, end, person, text) in self. subtitle_dict.items():
            track = get_track(text)

            if track == 9:
                text = "_"
                track = 0

            for i in range(max(start-1, 0), min(end, self.num_frames-1)):
                script[i][person][track] = text
        return script


    def visualize(self, wait_between_frames=1):

        for (idx, frame_a), (_, frame_b) in zip(self.pose_data[0]['frames'].items(), self.pose_data[1]['frames'].items()):

            wait_key = cv.waitKey(max(wait_between_frames, 0))
            if wait_key == 27:  # ESC
                break
            idx = int(idx)
            pose_a = frame_a['people'][0]
            pose_b = frame_b['people'][0]
            img_a = np.zeros(self.img_size, dtype=np.uint8)
            img_b = np.zeros(self.img_size, dtype=np.uint8)

            for (key, points_a), (_, points_b) in zip(pose_a.items(), pose_b.items()):
                if points_a:
                    img_a = draw_keypoints(img_a, key, points_a)
                    img_b = draw_keypoints(img_b, key, points_b)
            img_a = self.write_subtitles(img_a, 'A', idx)
            img_b = self.write_subtitles(img_b, 'B', idx)
            cv.imshow('DGS-Korpus', np.concatenate((img_a[:, 300:-300], img_b[:, 300:-300]), axis=1)[50:, :])
            cv.setWindowProperty('DGS-Korpus', cv.WND_PROP_TOPMOST, 1)

        cv.destroyAllWindows()


    def write_subtitles(self, img, person, idx):
        color = (255, 255, 255)
        font = ImageFont.truetype("utils/FreeMono.ttf", 15)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        subs = self.script[idx][person]
        draw.text((320, 650), subs[0], color, font)
        draw.text((320, 680), 'G: '+subs[1], color, font)
        draw.text((700, 680), 'M: '+subs[2], color, font)
        img = np.array(img_pil)
        return img


    def get_translations(self):
        sub_de, sub_dgs, sub_mouth = [], [], []
        list_de, list_dgs, list_mouth = [], [], []

        for _, subtitle in self.subtitle_dict.items():
            if get_track(subtitle[3]) ==0:
                sub_de.append(subtitle)
            elif get_track(subtitle[3]) == 9:
                subtitle[3] = " "
                sub_de.append(subtitle)
            elif get_track(subtitle[3]) == 1:
                sub_dgs.append(subtitle)
            else:
                sub_mouth.append(subtitle)

        for sentence in sub_de:
            cur_dgs, cur_mouth = [], []
            de_start, de_end, de_person, cur_de = sentence[0], sentence[1], sentence[2], sentence[3]

            if not cur_de or cur_de.isspace():
                continue

            for tokens in sub_dgs:
                start, end, person = tokens[0], tokens[1], tokens[2]
                if start >= de_start-1 and end <= de_end+1 and person == de_person:
                    cur_dgs.append(tokens[3])

            for mouthings in sub_mouth:
                start, end, person = mouthings[0], mouthings[1], mouthings[2]
                if start >= de_start-1 and end <= de_end+1 and person == de_person:
                    cur_mouth.append(mouthings[3])

            if cur_de and cur_dgs:
                list_de.append(cur_de)
                list_dgs.append(cur_dgs)
                list_mouth.append(cur_mouth)

        return list_de, list_dgs, list_mouth


    def log_pose_data(self):
        if is_processed(self.transcript_id):
            print(f"Transcript {self.transcript_id} has already been processed.")
            return

        folder = 'vocab/landmarks/dw-dgs/'
        ds = DgsDataset('vocab/translations/from_transcripts.pkl',
                        'models/word_embedding/cc.de.100.reduced.bin',
                        simplify=True)
        entry_counter = 0
        updated_tokens = []

        for _, entry in self.subtitle_dict.items():
            if len(self.pose_data) < 3: continue
            elif len(self.pose_data[2]['frames']['0']['people']) < 2: continue

            token = entry[3]

            if get_track(token) == 1 and token in ds.vocabulary:
                file_path = folder + clean_token(token) + ".pkl"
                key = self.pose_data[0]['id'] + " " + str(entry[0])

                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'rb') as f:
                            landmark_dict = pickle.load(f)
                    except PermissionError as e:
                        print(e)
                        continue

                    if len(landmark_dict) >= 100:
                        continue

                else:
                    landmark_dict = {}

                if key not in landmark_dict.keys():
                    if entry[2] == "A":
                        person = 0
                    elif entry[2] == "B":
                        person = 1
                    else:
                        raise ValueError("Person could not be identified")

                    landmarks = {'person': entry[2],
                                 'front': {'pose': [], 'face': [], 'hand_left': [], 'hand_right': []},
                                 'side': {'pose': [], 'face': [], 'hand_left': [], 'hand_right': []}}

                    for frame in range(entry[0], entry[1]):
                        if frame >= len(self.pose_data[person]['frames']) or frame >= len(self.pose_data[2]['frames']):
                            break
                        try:
                            landmarks['front']['pose'].append(
                                self.pose_data[person]['frames'][str(frame)]['people'][0]['pose_keypoints_2d'])
                            landmarks['front']['face'].append(
                                self.pose_data[person]['frames'][str(frame)]['people'][0]['face_keypoints_2d'])
                            landmarks['front']['hand_left'].append(
                                self.pose_data[person]['frames'][str(frame)]['people'][0]['hand_left_keypoints_2d'])
                            landmarks['front']['hand_right'].append(
                                self.pose_data[person]['frames'][str(frame)]['people'][0]['hand_right_keypoints_2d'])
                            landmarks['side']['pose'].append(
                                self.pose_data[2]['frames'][str(frame)]['people'][person]['pose_keypoints_2d'])
                            landmarks['side']['face'].append(
                                self.pose_data[2]['frames'][str(frame)]['people'][person]['face_keypoints_2d'])
                            landmarks['side']['hand_left'].append(
                                self.pose_data[2]['frames'][str(frame)]['people'][person]['hand_left_keypoints_2d'])
                            landmarks['side']['hand_right'].append(
                                self.pose_data[2]['frames'][str(frame)]['people'][person]['hand_right_keypoints_2d'])
                        except KeyError as e:
                            print(len(self.pose_data[2]['frames']), e)
                            continue

                    landmark_dict[key] = landmarks
                    entry_counter += 1
                    if token not in updated_tokens:
                        updated_tokens.append(token)
                try:
                    with open(file_path, 'wb') as f:
                        pickle.dump(landmark_dict, f)
                except PermissionError as e:
                    print(e)
                    continue

        with open('vocab/landmarks/dw-dgs/_completed_transcripts.txt', 'a') as f:
            f.write(self.transcript_id + "\n")

        print(f"{entry_counter} new gestures for {len(updated_tokens)} tokens added.")


#%%
# if __name__ == '__main__':
#     srt = 'transcripts/srt/'
#     openpose = 'transcripts/pose/'
#     translations = 'vocab/translations/from_transcripts.pkl'
#     start = 0
#
#     # log_all_translations(start, srt, openpose, translations)
#     # log_all_gestures(start, srt, openpose)
#     landmarks = check_landmark_completion()


#%%
if __name__ == '__main__':
    srt_folder = 'transcripts/srt/'
    openpose_folder = 'transcripts/pose/'
    tm = TranscriptManager(openpose_folder + os.listdir(openpose_folder)[0], srt_folder + os.listdir(srt_folder)[0], vis_mode=True)
    # de, dgs = tm.get_translations()
    tm.visualize(5)
    # tm.log_pose_data()
    # status = tm.check_landmark_completion()

