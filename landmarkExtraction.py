#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import copy
import time
import pickle

from PIL import Image, ImageTk

import numpy as np
import cv2 as cv
import mediapipe as mp
import tkinter as tk

from landmark_functions import Landmarks

import landmark_functions as lf
import model_setup as ms

stop_signal = False

def main():

    global stop_signal

    from_save_point = True
    video_directory = './vocab/vid/gebaerdenlernen'
    landmark_directory = './vocab/landmarks/gebaerdenlernen/'
    save_point_file = './vocab/landmarks/save_point.json'
    use_brect = True

    # Create GUI
    root = tk.Tk()
    root.title("Landmark Extraction")
    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))

    frame_imgs = tk.Frame(root)
    frame_imgs.pack(side='top')

    label_video = tk.Label(frame_imgs)
    label_video.pack(side='top')

    frame_bottom = tk.Frame(root)
    frame_bottom.pack(side='top', fill='x', padx=10, pady=5)

    button_stop = tk.Button(frame_bottom, text="Stop", command=stop_logging,
                            activebackground='red', activeforeground='white')
    button_stop.pack(side='right', pady=5)

    label_cur = tk.Label(frame_bottom, text="")
    label_cur.pack(side='left', pady=2)

    saved_progress = load_progress(save_point_file)

    t0 = time.time()

    for idx, file in enumerate(os.listdir(video_directory)):

        if from_save_point and idx < saved_progress:
            continue

        if stop_signal:
            with open(save_point_file, 'w') as f:
                json.dump(idx, f)
            break

        if file.lower().endswith('.mp4'):
            name = file[:-4]
            # print(name)
        else:
            continue

        file_path = video_directory + '/' + file

        # Camera preparation
        cap = cv.VideoCapture(file_path)
        fps = cap.get(cv.CAP_PROP_FPS)
        vid_total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        # frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Load MediaPipe
        hands = ms.create_hand_landmarker()
        body = ms.create_body_landmarker(0)
        face = ms.create_face_landmarker()

        t1 = time.time()
        lm = Landmarks()

        while True:

            try:
                root.update()
                if not root.winfo_exists():
                    break
            except tk.TclError as e:
                break

            # Video capture #####################################################
            ret, image = cap.read()

            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False

            timestamp = int((frame_count / fps) * 1000)
            frame_count = frame_count + 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            results_hand = hands.detect_for_video(mp_image, timestamp)
            results_body = body.detect_for_video(mp_image, timestamp)
            results_face = face.detect_for_video(mp_image, timestamp)

            image.flags.writeable = True

            d_video = time.time() - t1
            d_session = time.time() - t0
            label_cur.config(
                text="Sign: {}\nFrame {} of {}\t({:.2f} s)\nVideo Time:\t {:.2f} s\nSession Time:\t {:.2f} s".format(
                name, frame_count, vid_total_frames, timestamp / 1000, d_video, d_session),
                anchor='w', justify=tk.LEFT)

            if not results_hand.handedness:
                continue

            #  ####################################################################
            lm.timestamp.append(timestamp)

            if results_body.pose_landmarks:
                norm, world, img = lf.calc_landmark_list(debug_image,
                                                         results_body.pose_landmarks[0],
                                                         results_body.pose_world_landmarks[0])
                debug_image = lf.draw_body_landmarks(debug_image, img, body_colors=[(0, 255, 0), (0, 0, 0)])
            else:
                norm, world, img = [], [], []
            lm.body.norm.append(norm)
            lm.body.world.append(world)
            lm.body.img.append(img)

            if results_face.face_landmarks:
                norm, world, img = lf.calc_landmark_list(debug_image,
                                                         results_face.face_landmarks[0],
                                                         results_face.face_landmarks[0])
                debug_image = lf.draw_face_landmarks(debug_image, img)
            else:
                norm, world, img = [], [], []
            lm.face.norm.append(norm)
            lm.face.world.append(world)
            lm.face.img.append(img)

            if results_hand.handedness:

                num_hands = len(results_hand.handedness)
                # check if handedness is different
                change_side = [0, 0]
                if num_hands == 2 and results_hand.handedness[0][0].index == results_hand.handedness[1][0].index:
                    print("GLEICH")
                    if results_hand.handedness[0][0].score < results_hand.handedness[1][0].score:
                        change_side[0] = 1
                    else:
                        change_side[1] = 1

                for i, (hand_landmarks, hand_world_landmarks, handedness) in enumerate(zip(
                        results_hand.hand_landmarks, results_hand.hand_world_landmarks, results_hand.handedness)):

                    if change_side[i]:
                        handedness_id = int(not handedness[0].index)
                        handedness_score = 0
                    else:
                        handedness_id = handedness[0].index
                        handedness_score = handedness[0].score

                    # Bounding box calculation
                    brect = lf.calc_bounding_rect(debug_image, hand_landmarks)

                    # Landmark calculation
                    if handedness_id == 0:
                        norm, world, img = lf.calc_landmark_list(debug_image, hand_landmarks, hand_world_landmarks)
                        debug_image = lf.draw_hand_landmarks(debug_image, img, hand_colors=[(0, 0, 255), (0, 0, 0)])
                        lm.hands.right.norm.append(norm)
                        lm.hands.right.world.append(world)
                        lm.hands.right.img.append(img)
                        lm.hands.right.score.append(handedness_score)
                        if num_hands == 1:
                            lm.hands.left.norm.append([])
                            lm.hands.left.world.append([])
                            lm.hands.left.img.append([])
                            lm.hands.left.score.append([])
                    else:
                        norm, world, img = lf.calc_landmark_list(debug_image, hand_landmarks, hand_world_landmarks)
                        debug_image = lf.draw_hand_landmarks(debug_image, img, hand_colors=[(255, 0, 0), (0, 0, 0)])
                        lm.hands.left.norm.append(norm)
                        lm.hands.left.world.append(world)
                        lm.hands.left.img.append(img)
                        lm.hands.left.score.append(handedness_score)
                        if num_hands == 1:
                            lm.hands.right.norm.append([])
                            lm.hands.right.world.append([])
                            lm.hands.right.img.append([])
                            lm.hands.right.score.append([])

                    debug_image = lf.draw_bounding_rect(use_brect, debug_image, brect)
            else:
                pass

            # Screen reflection #############################################################
            debug_image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
            tk_video = ImageTk.PhotoImage(Image.fromarray(debug_image))
            label_video.config(image=tk_video)

        with open(landmark_directory + name + '.pkl', 'wb') as f:
            pickle.dump(lm, f)

        cap.release()
        cv.destroyAllWindows()

    try:
        root.destroy()
    except tk.TclError as e:
        pass


def stop_logging():
    global stop_signal
    stop_signal = True

def load_progress(spf):
    if os.path.exists(spf):
        with open(spf, 'r') as f:
            return json.load(f)
    else:
        return 0


if __name__ == '__main__':
    print("Loading ...")
    main()
