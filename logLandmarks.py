#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import landmark_functions as lf


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def log_landmarks(csv_path, control_dataset=False, mirror_image=False):

    if control_dataset:
        data_dir = 'datasets/setControl/train'
        num_images = 1839
    else:
        data_dir = 'datasets/setB/asl_alphabet_train'
        num_images = 87000

    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = False
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    datagen = ImageDataGenerator(dtype='uint8')
    image_iterator = datagen.flow_from_directory(data_dir,
                                                 class_mode='categorical',
                                                 batch_size=1)
    class_indices = image_iterator.class_indices
    indices_to_class = {v: k for k, v in class_indices.items()}

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    #  ########################################################################
    # numFrames = len(file_paths)
    frame = 1


    with open(csv_path, 'w'):
        pass

    for image, class_array in image_iterator:
        class_index = np.argmax(class_array)
        letter = indices_to_class[class_index]
        image = np.squeeze(image)

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(1)
        if key == 27 or frame > num_images:  # ESC
            break

        if mirror_image:
            image = cv.flip(image, 1)  # Mirror display

        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                handedness_id = handedness.classification[0].index

                # Bounding box calculation
                brect = lf.calc_bounding_rect(debug_image, hand_landmarks)

                # Landmark calculation
                landmark_list = lf.calc_landmark_list(debug_image, hand_landmarks, include_z=True)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = lf.pre_process_landmark(landmark_list, include_z=True)

                # Write to the dataset file
                lf.logging_csv(letter, handedness_id, pre_processed_landmark_list, csv_path)

                # Drawing part
                debug_image = lf.draw_bounding_rect(use_brect, debug_image, brect)
                # debug_image = lf.draw_landmarks(debug_image, landmark_list[:][:1])
                debug_image = lf.draw_landmarks(debug_image, [row[:2] for row in landmark_list])
                debug_image = lf.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                )
            print(frame, handedness.classification[0].label, letter)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

        frame = frame + 1

    cv.destroyAllWindows()


def main():
    # log_landmarks('keypoints/keypointB_3D_left.csv', mirror_image=False)
    # log_landmarks('keypoints/keypointB_3D_right.csv', mirror_image=True)
    log_landmarks('keypoints/keypointControl_1.csv', control_dataset=True)


if __name__ == '__main__':
    main()
