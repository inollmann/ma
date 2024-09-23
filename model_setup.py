import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def create_body_landmarker(model_version):
    body_model_paths = ['./mptasks/pose_landmarker_lite.task',
                        './mptasks/pose_landmarker_full.task',
                        './mptasks/pose_landmarker_heavy.task']
    body_model_path = body_model_paths[model_version]

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=body_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1)

    body_landmarker = PoseLandmarker.create_from_options(options)

    return body_landmarker


def create_face_landmarker():
    face_model_path = 'models/mptasks/face_landmarker.task'

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1)

    face_landmarker = FaceLandmarker.create_from_options(options)

    return face_landmarker


def create_hand_landmarker():
    hand_model_path = 'models/mptasks/hand_landmarker.task'

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.5)
    hand_landmarker = HandLandmarker.create_from_options(options)

    return hand_landmarker

