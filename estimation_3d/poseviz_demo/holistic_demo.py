import poseviz
import numpy as np
import mediapipe as mp
from cv2box import CVImage

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def main():
    joint_names = ['nose',
                   'left_eye_inner', 'left_eye', 'left_eye_outer',
                   "right_eye_inner", "right_eye", "right_eye_outer",
                   "left_ear", "right_ear",
                   "mouth_left", "mouth_right",
                   "left_shoulder", "right_shoulder",
                   "left_elbow", "right_elbow",
                   "left_wrist", "right_wrist",
                   "left_pinky", "right_pinky",
                   "left_index", "right_index",
                   "left_thumb", "right_thumb",
                   "left_hip", "right_hip",
                   "left_knee", "right_knee",
                   "left_ankle", "right_ankle",
                   "left_heel", "right_heel",
                   "left_foot_index", "right_foot_index"]
    # joint_edges = [[0, 1], [0, 4], [1, 2], [2, 3], [3, 7], [4, 5], [5, 6], [6, 8], [9, 10], [18, 20], [20, 16],
    #                [18, 16], [16, 22], [16, 14], [14, 12], [12, 11], [11, 13], [13, 15], [15, 21], [15, 17], [17, 19],
    #                [12, 24], [11, 23], [23, 24], [24, 26], [23, 25], [26, 28], [25, 27], [28, 32], [28, 30], [30, 32],
    #                [27, 29], [27, 31], [29, 31]]

    viz = poseviz.PoseViz(joint_names, mp_holistic.HAND_CONNECTIONS, world_up=(0, -1, 0))

    with mp_holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        image = CVImage('').rgb
        height = image.shape[0]
        width = image.shape[1]
        print(height, width)
        results = holistic.process(image)
        # pose_33 = results.pose_world_landmarks.landmark
        right_hand_21 = results.right_hand_landmarks.landmark
        # pose_33 = results.pose_world_landmarks.landmark
        # pose_33_np = []
        right_hand_21_np = []
        # for i in range(33):
        #     pose_33_np.append([pose_33[i].x*1000, pose_33[i].y*1000, pose_33[i].z*1000+3000])

        for i in range(21):
            right_hand_21_np.append(
                [right_hand_21[i].x * 5 * width - (width // 2), right_hand_21[i].y * 5 * height - (height // 2),
                 right_hand_21[i].z * width * 5])

        print(right_hand_21_np)
        # center_x = pose_33_np[23][0] - pose_33_np[24][0]
        # center_y = pose_33_np[23][1] - pose_33_np[24][1]
        # center_z = pose_33_np[24][2] + (pose_33_np[23][2] - pose_33_np[24][2])
        # print(center_x, center_y, center_z)

        # mp_drawing.plot_landmarks(
        #     results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # # Iterate over the frames of e.g. a video
        for i in range(1):
            #     # Get the current frame
            #     frame = np.zeros([512, 512, 3], np.uint8)
            frame = image

            # Make predictions here
            # ...

            # Update the visualization
            viz.update(
                frame=frame,
                boxes=np.array([[10, 20, 100, 100]], np.float32),
                poses=[right_hand_21_np],
                camera=poseviz.Camera.from_fov(55, frame.shape[:2]))


if __name__ == '__main__':
    main()
