import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# 12, 19, 28, 33, 35, 45, 46, 73, 75

def getFrames(wnum, seq): # 차례대로 단어 번호, 그 안의 영상 번호
    cap = cv2.VideoCapture(f"../dataset/{str(wnum).zfill(2)}/{str(seq).zfill(2)}_{str(wnum).zfill(2)}.MP4")
    max_len = 100

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []

    ftime = 0
    fgap = 1
    if int(fps) > 15:
        fgap = 2

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        if ftime % fgap == 0:
            image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frames.append(image)
        ftime += 1

    while len(frames) < max_len:
        frames.append(0)

    return frames

def faceMesh_video(wnum, seq):
    points_num = [0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466]
    faceMesh_payload = []
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        # print(getFrames(wnum, seq)[38])
    
        for idx, frame in enumerate(getFrames(wnum, seq)):
            # Convert the BGR image to RGB before processing.
            # print()
            # print(len(facemesh_payload))
            if type(frame) == int: # 프레임이 존재하지 않으면
                if idx == 0: # 첫 프레임부터 안 보이면 0 넣는다.
                    faceMesh_payload.append(0)
                    continue
                
                # print(f'{idx}번째 프레임에는 프레임이 감지되지 않음')
                if type(faceMesh_payload[idx - 1]) == int:
                    faceMesh_payload.append(0)
                else:
                    faceMesh_payload.append(faceMesh_payload[idx-1].copy())
                continue
            
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks: # 혹은 프레임에서 얼굴 감지가 안 되면 (영상이 짧음)
                if idx == 0: # 첫 프레임부터 안 보이면 0 넣는다.
                    faceMesh_payload.append(0)
                    continue
                
                # print(f'{idx}번째 프레임에는 얼굴이 감지되지 않음')
                if type(faceMesh_payload[idx - 1]) == int:
                    faceMesh_payload.append(0)
                else:
                    faceMesh_payload.append(faceMesh_payload[idx-1].copy())
                
                continue
                
            frame_landmarks = []
            
            for index, lm in enumerate(results.multi_face_landmarks[0].landmark):
                
                if index not in points_num:
                    continue
                else:
                    x = lm.x
                    y = lm.y
                    z = lm.z
                    frame_landmarks.append([x, y, z])
                
            faceMesh_payload.append(frame_landmarks)
    
        return faceMesh_payload


def handPose_video(wnum, seq):
    handPose_payload = []  # 0번 : 왼쪽 손, 1번: 오른쪽 손

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3) as hands:

        for idx, frame in enumerate(getFrames(wnum, seq)):
            if type(frame) == int:  # case 1: 프레임이 존재하지 않으면
                if idx == 0:  # 첫 프레임부터 안 보이면 0 넣는다.
                    handPose_payload.append([0, 0])
                    continue
                #                 print(len(handpose_payload))
                # print(f'{idx}번째 프레임에는 프레임이 감지되지 않음')
                handPose_payload.append(handPose_payload[idx-1].copy())
                continue

            results = hands.process(frame)
            if not results.multi_hand_landmarks:  # case 2: 프레임은 있는데 손 인식이 아예 안됨
                if idx == 0:
                    handPose_payload.append([0, 0])
                    continue
                # print(f'{idx}번째 프레임에는 손이 둘 다 감지되지 않음')
                handPose_payload.append(handPose_payload[idx-1].copy())
                continue

            if idx == 0:
                handPose_payload.append([0, 0])
            else:
                handPose_payload.append(handPose_payload[idx-1].copy())

            for hand in results.multi_handedness:  # case 3: 감지된 손 다 돌면서 처리
                if len(results.multi_handedness) == 1:  # 감지된 손이 1개
                    if hand.classification[0].label == "Left":  # 감지된 한 손이 왼손
                        # print(f'{idx}번째 프레임에서 왼손 감지')
                        frame_landmarks_left = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            x = lm.x
                            y = lm.y
                            z = lm.z
                            frame_landmarks_left.append([x, y, z])
                        # print(f'{len(frame_landmarks_left)}개의 특징점이 왼손에서 감지')
                        handPose_payload[idx][0] = frame_landmarks_left

                    else:  # 감지된 한 손이 오른손
                        # print(f'{idx}번째 프레임에서 오른손 감지')
                        frame_landmarks_right = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            x = lm.x
                            y = lm.y
                            z = lm.z
                            frame_landmarks_right.append([x, y, z])
                        # print(f'{len(frame_landmarks_left)}개의 특징점이 오른손에서 감지')
                        handPose_payload[idx][1] = frame_landmarks_right


                else:  # 두 손 모두 감지
                    if hand.classification[0].label == "Left":  # 감지된 한 손이 왼손
                        # print(f'{idx}번째 프레임에서 왼손 감지')
                        frame_landmarks_left = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            x = lm.x
                            y = lm.y
                            z = lm.z
                            frame_landmarks_left.append([x, y, z])
                        # print(f'{len(frame_landmarks_left)}개의 특징점이 왼손에서 감지')
                        handPose_payload[idx][0] = frame_landmarks_left

                    else:  # 감지된 한 손이 오른손
                        # print(f'{idx}번째 프레임에서 오른손 감지')
                        frame_landmarks_right = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            x = lm.x
                            y = lm.y
                            z = lm.z
                            frame_landmarks_right.append([x, y, z])
                        # print(f'{len(frame_landmarks_left)}개의 특징점이 오른손에서 감지')
                        handPose_payload[idx][1] = frame_landmarks_right

    return handPose_payload


def points_to_displacement(face_points, face_count, hand_points, hand_count): #face_count: 얼굴 특징점 갯수, hand_count: 한 손 특징점 갯수
    displacement_payload = []
    
    # face mash
        
    nz = False
    for idx, frame in enumerate(face_points):
#        print(f"얼굴 {idx}프레임: {frame}")
        if type(frame) == int:
            # print("인식 안됨")
            displacement_payload = np.append(displacement_payload, {"face":np.array([np.array([0.,0.,0.]) for _ in range(face_count)])})
        elif not nz:
#            print("처음 인식됨")
            nz = True
            displacement_payload = np.append(displacement_payload, {"face":np.array([np.array([0.,0.,0.]) for _ in range(face_count)])})
        else:
            displacements = []
            for i in range(face_count):
                # print(f"{frame[i][0]}-{points[idx][i][0]}")
                displacements.append(np.array([frame[i][0]-face_points[idx-1][i][0],
                                       frame[i][1]-face_points[idx-1][i][1],
                                       frame[i][2]-face_points[idx-1][i][2]]))
            displacements = np.array(displacements)
            # print(displacements)
            displacement_payload = np.append(displacement_payload, {"face":displacements})
            
            
    # hand pose estimation
    lr = ["left", "right"]
    nz = [False, False]
    for idx, frame in enumerate(hand_points):
        hand_displacements = {}
        for i, hand in enumerate(frame):
            if type(hand) == int:
                # print(f"{idx}: {lr[i]} 인식 안됨")
                hand_displacements[lr[i]] = np.array([np.array([0.,0.,0.]) for _ in range(hand_count)])
            elif not nz[i]:
                #print(f"{idx}: {lr[i]} 처음 인식됨")
                hand_displacements[lr[i]] = np.array([np.array([0.,0.,0.]) for _ in range(hand_count)])
                nz[i] = True
            else:
                hand_displacements[lr[i]] = []
                for j, point in enumerate(hand):
                    hand_displacements[lr[i]].append([point[0]-hand_points[idx-1][i][j][0],
                                                      point[1]-hand_points[idx-1][i][j][1],
                                                      point[2]-hand_points[idx-1][i][j][2]])
                hand_displacements[lr[i]] = np.array(hand_displacements[lr[i]])
                # print(hand_displacements)
        displacement_payload[idx]["hands"] = hand_displacements
            # print(f"{idx}: {displacement_payload[idx]['hands']}")
            
    return displacement_payload



def faceMesh_2(x):
    points_num = [0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466]
    faceMesh_payload = []
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        # print(getFrames(wnum, seq)[38])

        for idx, frame in enumerate(x):
            # Convert the BGR image to RGB before processing.
            # print()
            # print(len(facemesh_payload))
            if type(frame) == int: # 프레임이 존재하지 않으면
                if idx == 0: # 첫 프레임부터 안 보이면 0 넣는다.
                    faceMesh_payload.append(0)
                    continue

                # print(f'{idx}번째 프레임에는 프레임이 감지되지 않음')
                if type(faceMesh_payload[idx - 1]) == int:
                    faceMesh_payload.append(0)
                else:
                    faceMesh_payload.append(faceMesh_payload[idx-1].copy())
                continue
            results = face_mesh.process(frame)

            if not results.multi_face_landmarks: # 혹은 프레임에서 얼굴 감지가 안 되면 (영상이 짧음)
                if idx == 0: # 첫 프레임부터 안 보이면 0 넣는다.
                    faceMesh_payload.append(0)
                    continue

                # print(f'{idx}번째 프레임에는 얼굴이 감지되지 않음')
                if type(faceMesh_payload[idx - 1]) == int:
                    faceMesh_payload.append(0)
                else:
                    faceMesh_payload.append(faceMesh_payload[idx-1].copy())

                continue

            frame_landmarks = []

            for index, lm in enumerate(results.multi_face_landmarks[0].landmark):

                if index not in points_num:
                    continue
                else:
                    x = lm.x
                    y = lm.y
                    z = lm.z
                    frame_landmarks.append([x, y, z])

            faceMesh_payload.append(frame_landmarks)

        return faceMesh_payload



def handPose_2(x:np.ndarray):
    handPose_payload = []  # 0번 : 왼쪽 손, 1번: 오른쪽 손

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3) as hands:

        for idx, frame in enumerate(x):
            if type(frame) == int:  # case 1: 프레임이 존재하지 않으면
                if idx == 0:  # 첫 프레임부터 안 보이면 0 넣는다.
                    handPose_payload.append([0, 0])
                    continue
                #                 print(len(handpose_payload))
                # print(f'{idx}번째 프레임에는 프레임이 감지되지 않음')
                handPose_payload.append(handPose_payload[idx - 1].copy())
                continue

            results = hands.process(frame)
            if not results.multi_hand_landmarks:  # case 2: 프레임은 있는데 손 인식이 아예 안됨
                if idx == 0:
                    handPose_payload.append([0, 0])
                    continue
                # print(f'{idx}번째 프레임에는 손이 둘 다 감지되지 않음')
                handPose_payload.append(handPose_payload[idx - 1].copy())
                continue

            if idx == 0:
                handPose_payload.append([0, 0])
            else:
                handPose_payload.append(handPose_payload[idx - 1].copy())

            for hand in results.multi_handedness:  # case 3: 감지된 손 다 돌면서 처리
                if len(results.multi_handedness) == 1:  # 감지된 손이 1개
                    if hand.classification[0].label == "Left":  # 감지된 한 손이 왼손
                        # print(f'{idx}번째 프레임에서 왼손 감지')
                        frame_landmarks_left = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            x = lm.x
                            y = lm.y
                            z = lm.z
                            frame_landmarks_left.append([x, y, z])
                        # print(f'{len(frame_landmarks_left)}개의 특징점이 왼손에서 감지')
                        handPose_payload[idx][0] = frame_landmarks_left

                    else:  # 감지된 한 손이 오른손
                        # print(f'{idx}번째 프레임에서 오른손 감지')
                        frame_landmarks_right = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            x = lm.x
                            y = lm.y
                            z = lm.z
                            frame_landmarks_right.append([x, y, z])
                        # print(f'{len(frame_landmarks_left)}개의 특징점이 오른손에서 감지')
                        handPose_payload[idx][1] = frame_landmarks_right


                else:  # 두 손 모두 감지
                    if hand.classification[0].label == "Left":  # 감지된 한 손이 왼손
                        # print(f'{idx}번째 프레임에서 왼손 감지')
                        frame_landmarks_left = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            x = lm.x
                            y = lm.y
                            z = lm.z
                            frame_landmarks_left.append([x, y, z])
                        # print(f'{len(frame_landmarks_left)}개의 특징점이 왼손에서 감지')
                        handPose_payload[idx][0] = frame_landmarks_left

                    else:  # 감지된 한 손이 오른손
                        # print(f'{idx}번째 프레임에서 오른손 감지')
                        frame_landmarks_right = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            x = lm.x
                            y = lm.y
                            z = lm.z
                            frame_landmarks_right.append([x, y, z])
                        # print(f'{len(frame_landmarks_left)}개의 특징점이 오른손에서 감지')
                        handPose_payload[idx][1] = frame_landmarks_right

    return handPose_payload

def getFrames2(filename): # 차례대로 단어 번호, 그 안의 영상 번호
    cap = cv2.VideoCapture(filename)
    max_len = 100

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []

    ftime = 0
    fgap = 1
    if int(fps) > 15:
        fgap = 2

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        if ftime % fgap == 0:
            image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frames.append(image)
        ftime += 1

    while len(frames) < max_len:
        frames.append(np.zeros((720, 1280, 3), dtype=np.uint8))
    frames = np.array(frames)
    return frames
