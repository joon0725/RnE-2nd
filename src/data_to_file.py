import numpy as np
import os
import pickle
import math
import cv2
from gensim.models.fasttext import load_facebook_model
from dev.preprocessing import getFrames2
from dev.preprocessing import faceMesh_2
from dev.preprocessing import handPose_2
from dev.preprocessing import points_to_displacement

ko_model = load_facebook_model("../ko.bin")

train_files = []
folder_list = os.listdir("../dataset")
folder_list.remove("class_label.p")
folder_list.remove("00")
folder_list.remove("17")
folder_list.remove("18")
folder_list = sorted(folder_list)
l = 100
for i in folder_list:
    try:
        k = int(i)
    except:
        continue
    file_list = sorted(os.listdir(f'../dataset/{i}'))
    train_files += [*map(lambda x: f'../dataset/{i}/'+x, file_list)][:14]
print(train_files)

label = ["안녕", "무엇이", "고기", "비빔밥", "기쁨", "취미", "나", "영화", "얼굴", "보다", "이름", "읽다", "고맙다", "같은", "미안하다", "먹다", "괜찮다", "노력하다", "다음", "나이", "다시", "얼마나", "날", "나이스", "언제", "우리", "지하철", "친근하게", "버스", "타다", "핸드폰", "어디", "번호", "위치", "안내", "책임감", "누가", "도착", "가족", "시간", "소개", "받다", "부탁", "걷다", "부모", "10분", "여동생", "공부하다", "사람", "지금", "특별한", "어제", "교육", "시험", "끝", "너", "걱정하다", "결혼", "노력", "아니", "달다", "아직", "결국", "태어나다", "성공하다", "호의", "서울", "저녁", "경험", "초대", "음식", "원하다", "방문하다", "1시간", "멀다", "좋은", "다루다"]
print(len(label))
#12, 19, 28, 33, 35, 45, 46, 53, 73, 75
k = [*map(lambda x: label[x], [12, 17, 18, 19, 28, 33, 35, 45, 46, 53, 73, 75])]
label = [ko_model.wv[x] for x in label if x not in k for i in range(14)]
print(len(label))
print(label[0].shape)

X = []
cnt = 0
for n in train_files:
    cnt += 1
    vid = getFrames2(n)
    tmp = points_to_displacement(faceMesh_2(vid), 128, handPose_2(vid), 21)
    face = np.array([tmp[j]['face'] for j in range(len(tmp))])
    hands_l = np.array([tmp[j]['hands']['left'] for j in range(len(tmp))])
    hands_r = np.array([tmp[j]['hands']['right'] for j in range(len(tmp))])
    hands = np.concatenate((hands_l, hands_r))
    print(f"{cnt}/{len(train_files)}")
    tmp = [face, hands]
    X.append(tmp)
X = np.array(X)

with open("../dataset/model_with_vector_X.p", 'wb') as f:
    pickle.dump(X, f)
