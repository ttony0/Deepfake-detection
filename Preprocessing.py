import os
import cv2
import glob
import random
import numpy as np
import face_recognition
from Config import *
from tqdm import tqdm

PROCESS_NUM = None


def video_filter(video_list):
    frame = []
    for video in tqdm(video_list, desc='filtering video'):
        cap = cv2.VideoCapture(video)
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < 150 or not cap.isOpened():
            video_list.remove(video)
        else:
            frame.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    # print("frames:", frame)
    print("number of video:", len(video_list))
    print("average frame per video:", np.mean(frame))
    cap.release()
    return video_list


def frames_extract(video):
    vc = cv2.VideoCapture(video)
    frames = []
    exist, frame = vc.read()
    if not exist:
        return False, frames
    while exist:
        frames.append(frame)
        exist, frame = vc.read()
    return True, frames


def extract_face(video_list, out_dir):
    for idx, video in enumerate(video_list):
        out_path = os.path.join(out_dir, video.split('\\')[-1])
        cap = cv2.VideoCapture(out_path)
        if os.path.exists(out_path):
            if cap.isOpened():
                print(video.split('\\')[-1], 'file already exist!')
                cap.release()
                continue
            else:
                print(video.split('\\')[-1], 'file is corrupted!')
                cap.release()
                os.remove(out_path)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112))
        flag, frames = frames_extract(video)
        if not flag:
            print(video.split('\\')[-1], 'file read failed!')
            continue
        else:
            frames = tqdm(frames, desc='Processing ' + video.split('\\')[-1] + '(' + str(idx + 1) + '/' + str(
                len(video_list)) + ')', unit='img')
            for i, frame in enumerate(frames):
                face = face_recognition.face_locations(frame)
                if len(face) != 0:
                    top, right, bottom, left = face[0]
                try:
                    out.write(cv2.resize(frame[top:bottom, left:right, :], (112, 112)))
                except:
                    pass
        out.release()


def data_processing():
    video_list = []
    for dir in RESULT_GLOB_PATH.values():
        for path in dir:
            video_list += sorted(glob.glob(path))
    video_list = video_filter(video_list)
    random.shuffle(video_list)
    train_list = video_list[:int(p * len(video_list))]
    test_list = video_list[int(p * len(video_list)):]
    print('train data:', len(train_list))
    print('test_data:', len(test_list))
    print('')

    return train_list, test_list


if __name__ == '__main__':
    # real facial video extract
    video_list = []
    for path in DATA_GLOB_PATH['REAL']:
        video_list += sorted(glob.glob(path))
    video_list = video_filter(video_list)[:PROCESS_NUM]
    extract_face(video_list, RESULT_PATH['REAL'])

    # fake facial video extract
    video_list = []
    for path in DATA_GLOB_PATH['FAKE']:
        video_list += sorted(glob.glob(path))
    video_list = video_filter(video_list)[:PROCESS_NUM]
    extract_face(video_list, RESULT_PATH['FAKE'])