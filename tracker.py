# 跟 踪 器 ：运动特征跟踪器，卡尔曼滤波

import numpy as np
import kalman_filter as kf
import iou_matching as im
import track
import cv2
import cfg
import Visualize as vs
import detection

class DataLoader:
    def __init__(self, video_dir):
        self.load_video(video_dir)
        self.count = 1  # frames

    def load_video(self, video_dir):
        try:
            self.cap = cv2.VideoCapture(video_dir)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.flag, self.frame = self.cap.read()
            print("Successfully Load")
            print('fps of video: ', round(self.fps))
            width = self.frame.shape[0]
            height = self.frame.shape[1]
            print('resolution is', (width, height))
            self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        except:
            print("Can't open the video")

    def next_frame(self):
        self.flag, self.frame = self.cap.read()

class Tracker:
    def __init__(self):
        self.config = cfg.C_()
        self.inital()
        self.tracks = []
        self.num_tracks = 0
        self.num_candidates = 0
        self.KF = kf.KalmanTracker()
        self.max_age = 30
        self.init = 3
        self.matched_tracks = None
        self.unmatched_tracks = None
        self.matched_detections = None
        self.unmatched_detections = None
        self.count = 1 # frames

    def inital(self):
        self.parameter()
        self.load_video()
        self.load_detector()
        print('The tracking method is ', self.config.mode)

    def parameter(self):
        self.conf_threshold = self.config.conf_thresh
        self.nms_threshold = self.config.nms_thresh

    def load_video(self):
        try:
            self.cap = cv2.VideoCapture(self.config.video_dir)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.flag, self.frame = self.cap.read()
            print("Successfully Load the video")
            print('FPS of the video: ', round(self.fps))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
            print('Resolution is', (width, height))
            self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.out = cv2.VideoWriter(self.config.save_dir, self.fourcc, 10, (width, height))
        except:
            print("Can't load the video")

    def next_frame(self):
        self.flag, self.frame = self.cap.read()
        self.count += 1

    def load_detector(self):
        self.net = cv2.dnn.readNet(self.config.net_dir, self.config.cfg_dir)
        with open(self.config.classes_dir, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def transpose(self):
        # rotate frame
        self.frame = cv2.transpose(self.frame)

    def draw_bbox(self):
        # draw bbox
        vs.draw_bbox(self.frame, self.tracks)

    def detect(self):
        self.candidates = detection.detector(self)
        self.num_candidates = len(self.candidates)

    def create_tracks(self):
        for obj in self.unmatched_detections:
            temp = track.Track(obj.tlwh, obj.img_crop,
                               mean=None, covariance=None, track_id=None,
                               n_init=self.init, max_age=self.max_age)
            temp.mean, temp.covariance = self.KF.initiate(im.tlwh2kalman_measurement(temp.tlwh))
            self.tracks.append(obj)

    def confirmed_tracks(self, traj):
        if traj.hits == 3:
            traj.state = track.TrackState.Confirmed

    def update_matched_tracks(self):
        pass
