# Author: Fangdong Wu
# Time: 04.04.2021
# This class design is referred as https://github.com/mikel-brostrom/Yolov3_DeepSort_Pytorch
# The class represent tracks

import numpy as np

class TrackState:
    #  表示轨迹状态
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    def __init__(self,  bbox, img, mean, covariance, track_id , n_init,
                 max_age):
        """
        Parameters
        ---------
        :param bbox: bounding boxes
        :param mean: 初始状态分布的平均向量
        :param covariance: 初始状态分布的方差矩阵
        :param track_id: a unique track identifier
        :param n_init:
        :param max_age:tracks' max lifespan{int}

        Attributes
        ----------
        tlwh：{x, y, width, height};x, y is top-left point coordinate of bounding boxes
        img_crop: cropped picture in bbox's region
        mean: ndarray
            a mean vector of the initial state distribution
        covariance:
        track_id: int
            a unique track identifier
        age: int
            Total number of frames since first occurance
        time_since_update: int
            Total number of frames since last measurement update or tentative time
        state: TrackState
            The current track state:{Tentative, Confirmed, Deleted}

        """
        self.tlwh = bbox
        self.img_crop = img
        self.feature = None
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self._n_init = n_init
        self._max_age = max_age
        self.center_hist = []
        self.center_hist.append([self.tlwh[0] + 1/2 * self.tlwh[2],
                                 self.tlwh[1] + 1/2 * self.tlwh[3]])

    def update_center(self):
        self.center_hist.append([self.tlwh[0] + 1 / 2 * self.tlwh[2],
                                 self.tlwh[1] + 1 / 2 * self.tlwh[3]])

    def update_tlwh(self):
        self.tlwh[0] = self.mean[0]
        self.tlwh[1] = self.mean[1]
        self.tlwh[2] = self.mean[2] * self.mean[3]
        self.tlwh[3] = self.mean[3]

    ## 目标丢失，轨迹停止
    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    #   return轨迹状态
    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def update_matched(self, detection):
        self.tlwh = detection.tlwh
        self.update_center()
        self.img_crop = detection.img_crop
        self.age += 1
        self.time_since_update = 0
        self.hits += 1


    def update_unmatched(self):
        self.hits = 0
        self.time_since_update += 1
        self.age += 1