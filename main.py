# -*- coding: utf-8 -*-
# Author: Fangdong Wu
# Time: 04.02.2021
# This fruit counting program is implemented in Python3.8

import cv2
import tracker
import track
import iou_matching as im
import cascade
import Visualize as vs

def main():
    MOT = tracker.Tracker()
    print(MOT.fourcc)
    cv2.namedWindow("Object Detection", 0)
    cv2.resizeWindow("Object Detection", 1080, 1920)
    num_obj = 0
    while MOT.flag:
        print('-----------------------------------')
        # Detection
        MOT.detect()
        MOT.num_candidates = len(MOT.candidates)
        # Tracking
        if (MOT.num_tracks == 0) & (MOT.num_candidates != 0):
            MOT.num_tracks = MOT.num_candidates
            for i in range(MOT.num_candidates):
                track_temp = track.Track(MOT.candidates[i].tlwh, MOT.candidates[i].img_crop, mean=None,
                                         covariance=None, track_id=None,
                                         n_init=3, max_age=20)
                track_temp.mean, track_temp.covariance = MOT.KF.initiate(im.tlwh2kalman_measurement(MOT.candidates[i].tlwh))
                MOT.tracks.append(track_temp)

        elif (MOT.num_tracks != 0) & (MOT.num_candidates != 0):
            # cascade matching or sort matching
            '''''''''''''###########################'''''''''''''
            MOT.unmatched_detections = MOT.candidates
            if MOT.config.mode == 'cascade':
                num_obj = cascade.match(MOT.tracks, MOT.candidates, num_obj, MOT.KF)
            else:
                print("config setting is wrong")

        MOT.num_tracks = len(MOT.tracks)
        print('Number of Fruit is', num_obj)
        # draw boxes
        MOT.draw_bbox()
        vs.draw_num_obj(MOT, num_obj)
        cv2.imshow("Object Detection", MOT.frame)
        MOT.out.write(MOT.frame)  # 写入处理好的帧
        cv2.waitKey(1)
        MOT.next_frame()
        print('The count frame is', MOT.count)
        print('-----------------------------------', '\n')
        MOT.count += 1
    MOT.cap.release()
    MOT.out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()