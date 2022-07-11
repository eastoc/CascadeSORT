import yaml

class C_():
    def __init__(self):
        self.load("yaml/apple.yaml")

    def load(self, YAML_DIR):
        with open(YAML_DIR, 'r', encoding='utf-8') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        self.net_dir = data['YOLO']['WEIGHT']
        self.cfg_dir = data['YOLO']['CFG']
        self.classes_dir = data['YOLO']['CLASS_NAMES']
        self.conf_thresh = data['YOLO']['SCORE_THRESH']
        self.nms_thresh = data['YOLO']['NMS_THRESH']

        self.video_dir = data['TRACK']['VIDEO_DIR']
        self.mode = data['TRACK']['MODE']
        self.save_dir = data['TRACK']['SAVE_DIR']