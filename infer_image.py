import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from point_rend import add_pointrend_config
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.detection_utils import read_image

def init_point_rend(config_path, weights_path):
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

if __name__ == '__main__':

    th_instance_prob = .9

    # add path to yaml config file
    model_config_path = './configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml'
    # add path to pkl checkpoint file
    model_weights_path = './checkpoints/model_final_3c3198.pkl'

    pointrend_predictor, cfg = init_point_rend(model_config_path, model_weights_path)

    # add path to an image
    img = read_image('metallica.jpg', format="BGR")

    output = pointrend_predictor(img)

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
    cpu_device = torch.device("cpu")
    instance_mode = ColorMode.IMAGE

    instances = output["instances"].to(cpu_device)

    instances = instances[instances.scores > th_instance_prob]

    visualizer = Visualizer(img, metadata, instance_mode=instance_mode)

    vis_output = visualizer.draw_instance_predictions(predictions=instances)

    print(vis_output.get_image().shape)

    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.imshow('res', vis_output.get_image())
    cv2.waitKey(0)