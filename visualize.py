import argparse
import glob
import os
import time
import cv2
import tqdm
import torch
from pycocotools.coco import COCO

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from dcfs.config import get_cfg, set_global_cfg
from dcfs.engine import default_setup
from detectron2.data import MetadataCatalog
from dcfs.engine import DefaultPredictor
import random

from detectron2.utils.visualizer import ColorMode, Visualizer

class VisualizationDemo():
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, confidence_threshold):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        # SparseRCNN uses RGB input as default 
#         image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                instances = instances[instances.scores > confidence_threshold]
                predictions["instances"] = instances
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


def setup(args):
    cfg = get_cfg()
    # 允许添加新键
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/coco/dcfs_gfsod_r101_novel_10shot_step8_seedx.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--annotation",
        default="datasets/cocosplit/datasplit/5k.json",
        metavar="FILE",
        help="path to annotation file",
    )
    parser.add_argument(
        "--output",
        default="datasets/coco/det_vis",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

WINDOW_NAME = "COCO detections"

if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)
    demo = VisualizationDemo(cfg)
    

    if args.annotation:
        # import pdb
        # pdb.set_trace()
        data_dir = 'datasets/coco/trainval2014/val2014' 
        annotation_file = args.annotation
        coco = COCO(annotation_file)
        image_ids = coco.getImgIds()[:5000]  # 获取前5000张图像

        for img_id in image_ids:
            img_info = coco.loadImgs(img_id)[0]
            path = os.path.join(data_dir, img_info['file_name'])

            # save gt info
            output_dir = 'datasets/coco/gt_vis'
            image = cv2.imread(path)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                bbox = ann['bbox']
                category_id = ann['category_id']
                category_name = coco.loadCats(category_id)[0]['name']
                x, y, w, h = map(int, bbox)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            save_path = os.path.join(output_dir, img_info['file_name'])
            cv2.imwrite(save_path, image)

            # save prediction info
            img = read_image(path, format="RGB")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                out_filename = os.path.join(args.output, img_info['file_name'])
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit