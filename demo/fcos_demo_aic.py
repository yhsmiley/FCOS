# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2, os

from fcos_core.config import cfg
from predictor_aic import AICDemo

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config_file",
        default="configs/aic/fcos_imprv_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="models/FCOS_imprv_R_50_FPN_1x.pth",
        metavar="FILE",
        help="path to the trained model",
    )
    parser.add_argument(
        "--images_dir",
        default="demo/images",
        metavar="DIR",
        help="path to demo images directory",
    )
    parser.add_argument(
        "--min_image_size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights

    cfg.freeze()

    # The following per-class thresholds are computed by maximizing
    # per-class f-measure in their precision-recall curve.
    # Please see compute_thresholds_for_classes() in coco_eval.py for details.
    thresholds_for_classes = [
        0.2295444905757904, 0.2409958839416504, 0.2051820456981659, 
        0.32551994919776917, 0.6328923106193542, 0.36557620763778687,
        0.2634440064430237, 0.47974541783332825, 0.2850644588470459
    ]

    demo_im_names = os.listdir(args.images_dir)

    # prepare object that handles inference plus adds predictions on top of image
    aic_demo = AICDemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=args.min_image_size
    )

    for im_name in demo_im_names:
        img = cv2.imread(os.path.join(args.images_dir, im_name))
        if img is None:
            continue
        start_time = time.time()
        composite = aic_demo.run_on_opencv_image(img)
        print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
        cv2.imshow(im_name, composite)
    print("Press any key to exit ...")
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
        exit()

if __name__ == "__main__":
    main()

