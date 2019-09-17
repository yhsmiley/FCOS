import os
import torch
import argparse
from fcos_core.config import cfg
from fcos_core.utils.c2_model_loading import load_c2_format


def removekey(d, listofkeys):
    r = dict(d)
    for key in r:
        print(key)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="pretrained_model/R-101.pkl",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="./pretrained_model/fcos_imprv_R_101_FPN_1x_no_last_layers.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="configs/aic/fcos_imprv_R_101_FPN_1x.yaml",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

cfg.merge_from_file(args.cfg)
_d = load_c2_format(cfg, DETECTRON_PATH)
newdict = _d

newdict['model'] = removekey(_d['model'],
                             ['cls_score.bias', 'cls_score.weight', 'bbox_pred.bias', 'bbox_pred.weight'])
torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))