import onnx
import torch.onnx
# import torchvision
import torch.nn as nn

from fcos_core.config import cfg
from fcos_core.modeling.detector.generalized_rcnn import GeneralizedRCNN

## invoking exporter
# Standard ImageNet input - 3 channels, 224x224,
# values don't matter as we care about network structure.
# But they can also be real inputs.
dummy_input = torch.randn(1, 3, 224, 224).cuda()
# dummy_input.to(torch.device('cuda:0'))

# Obtain your model, it can be also constructed in your script explicitly
# model = torchvision.models.alexnet(pretrained=True)
# model = torch.load(model_path)
model_path = 'models/FCOS_imprv_R_101_FPN_2x.pth'
cfg_file = 'configs/fcos/fcos_imprv_R_101_FPN_2x.yaml'
cfg.merge_from_file(cfg_file)
cfg.freeze()
# model = GeneralizedRCNN(cfg.MODEL.META_ARCHITECTURE)
model = GeneralizedRCNN(cfg)
state_dict = torch.load(model_path)
model = nn.DataParallel(model)
model.load_state_dict(state_dict, strict=False)




# Invoke export
torch.onnx.export(model.module, dummy_input, "fcos_dummy.onnx")


# ## inspect the model
# # Load the ONNX model
# model_loaded = onnx.load("fcos_dummy.onnx")

# # Check that the IR is well formed
# onnx.checker.check_model(model_loaded)

# # Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model_loaded.graph))