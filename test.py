import cv2
import torchvision
import torch
from torchvision.transforms.functional import convert_image_dtype
from torchvision.io import read_image
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteHead
from functools import partial
import torch.nn as nn
from torchvision.models.detection import _utils as det_utils
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
backbone = model.backbone
anchor_generator = model.anchor_generator
size = (320, 320)
num_classes = 4
num_anchors = anchor_generator.num_anchors_per_location()
norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
out_channels = det_utils.retrieve_out_channels(backbone, size)
model = SSD(backbone, anchor_generator, size, num_classes,head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer))

model.load_state_dict(torch.load("weight/199_0.15219635752340158.pth"))
model.eval()
orig_image = cv2.imread("test_zip/test/mixed_23.jpg")

image = read_image("test_zip/test/mixed_23.jpg")
image = image.unsqueeze(0)
image = convert_image_dtype(image, dtype=torch.float)
output = model(image)[0]
boxes = output["boxes"]

scores = output["scores"]
labels = output["labels"]
iou = 0.4
boxes = boxes[scores>iou]
labels = labels[scores>iou]
scores = scores[scores>iou]
classes = ["background","apple","banana","orange"]
for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{classes[labels[i]]}: {scores[i]:.2f}"
    cv2.putText(orig_image, label,
                (int(box[0])+20, int(box[1])+40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(scores)} objects. The output image is {path}")