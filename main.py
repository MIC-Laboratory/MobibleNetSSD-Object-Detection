
import torchvision
import transforms as T
from my_dataset import fruit_dataset
import torch
import utils
from engine import train_one_epoch,evaluate

from torchvision.models.detection.ssd import SSD

from torchvision.models.detection.ssdlite import SSDLiteHead
from functools import partial
import torch.nn as nn
from torchvision.models.detection import _utils as det_utils
from tqdm import tqdm


"""
Just for recall. We are using 4 step in ImageClassification
1. Prepare Data
2. Prepare Network
3. Define Optimizer,Loss and scheduler
4. Define TrainLoop and TestLoop

We will follow the same step in Object detection.
The different between Image classification and object detection implementaion
is you will need to download the depency and import it.

Here, I downloaded:
coco_eval.py
coco_utils.py
engine.py
transforms.py
utils.py

"""

# data preparation (Use our own dataset instead of pytorch offical)

"""
We need to import transforms.py rather than from torchvision import transforms
because object detection has the bounding box. if you rotate or filp the image.
the bounding box will also rotate or filp.
The original transforms only works for image, it didn't consider bounding box
"""

train_transforms = T.Compose([
    T.PILToTensor(),
    T.RandomHorizontalFlip(0.5)
])

test_transforms = T.Compose([
    T.PILToTensor(),
])

"""
I created my own dataset. Please check data_processing.py and my_dataset.py
to learn how to create you own dataset

The argument collate_fn you may not know
Here is the example can help you understand

Let say we have 2 images. The first image contain apple,banana,and orange
The second image only contain apple.

And we want this 2 image in a batch
The problem would be how do you stack this 2 images and the furit they contain.
you will end up
apple,banana,orange
apple

They are not align. And pytorch want every image in a batch be align

pass collate_fn can bypass this limitation

"""
train_dataset = fruit_dataset("train_zip/train",train_transforms)
test_dataset = fruit_dataset("test_zip/test",test_transforms)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

"""
You may wonder why preparing network is so complex
Here is why
In imageclassification. If you have 5 classes
you only need to do something like
model = VGG("vgg16",num_classes=5)

However, in object detection
if you look at the source code of ssdlite320_mobilenet_v3_large()

you will see this function return:
SSD(backbone, anchor_generator, size, num_classes,
                head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer), **kwargs)
And we really don't want to change any of this except for num_classes

so we copy everything over but change num_classes
A lots of code you don't know is I copy from the original ssdlite320_mobilenet_v3_large() function
"""
# load a model pre-trained on COCO
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
backbone = model.backbone

anchor_generator = model.anchor_generator

size = (320, 320)
num_classes = 4
num_anchors = anchor_generator.num_anchors_per_location()
norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

out_channels = det_utils.retrieve_out_channels(backbone, size)
defaults = {
    "score_thresh": 0.001,
    "nms_thresh": 0.55,
    "detections_per_img": 300,
    "topk_candidates": 300,
    # Rescale the input in a way compatible to the backbone:
    # The following mean/std rescale the data from [0, 1] to [-1, -1]
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}
kwargs = {**defaults}
model = SSD(backbone, anchor_generator, size, num_classes,head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),**kwargs)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model.to(device)

"""
In here I didn't define loss function becase the model
that provided by pytorch already comewith loss

If you wonder what loss they use
Check mutibox loss

"""
# construct an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)
                            
num_epochs = 200
"""
In training mode. The model will return a loss dictionary
In eval mode. The model will return the boxes,confidencial score and the classes
regression_loss means the different between the predicted bounding box and the real bounding box
classification_loss means the different between predicted classes and the labels
"""
def train_once(epoch,optimizer,network,dataloader):
    network.train()
    running_loss = 0.0

    with tqdm(total=len(dataloader)) as pbar:
        for i,data in enumerate(dataloader):
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = network(images,targets)
            regression_loss = loss_dict["bbox_regression"]
            classification_loss = loss_dict["classification"]
            loss = regression_loss + classification_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.update()
            pbar.set_description(f"Epoch: {epoch} | Avg Loss: {running_loss/(i+1):.4f}")
    return running_loss/len(dataloader)

"""
I am using the evaluate function from engine because 
unlike imageclassification, we only care about the predicted class and the label
we also care about the bounding box
so we have a concept called IoU
the evaluate function can show up the IoU value

"""
for epoch in range(num_epochs):

    loss = train_once(epoch,optimizer,model,train_loader)
    # evaluate on the test dataset

    evaluate(model, test_loader, device=device)
    # update the learning rate
    lr_scheduler.step()
    torch.save(model.state_dict(),f"weight/{epoch}_{loss}.pth")

print("That's it!")