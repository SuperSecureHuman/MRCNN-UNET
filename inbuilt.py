import torch
import torch.nn 

# import fastRCNN model
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision


model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights).cuda()
# For training
images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
images, boxes = images.cuda(), boxes.cuda()
boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
labels = torch.randint(1, 91, (4, 11)) 
labels = labels.cuda()
images = list(image for image in images)
targets = []
for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)
output = model(images, targets)
# For inference
#model.eval()
#x = [torch.rand(3, 300, 400).cuda(), torch.rand(3, 500, 400)]
#predictions = model(x)