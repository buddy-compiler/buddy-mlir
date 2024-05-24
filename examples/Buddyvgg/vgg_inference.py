import torch
from torchvision import models
from PIL import Image
from torchvision import transforms
import json

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0, 0, 0), (1, 1, 1))])

# load image
img = Image.open("image/tulips.jpg")
# [N, C, H, W]
img = data_transform(img)  
# expand batch dimension
img = torch.unsqueeze(img, dim=0)  # add batch dim
print(img[0][0][0])
import torch.nn as nn
# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = models.vgg16()
# load model weights
model_weight_path = "vgg.pth"
model.load_state_dict(torch.load(model_weight_path))
model.avgpool=nn.Sequential()
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))  # rimove batch dim
    predict = torch.softmax(output, dim=0)  
    predict_cla = torch.argmax(predict).numpy()  # get the max
print(class_indict[str(predict_cla)], predict[predict_cla].item())
