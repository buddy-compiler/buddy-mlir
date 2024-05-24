import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils,models
import numpy as np
import torch.optim as optim
import os
import json
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# preprocess  data_transform
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  
                                 transforms.RandomHorizontalFlip(),  
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
image_path = "/home/dongyuqi/flower_photos/flower_photos"
train_dataset = datasets.ImageFolder(root=image_path+'/train',
                                     transform=data_transform["train"])
train_num = len(train_dataset)
flower_list = train_dataset.class_to_idx  # get class index :{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
cla_dict = dict((val, key) for key, val in flower_list.items())  

# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:  # save json
    json_file.write(json_str)
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path+'test',
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)
net = models.vgg16(pretrained=True)

net.to(device)
loss_function = nn.CrossEntropyLoss()
pata = list(net.parameters())  
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './vgg.pth'
best_acc = 0.0
for epoch in range(10):
    # train
    net.train()  # dropout
    running_loss = 0.0
    t1 = time.perf_counter()  # get train time
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter()-t1)

    # validate
    net.eval()  # remove dropout
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, acc / val_num))

print('Finished Training')

