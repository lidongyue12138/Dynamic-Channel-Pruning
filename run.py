from CIFAR_DataLoader import CifarDataManager
from CifarNet import CifarNet

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"       # 使用第二块GPU（从0开始）

d = CifarDataManager()
model = CifarNet()

# train_data, train_label = d.train.images, d.train.labels
# test_data, test_label = d.test.images, d.test.labels

test_X, test_y = d.test.get_whole_data()

# Training
for i in range(500):
    batch_X, batch_y = d.train.next_batch(64)
    loss = model.train_model(batch_X, batch_y)

    if i%50==0:
        print("Loss at iter %d: %f" %(i, loss))
        model.test_acc(test_X, test_y)
        # print("testing accuracy: %f" %model.test_acc(test_data, test_label))

