from CIFAR_DataLoader import CifarDataManager
from CifarNet import CifarNet

d = CifarDataManager()
model = CifarNet()

# train_data, train_label = d.train.images, d.train.labels
# test_data, test_label = d.test.images, d.test.labels

for i in range(500):
    batch_X, batch_y = d.train.next_batch(100)
    loss = model.train_model(batch_X, batch_y)

    if i%50==0:
        print("Loss at iter %d: %f" %(i, loss))
        # print("testing accuracy: %f" %model.test_acc(test_data, test_label))

