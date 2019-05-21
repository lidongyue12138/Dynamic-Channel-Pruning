from CIFAR_DataLoader import CifarDataManager
from CifarNet import CifarNet

d = CifarDataManager()
model = CifarNet()

# test_data, test_label = d.test.next_batch(1000)

for i in range(100):
    batch_X, batch_y = d.train.next_batch(500)
    loss = model.train_model(batch_X, batch_y)

    if i%10==0:
        print("Loss at iter %d: %f" %(i, loss))
        # print("testing accuracy: %f" %model.test_acc(test_data, test_label))

