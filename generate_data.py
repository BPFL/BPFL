import json
import numpy as np
import torch
import torchvision
from torchvision import transforms


class FEMNIST(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)

def generate_client_data(n):
    test_data=[]
    test_label=[]
    # Get the FEMNIST dataset; we use LEAF framework(https://leaf.cmu.edu/)
    for j in range(n):
        i=np.random.randint(36)
        user_tr_data = []
        user_tr_labels = []
        f = './data/train/all_data_%d_niid_0_keep_100_train_9.json' % i
        with open(f, 'r') as myfile:
            data = myfile.read()
        obj = json.loads(data)
        for user in obj['users']:
            user_tr_data.append(obj['user_data'][user]['x'])
            user_tr_labels.append(obj['user_data'][user]['y'])
        user_te_data = []
        user_te_labels = []
        f = './data/test/all_data_%d_niid_0_keep_100_test_9.json' % i
        with open(f, 'r') as myfile:
            data = myfile.read()
        obj = json.loads(data)
        for user in obj['users']:
            user_te_data.append(obj['user_data'][user]['x'])
            user_te_labels.append(obj['user_data'][user]['y'])
        user_tr_data_tensors = []
        user_tr_label_tensors = []
        for i in range(len(user_tr_data)):
            user_tr_data_tensor = torch.from_numpy(np.array(user_tr_data[i])).type(torch.FloatTensor)
            user_tr_label_tensor = torch.from_numpy(np.array(user_tr_labels[i])).type(torch.LongTensor)
            user_tr_data_tensors.append(user_tr_data_tensor)
            user_tr_label_tensors.append(user_tr_label_tensor)
        print("number of clients: ", len(user_tr_data_tensors))
        k=np.random.randint(len(user_tr_data_tensors))
        inputs = user_tr_data_tensors[k]
        inputs=inputs.resize(len(user_tr_data_tensors[k]),1,28,28)
        targets = user_tr_label_tensors[k]

        test_data=np.array(user_te_data[k]) if len(test_data)==0 else np.concatenate((test_data,np.array(user_te_data[k])),axis=0)
        test_label=np.array(user_te_labels[k]) if len(test_label)==0 else np.concatenate((test_label,np.array(user_te_labels[k])),axis=0)
        torch_data_train = FEMNIST(inputs, targets)
        torch.save(torch_data_train,"./dataset/FEMNIST/{}/client_data_{}.dat".format(n,j))

    test_data=torch.from_numpy(np.array(test_data)).resize(len(test_data),1,28,28).type(torch.FloatTensor)
    test_label=torch.from_numpy(np.array(test_label)).type(torch.LongTensor)
    torch_data_test = FEMNIST(test_data, test_label)
    torch.save(torch_data_test, "./dateset/FEMNIST/{}/test_data.dat".format(n))

def generate_server_data(num):
    train_data = []
    train_label = []
    # Get the FEMNIST dataset; we use LEAF framework(https://leaf.cmu.edu/)
    for j in range(num):
        i = np.random.randint(36)
        user_tr_data = []
        user_tr_labels = []
        f = './data/train/all_data_%d_niid_0_keep_100_train_9.json' % i
        with open(f, 'r') as myfile:
            data = myfile.read()
        obj = json.loads(data)
        for user in obj['users']:
            user_tr_data.append(obj['user_data'][user]['x'])
            user_tr_labels.append(obj['user_data'][user]['y'])
        user_te_data = []
        user_te_labels = []
        f = './data/test/all_data_%d_niid_0_keep_100_test_9.json' % i
        with open(f, 'r') as myfile:
            data = myfile.read()
        obj = json.loads(data)
        for user in obj['users']:
            user_te_data.append(obj['user_data'][user]['x'])
            user_te_labels.append(obj['user_data'][user]['y'])
        user_tr_data_tensors = []
        user_tr_label_tensors = []
        for i in range(len(user_tr_data)):
            user_tr_data_tensor = torch.from_numpy(np.array(user_tr_data[i])).type(torch.FloatTensor)
            user_tr_label_tensor = torch.from_numpy(np.array(user_tr_labels[i])).type(torch.LongTensor)
            user_tr_data_tensors.append(user_tr_data_tensor)
            user_tr_label_tensors.append(user_tr_label_tensor)
        print("number of clients: ", len(user_tr_data_tensors))
        k = np.random.randint(len(user_tr_data_tensors))
        p=np.random.randint(len(user_tr_data_tensors[k]))
        inputs = user_tr_data_tensors[k][p]
        inputs = inputs.resize(1, 1, 28, 28)
        targets = user_tr_label_tensors[k][p]
        train_data.append(inputs.numpy())
        train_label.append(targets.numpy())
    train_data = torch.from_numpy(np.array(train_data)).resize(len(train_data), 1, 28, 28).type(torch.FloatTensor)
    train_label = torch.from_numpy(np.array(train_label)).type(torch.LongTensor)
    torch_data_test = FEMNIST(train_data, train_label)
    torch.save(torch_data_test, "./server_data/FEMNIST/server.dat")

def generate_iid_server_data(num):
    train_data_full = torchvision.datasets.MNIST(root="./dataset", train=True, download=True,
                                                 transform=transforms.ToTensor())
    train_data_size = len(train_data_full)
    split_data_size = [num,train_data_size-num]
    train_data = torch.utils.data.random_split(train_data_full, split_data_size)
    print(len(train_data[0]))
    torch.save(train_data[0], "./server_data/MNIST/server.dat")

    train_data_full = torchvision.datasets.FashionMNIST(root="./dataset", train=True, download=True,
                                                 transform=transforms.ToTensor())
    train_data_size = len(train_data_full)
    split_data_size = [num, train_data_size - num]
    train_data = torch.utils.data.random_split(train_data_full, split_data_size)
    torch.save(train_data[0], "./server_data/FMNIST/server.dat")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_data_full = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True,
                                                 transform=transform)
    train_data_size = len(train_data_full)
    split_data_size = [num, train_data_size - num]
    train_data = torch.utils.data.random_split(train_data_full, split_data_size)
    torch.save(train_data[0], "./server_data/CIFAR10/server.dat")

generate_client_data(50)
generate_client_data(100)
generate_client_data(150)
generate_client_data(200)
generate_server_data(200)
generate_iid_server_data(200)