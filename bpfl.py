import sys
import torchvision
from tqdm import tqdm
from torchvision import transforms
from attack import *
from util import *
from train_test import *
import copy
import argparse
from ZKP import *


parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('-data_name', type=str, default="mnist", help='mnist,fmnist,cifar10,femnist')
parser.add_argument('-client_all', type=int, default=50, help='total number of clients')
parser.add_argument('-mali_num', type=int, default=10, help='number of malicious clients')
parser.add_argument('-attack_type', type=str, default="add_noise", help='attack type')
parser.add_argument('-global_epoch', type=int, default=300, help='global_epoch')
args = parser.parse_args()
data_name=args.data_name
client_all=args.client_all
mali_num=args.mali_num
client_num=client_all-mali_num
aggregation_type="bpfl"
attack_type=args.attack_type
global_epoch=args.global_epoch
train_data = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:{}".format(device))
print(data_name,client_num,mali_num,aggregation_type,attack_type,global_epoch)

if data_name=="mnist":
    server_model = CNN().to(device).state_dict()
    global_model=copy.deepcopy(server_model)
    model_name="cnn"
    server_data = torch.load("./server_data/MNIST/server_data.dat")
    test_data = torchvision.datasets.MNIST(root="./dataset", train=False, download=True,
                                           transform=torchvision.transforms.ToTensor())
    train_data_full = torchvision.datasets.MNIST(root="./dataset", train=True, download=True,
                                            transform=transforms.ToTensor())
    train_data_size = len(train_data_full)
    client_data_size=int(train_data_size/(client_all))
    split_data_size = [client_data_size for i in range(client_all)]
    train_data = torch.utils.data.random_split(train_data_full,split_data_size)
    fed_lr=0.1
    local_epoch=5
    cos_threshold=0.9900
    euc_threshold=0.9300
elif data_name=="fmnist":
    server_model = LeNet().to(device).state_dict()
    global_model=copy.deepcopy(server_model)
    model_name="lenet"
    server_data = torch.load("./server_data/FMNIST/server_data.dat")
    test_data = torchvision.datasets.FashionMNIST(root="./dataset", train=False, download=True,
                                           transform=torchvision.transforms.ToTensor())
    train_data_full = torchvision.datasets.FashionMNIST(root="./dataset", train=True, download=True,
                                            transform=transforms.ToTensor())
    train_data_size = len(train_data_full)
    client_data_size=int(train_data_size/(client_all))
    split_data_size = [client_data_size for i in range(client_all)]
    train_data = torch.utils.data.random_split(train_data_full,split_data_size)
    fed_lr=0.1
    local_epoch=5
    cos_threshold=0.9900
    euc_threshold=0.9300
elif data_name=="femnist":
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
    server_model = LeNet().to(device).state_dict()
    global_model=copy.deepcopy(server_model)
    model_name="lenet"
    server_data = torch.load("./server_data/FEMNIST/server_data.dat")
    test_data = torch.load("./dataset/FEMNIST/{}/test_data.dat".format(client_all))
    train_data=[]
    for i in range(client_all):
        train_data.append(torch.load("./dataset/FEMNIST/{}/client_data_{}.dat".format(client_all,i)))
    fed_lr=0.1
    local_epoch=5
    cos_threshold=0.9900
    euc_threshold=0.9300
elif data_name=="cifar10":
    server_model = ResNet20().to(device).state_dict()
    global_model=copy.deepcopy(server_model)
    model_name="resnet"
    server_data = torch.load("./server_data/CIFAR10/server_data.dat")
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True,
                                           transform=transform)
    train_data_full = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True,
                                            transform=transform)
    train_data_size = len(train_data_full)
    client_data_size=int(train_data_size/(client_all-1))
    split_data_size = [client_data_size for i in range(client_all-1)]
    split_data_size.append(train_data_size-client_data_size*(client_all-1))
    train_data = torch.utils.data.random_split(train_data_full,split_data_size)
    fed_lr=0.1
    local_epoch=5
    cos_threshold=0.9900
    euc_threshold=30.0000
else:
    print("Parameter error.")
    exit()

dim=len(from_tensor_to_list(from_w_to_tensor(global_model)))
keypair = generate_keypairs(dim)
s=MVNP(client_all)
np.random.seed(s)
r=np.random.randint(100,size=dim)
ck=get_check_value(r)
r=torch.tensor(r).cuda()
test_acc=[]

for e in range(global_epoch):
    #server train
    torch.cuda.empty_cache()
    server_model=train(model_name,server_model,server_data,device,fed_lr,local_epoch)
    server_w=from_w_to_tensor(server_model)

    #client train
    client_w=[]
    client_proof=[]
    for i in tqdm(range(client_num)):
        torch.cuda.empty_cache()
        local_model = train(model_name, global_model, train_data[i], device,fed_lr,local_epoch)
        local_w = from_w_to_tensor(local_model)
        mask_w = to_mask(local_w, r)
        client_w = mask_w[None, :] if len(client_w) == 0 else torch.cat((client_w, mask_w[None, :]), 0)
        # generate proof
        proof = generate_proof(keypair.pk, dim,from_tensor_to_list(local_w) , from_tensor_to_list(server_w), from_tensor_to_list(mask_w),from_tensor_to_list(r),ck, cos_threshold,euc_threshold)
        client_proof.append(proof)

    #malicious clients
    if(mali_num>0):
        mali_model = train(model_name, global_model, train_data[-1], device,fed_lr,local_epoch)
        if attack_type=="add_noise":
            mali_w = add_noise(from_w_to_tensor(mali_model))
        elif attack_type=="sign_flipping":
            mali_w=sign_flipping(from_w_to_tensor(mali_model))
        elif attack_type=="bpfl_attack":
            if e==0:
                old_server_w=from_w_to_tensor(mali_model)
            dev_type = 'sign'
            mali_w = agr_tailored_attack(old_server_w, from_w_to_tensor(mali_model), mali_num, cos_threshold,euc_threshold,dev_type)
            old_server_w=old_server_w
        elif attack_type=="min_max":
            dev_type = 'sign'
            torch.cuda.empty_cache()
            mali_w = min_max(to_mask(client_w,0-r), from_w_to_tensor(mali_model), mali_num, dev_type)
        elif attack_type=="min_sum":
            dev_type = 'sign'
            torch.cuda.empty_cache()
            mali_w = min_sum(to_mask(client_w,0-r), from_w_to_tensor(mali_model), mali_num, dev_type)
        mask_mali_w=to_mask(mali_w,r)
        for m_n in range(mali_num):
            client_w = mask_mali_w[None, :] if len(client_w) == 0 else torch.cat((client_w,mask_mali_w[None, :]), 0)

    global_w=[]
    verified_num=0
    for g,f in zip(client_w,client_proof):
        verify_result = verify_proof(keypair.vk, f,from_tensor_to_list(server_w), from_tensor_to_list(g),ck, cos_threshold,euc_threshold)
        if verify_result:
            global_w = g.unsqueeze(0) if len(global_w) == 0 else torch.cat((global_w,g.unsqueeze(0)), 0)
            verified_num+=1
    print("verified num:{}".format(verified_num))
    if verified_num>0:
        global_w=mean(global_w)
    else:
        global_w=to_mask(from_w_to_tensor(global_model),r)
        server_model=copy.deepcopy(global_model)

    client_w=None
    global_w=to_mask(global_w,0-r)
    start_idx = 0
    for key in server_model.keys():
        param_ = global_w[start_idx:start_idx + len(server_model[key].view(-1))].reshape(server_model[key].data.shape)
        start_idx = start_idx + len(server_model[key].data.view(-1))
        param_ = param_.cuda()
        global_model[key] = param_
    #test model
    loss,acc=test_model(model_name,global_model,test_data,device)
    test_acc.append(acc)
    print("epoch:{} loss:{},accuracy:{}".format(e,loss,acc))
    #text
    with open("./output/{}_{}_{}_{}_{}.txt".format(data_name,client_all,mali_num,aggregation_type,attack_type),"a") as file:
        file.write("{} {:.4f}\n".format(e,acc))
    if e>0 and e%50==0:
        fed_lr*=0.5
        with open("./output/{}_{}_{}_{}_{}.txt".format(data_name,client_all,mali_num,aggregation_type,attack_type),"a") as file:
            file.write("new lr:{}\n".format(fed_lr))