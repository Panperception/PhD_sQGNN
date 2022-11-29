import torch.utils.data as Data
import numpy as np
import torch
import pennylane as qml
from torch import nn
from torch_geometric.datasets import TUDataset
from torch.utils.data import SubsetRandomSampler

batch_size = 256
total_epoch = 200
lr = 0.07
classes_num = 2
testing_split = .1
shuffle_dataset = True
random_seed = 3

np_data = np.loadtxt('./DCH_E/MUTAG_DHC_E.csv',delimiter=',')
data = torch.Tensor(np_data)

dataset = TUDataset(root='dataset/MUTAG', name='MUTAG')
Y = np.array([])
for i in range(len(dataset)):
    Y = np.append(Y,np.array(dataset[i].y)[0])

print(len(Y))
print(data.shape)

all_dataset = Data.TensorDataset(data,torch.LongTensor(Y))

dataset_size = len(all_dataset)
indices = list(range(dataset_size))
split = int(np.floor(testing_split * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, test_indices = indices[split:], indices[:split]

test_sampler = SubsetRandomSampler(test_indices)

test_loader = Data.DataLoader(dataset=all_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                              pin_memory=True, sampler=test_sampler)

dev = qml.device("default.qubit", wires=1)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

@qml.qnode(dev)
def circuit(inputs, weights):
    qml.Rot(weights[0] + inputs[0] * weights[1],weights[2] + inputs[1] * weights[3],
            weights[4] + inputs[2] * weights[5],wires=0)
    qml.Rot(weights[6] + inputs[3] * weights[7],weights[8] + 0 * weights[9],
            weights[10] + 0 * weights[11],wires=0)


    return [qml.expval(qml.Hermitian([[1,0],[0,0]], wires=[0]))]

class QGN(nn.Module):
    def __init__(self):
        super().__init__()
        phi = 12

        weight_shapes = {"weights": phi}
        self.qlayer_1 = qml.qnn.TorchLayer(circuit, weight_shapes)
        #self.post_net = nn.Linear(1, 2)

    def forward(self, input_features):
        out = self.qlayer_1(input_features)
        out = torch.FloatTensor(out)
        out = out.to(device)
        #out = self.post_net(out)

        return out

class FLoss(torch.nn.Module):
    def __init__(self):
        super(FLoss, self).__init__()

    def forward(self, output, target):
        x_t = torch.randn([len(output),1])
        for i in range(len(output)):
            if target[i] == 0:
                x_t[i][0] = output[i][0]
            else:
                x_t[i][0] = output[i][1]
        f_loss = (1 - x_t)**2
        return torch.mean(f_loss)



test_model = QGN()
test_model.load_state_dict(torch.load('params_MUTAG_1.pkl'))
test_model = test_model.to(device)
test_model.eval()

test_criterion = FLoss()

print('Start testing')
print('------------------------------------------------------------------------------')

for epoch in range(1):
    with torch.no_grad():

        running_test_loss = 0
        test_total = 0
        test_correct = 0

        for i, test_data in enumerate(test_loader):
            test_DCH, test_labels = test_data

            test_DCH = test_DCH.to(device)
            test_labels = test_labels.to(device)

            test_outputs = test_model(test_DCH)
            test_outputs_2 = torch.cat([test_outputs, 1 - test_outputs], 1)


            test_loss = test_criterion(test_outputs_2, test_labels)
            test_per = test_loss.item()
            running_test_loss += test_loss.item()

            test_predicted = test_outputs_2.argmax(dim=1)

            test_total += test_labels.size(0)
            test_correct += test_predicted.eq(test_labels).sum().item()

            test_loss = running_test_loss / len(test_loader)
            test_accu = 100. * test_correct / test_total

        print('Loss: {:.4f}, Acc: {:.4f}, Loss_item: {:.4f}'.format(test_loss, test_accu, test_per))

