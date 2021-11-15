import enum
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import pickle
import numpy as np
import time
from model import CBOW_Model, SkipGram_Model

def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate, 
    so thatlearning rate after the last epoch is 0.
    """
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    return lr_scheduler

def split_data_modify(pair):
    dataX = []
    dataY = []
    for data in pair:
        dataX.append(data[0])
        dataY.append(data[1])
    
    data_X = torch.from_numpy(np.array(dataX))
    data_Y = torch.from_numpy(np.array(dataY))

    return data_X, data_Y

def get_dataloader(ratio, data_X, data_Y, batch_size):
    
    train_size = int(ratio * len(data_X))
    test_size = len(data_X) - train_size
    dataset = torch.utils.data.TensorDataset(data_X, data_Y)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=8)

    print("trainging data: ", len(train_data))
    print("testing data: ", len(test_data))
    
    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    train_running_correct = 0

    for i, data in enumerate(train_loader):
#         print(data[0])
#         print(data[0].shape)
#         print(type(data[0]))
        inputs, labels = data[0].to(device), data[1].to(device)
#         Normalize
#         inputs_m, inputs_s = inputs.mean(), inputs.std()
#         inputs = (inputs - inputs_m) / inputs_s

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()*labels.size(0)

        # _, prediction = torch.max(outputs,1)
        # train_running_correct += (prediction == labels).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_loader.dataset)
    # train_acc = 100. * train_running_correct/len(train_loader.dataset)
    return train_loss

def testing(model, test_loader, criterion):
    model.eval()
    val_running_correct = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs= model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()*labels.size(0)
            # _, prediction = torch.max(outputs,1)
            # val_running_correct += (prediction == labels).sum().item()
    
    val_loss = running_loss/len(test_loader.dataset)
    # acc = 100. * val_running_correct/len(test_loader.dataset)
    
    return val_loss
with open('./file/word2idx.pickle','rb') as file:
    word2idx = pickle.load(file)
with open('./file/pairword.pickle','rb') as file:
    pair = pickle.load(file)

print(len(word2idx))

LR = 1e-4
BATCH_SIZE = 512
EPOCH = 200
ratio = 0.8
MODEL_NAME = 'Model_SkipGram'

skip_gram = SkipGram_Model(vocab_size = len(word2idx), embedding_dim = 600).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(skip_gram.parameters(), lr=LR)
# lr_scheduler = get_lr_scheduler(optimizer, epoch = EPOCH, verbose=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_X, data_Y = split_data_modify(pair)
train_loader, test_loader = get_dataloader(ratio, data_X, data_Y, BATCH_SIZE)

recoder = []
epochs, Train_losses, Train_accs, Test_losses, Test_accs = [], [], [], [], []
Best_loss = 100

for epoch in range(EPOCH):
    epoch_start_time = time.time()
    Train_loss = train(skip_gram, train_loader, criterion, optimizer)
    Test_loss = testing(skip_gram, test_loader, criterion)

    epoch_secs = int(time.time() - epoch_start_time)
    recoder.append([epoch, Train_loss, Test_loss])
    epochs.append(epoch)
    Train_losses.append(Train_loss)
    Test_losses.append(Test_loss)
    print(f'Epoch:{epoch}| Train loss :{Train_loss:.4f}| Test loss:{Test_loss:.4f}| Time:{epoch_secs}s')

    """Save Model"""
    if Test_loss < Best_loss:
        Best_loss = Test_loss
        torch.save(skip_gram.state_dict() , './model/SkipGram_lemma' + '_' + str(epoch) + '.pt')
    
    """ Dump recorder """
    with open('SkipGram_lemma.csv', 'a') as f:
        f.write(f'Epoch:{epoch}| Train loss :{Train_loss:.4f}| Test loss:{Test_loss:.4f}|Time:{epoch_secs}s\n')
        
    with open('./SkipGram_lemma.pickle', 'wb') as file:
        pickle.dump(epochs, file)
        pickle.dump(Train_losses, file)
        pickle.dump(Test_losses, file)