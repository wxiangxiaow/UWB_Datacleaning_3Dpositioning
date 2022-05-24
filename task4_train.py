from collections import OrderedDict
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class sample_nomolization(object):
    def __init__(self):
        self.data_all = []
        self.corr_data = []
        self.wrong_data = []

        self.data_max = 0
        correct_data_dir = './save_txt_mean_18/normal/'
        wrong_data_dir = './save_txt_mean_18/error/'
        len_corr = len(os.listdir(correct_data_dir))
        len_wro = len(os.listdir(wrong_data_dir))
        #collect right data
        for i in range(len_corr):
            correct_data_path = correct_data_dir + str(i+1) + '.txt'
            with open(correct_data_path, 'r') as file:
                for index in file.readlines():
                    index = index.strip('\n').strip('[').strip(']').split(',')[1:]
                    indexs = []
                    for j in range(len(index)):
                        m = int(index[j].strip(' '))
                        indexs.append(m)
                        if m > self.data_max:
                            self.data_max = m
                    self.corr_data.append(indexs)

            file.close()
        #collect wrong data
        for i in range(len_wro):
            wrong_data_path = wrong_data_dir + str(i+1) + '.txt'
            lines = []
            with open(wrong_data_path, 'r') as file:
                for index in file.readlines():
                    index = index.strip('\n').strip('[').strip(']').split(',')[1:]
                    indexs = []
                    for j in range(len(index)):
                        m = int(index[j].strip(' '))
                        indexs.append(m)
                        if m > self.data_max:
                            self.data_max = m
                    self.wrong_data.append(indexs)
            file.close()
        
        self.corr_data_ = np.array(self.corr_data)/self.data_max 
        self.wrong_data_ = np.array(self.wrong_data)/self.data_max 
        
        #加上label
        self.corr_data = (self.corr_data_).tolist()
        self.wrong_data = (self.wrong_data_).tolist()
        for i in range(len(self.corr_data)):
            tensor = [float(self.corr_data[i][0]), float(self.corr_data[i][1]), float(self.corr_data[i][2]), float(self.corr_data[i][3]), int(1)]
            self.data_all.append(tensor) 
        for i in range(len(self.wrong_data)):
            self.data_all.append([float(self.wrong_data[i][0]), float(self.wrong_data[i][1]), float(self.wrong_data[i][2]), float(self.wrong_data[i][3]), int(0)])   
        
        print('correct data longth = ' + str(len(self.corr_data)))    #6072
        print('wrong data longth = ' + str(len(self.wrong_data)))     #19642
        print('max data = ' + str(self.data_max))

    def __getitem__(self, index):
        data = np.array(self.data_all[index][:-1])
        label = self.data_all[index][-1]
        data = torch.Tensor(data)

        return data, label
    
    def __len__(self):
        return len(self.data_all)
    

def train(model, train_loader, valid_loader, criterion, optimizer, device, epoch, scheduler, train_on_gpu):
    if train_on_gpu:
        model.to(device)
    train_loss = 0.0
    valid_loss = 0.0
    if scheduler != None:
        scheduler.step()
    ########train#########
    model.train()
    for data, target in train_loader:
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    ########valid#########
    model.eval()
    num_correct, num_data = 0, 0
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)
        ############# calculate the accurecy
        _, pred = torch.max(output, 1) 
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu \
                                else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += sum(correct)
        num_data += correct.shape[0]
    ###################################
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    accuracy = (100 * num_correct / num_data)
    print('Epoch: {} \n-----------------\n \tTraining Loss: {:.6f} \t Validation Loss: {:.6f} \t accuracy : {:.4f}% '.format(epoch, train_loss, valid_loss,accuracy))
    model.to(device)
    return train_loss, valid_loss, accuracy

def save_checkpoint(epoch, epoch_since_improvement, model, optimizer, loss, best_loss, is_best):
    state = {
                'epoch' : epoch,
                'epoch_since_improvement' : epoch_since_improvement,
                'loss' : loss,
                'best_loss' : best_loss,
                'model' : model,
                'optimizer' : optimizer
            }
    path = '/home/user/mathmodel/checkpoint_mean/'
    if not os.path.exists(path):
        os.makedirs(path)
    filename = 'mean_18_adagrad_checkpoint.pth.tar'
    torch.save(state, path + filename)
    if is_best:
        torch.save(state, path + 'BEST_' + filename)


def test_model(model, test_loader, device, criterion,classes):
    train_on_gpu = True
    test_loss = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    model.to(device)
    model.eval()
    for data,target in test_loader:
        if train_on_gpu:
            #data,target = data.cuda(),target.cuda()
            data,target = data.to(device),target.to(device)
        output = model(data)
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        
        _, pred = torch.max(output, 1) 
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu \
                                else np.squeeze(correct_tensor.cpu().numpy())
        for i in range(data.shape[0]):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}'.format(test_loss))
    print('Test Accuracy (Overall): %2d%% (%2d/%2d) \n ----------------------' % (100. * np.sum(class_correct) / np.sum(class_total),np.sum(class_correct), np.sum(class_total)))
    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of %s : %d%% (%2d/%2d)' % (classes[i], 100 * class_correct[i] / class_total[i],np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %s: N/A (no training examples)' % (classes[str(i+1)]))


def main():

    Epoch = 2000
    device = torch.device('cuda:0')
    batch_size = 128
    epochs_since_improvement = 0
    best_loss = 1
    start_epoch = 0

    #for draw
    vali_loss = []
    trai_loss = []
    accuray = []
    epoch_ = []

    #data
    train_data = sample_nomolization()
    #distance=15 [21693, 6197, 9296] max=7070
    #disrance=18
    train_data, valid_data, test_data = torch.utils.data.random_split(train_data, [16486, 4708, 7062])
    print('train_data size: ', len(train_data))
    print('valid_data_size: ', len(valid_data))
    print('test_data_size: ', len(test_data))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=0, shuffle=True)


    #model
    model = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(4, 128)),
        ('r1',  nn.ReLU()),
        ('fc2', nn.Linear(128, 512)),
        ('r2',  nn.ReLU()),
        ('fc3', nn.Linear(512, 512)),
        ('r3',  nn.ReLU()),
        ('d1',  nn.Dropout(0.3)),
        ('fc4', nn.Linear(512, 64)),
        ('r4',  nn.ReLU()),
        ('fc5', nn.Linear(64, 8)),
        ('r5',  nn.ReLU()),
        ('fc6', nn.Linear(8, 2))
    ]))

    if torch.cuda.is_available():
        train_on_gpu = True
        print('train on GPU :)')
    else:
        print('train on CPU :(')
        train_on_gpu = False
    #train

    checkpoint = None
    #checkpoint = torch.load('./checkpoint/checkpoint.pth.tar', map_location='cuda:0')
    if checkpoint is None:
        #optimizer = optim.Adam(model.parameters())
        optimizer = optim.Adagrad(model.parameters())
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
    else:
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epoch_since_improvement']
        best_loss = checkpoint['best_loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(start_epoch, Epoch):
        tra_loss, val_loss, acc = train(
                                        model,
                                        train_loader,
                                        valid_loader,
                                        criterion = criterion,
                                        optimizer = optimizer,
                                        device = device,
                                        epoch = epoch,
                                        scheduler = None,
                                        train_on_gpu = train_on_gpu
        )

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        vali_loss.append(val_loss)
        trai_loss.append(tra_loss)
        accuray.append(acc)
        epoch_.append(epoch)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    
        else:
            epochs_since_improvement = 0
    
        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_loss, best_loss, is_best) 
    
    
    class_to_idx = {'error': 0, 'Normal': 1}
    cat_to_name = {class_to_idx[i]: i for i in list(class_to_idx.keys())}
    test_model(model, test_loader, device, criterion, cat_to_name)

    plt.plot(epoch_, vali_loss, label='Valid Loss')
    plt.plot(epoch_, trai_loss, label='Train Loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig('Loss.png')
    plt.show()
    plt.close()

    plt.plot(epoch_, accuray, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.savefig('accuracy.png')
    plt.show()



if __name__ == '__main__':
    main()


