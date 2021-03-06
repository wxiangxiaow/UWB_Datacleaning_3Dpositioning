import torch
import numpy as np

test_data = [
    ['T:055943544:RR:0:0:2940:2940:52:19764'],
    ['T:055943544:RR:0:1:4290:4290:52:19764'],
    ['T:055943545:RR:0:2:2840:2840:52:19764'],
    ['T:055943545:RR:0:3:4190:4190:52:19764'],

    ['T:060341974:RR:0:0:5240:5240:171:20907'],
    ['T:060341974:RR:0:1:5360:5360:171:20907'],
    ['T:060341974:RR:0:2:2040:2040:171:20907'],
    ['T:060341974:RR:0:3:2940:2940:171:20907'],

    ['T:060548859:RR:0:0:4800:4800:9:21513'],
    ['T:060548860:RR:0:1:2610:2610:9:21513'],
    ['T:060548860:RR:0:2:4750:4750:9:21513'],
    ['T:060548860:RR:0:3:2550:2550:9:21513'],

    ['T:060452358:RR:0:0:5010:5010:252:21244'],
    ['T:060452358:RR:0:1:4120:4120:252:21244'],
    ['T:060452359:RR:0:2:3810:3810:252:21244'],
    ['T:060452359:RR:0:3:2020:2020:252:21244'],

    ['T:053906604:RR:0:0:2840:2840:7:13831'],
    ['T:053906604:RR:0:1:4490:4490:7:13831'],
    ['T:053906604:RR:0:2:2860:2860:7:13831'],
    ['T:053906604:RR:0:3:4190:4190:7:13831'],

    ['T:053627102:RR:0:0:5010:5010:15:13071'],
    ['T:053627102:RR:0:1:5320:5320:15:13071'],
    ['T:053627102:RR:0:2:1990:1990:15:13071'],
    ['T:053627102:RR:0:3:2930:2930:15:13071'],

    ['T:053524347:RR:0:0:5050:5050:228:12772'],
    ['T:053524347:RR:0:1:3740:3740:228:12772'],
    ['T:053524347:RR:0:2:3710:3710:228:12772'],
    ['T:053524347:RR:0:3:2070:2070:228:12772'],

    ['T:055501490:RR:0:0:5050:5050:236:18412'],
    ['T:055501491:RR:0:1:4110:4110:236:18412'],
    ['T:055501491:RR:0:2:3710:3710:236:18412'],
    ['T:055501491:RR:0:3:2110:2110:236:18412'],

    ['T:054658330:RR:0:0:4840:4840:223:16095'],
    ['T:054658330:RR:0:1:2600:2600:223:16095'],
    ['T:054658330:RR:0:2:4960:4960:223:16095'],
    ['T:054658330:RR:0:3:2700:2700:223:16095'],

    ['T:055551320:RR:0:0:2740:2740:219:18651'],
    ['T:055551320:RR:0:1:2720:2720:219:18651'],
    ['T:055551320:RR:0:2:4670:4670:219:18651'],
    ['T:055551321:RR:0:3:4790:4790:219:18651']
]

def preprocess():
    lines = []
    index = []
    for i in range(len(test_data)):
        indes = str(test_data[i]).split(':')
        index.append(int(indes[5]))
        if int(indes[4]) == int(3):
            lines.append(index)
            index = []
    
    #when training, train_data_max = 7070
    train_data_max = 7070
    lines_ = np.array(lines)/train_data_max
    lines = lines_.tolist()
    return lines

def main():
    device = torch.device('cuda:0')
    opti = 'mean_18_Adam_'
    checkpoint = torch.load('./checkpoint_mean/'+opti+'checkpoint.pth.tar', map_location='cuda:0')
    model = checkpoint['model']
    data = preprocess()
    k = 1
    print('Now optimizer is '+opti)
    for index in data:
        data_tensor = torch.Tensor(index)
        input = data_tensor.to(device)
        probs = torch.exp(model.forward(input))

        
        probs = probs.cpu().detach().numpy().tolist()
        if probs[0] > probs[1]:
            info = 'error'
            pro = float(probs[0])/(float(probs[0])+float((probs[1])))*100.
        else:
            info = 'correct'
            pro = float(probs[1])/(float(probs[0])+float((probs[1])))*100.
        
        print('data'+str(k)+' attribute : '+info + ', prob = '+ str(pro))
        k = k+1
    return 0

if __name__ == '__main__':
    main()