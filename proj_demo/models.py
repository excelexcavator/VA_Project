import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import sys

class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, cell_num=1):
        super(FeatAggregate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.rnn = nn.LSTM(input_size, hidden_size, cell_num, batch_first=True)

    def forward(self, feats):
        h0 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)
        c0 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)

        if feats.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        # aggregated feature
        feat, _ = self.rnn(feats, (h0, c0))
        return feat[:,-1,:]

# Visual-audio multimodal metric learning: LSTM*2+FC*2
class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.VFeatPool = FeatAggregate(1024, 128)
        self.AFeatPool = FeatAggregate(128, 128)
        self.bn1 = nn.BatchNorm1d(128, eps=1e-5)
        self.bn2 = nn.BatchNorm1d(128, eps=1e-5)
        self.fc_v = nn.Linear(128, 32)
        self.fc_a = nn.Linear(128, 32)
        self.fc = nn.Linear(32, 1)
        self.init_params()
        self.dropnum = 0.3

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight)

    def forward(self, vfeat, afeat):
        vfeat = self.VFeatPool(vfeat)
        vfeat = self.bn1(vfeat)
        vfeat = F.dropout(vfeat, self.dropnum)
        vfeat = F.relu(vfeat)
        afeat = self.AFeatPool(afeat)
        afeat = self.bn2(afeat)
        afeat = F.dropout(afeat, self.dropnum)
        afeat = F.relu(afeat)      
        vfeat = self.fc_v(vfeat)
        vfeat = F.relu(vfeat)
        vfeat = self.fc(vfeat)
        afeat = self.fc_a(afeat)
        afeat = F.relu(afeat)
        afeat = self.fc(afeat)
        return F.pairwise_distance(vfeat, afeat)

class VA_Linear(nn.Module):
    def __init__(self):
        super(VA_Linear,self).__init__()
        self.fc_v = nn.Linear(1024, 128)
        self.fc_a = nn.Linear(128, 128)
        self.fc_v2= nn.Linear(128, 128)
        self.fc1 = nn.Linear(128, 96)
        self.fc2 = nn.Linear(96, 64)
        # self.bn_v = nn.BatchNorm1d(96, eps=1e-5)
        # self.bn_a = nn.BatchNorm1d(96, eps=1e-5)
        self.ap = nn.AvgPool1d(120)
        self.dropnum = 0
        self.lk = nn.LeakyReLU(0.05)
        self.init_params()
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        #vfeat = vfeat.transpose(2, 1).contiguous()
        vfeat = self.fc_v(vfeat)
        vfeat1 = vfeat
        vfeat = self.lk(vfeat)
        vfeat = F.dropout(vfeat, self.dropnum)
        vfeat = self.ap(vfeat.transpose(2, 1).contiguous())
        vfeat = vfeat.view(vfeat.size(0), -1)
        vfeat1 = vfeat
        vfeat = self.fc_v2(vfeat)
        vfeat = self.lk(vfeat)
        vfeat = vfeat + vfeat1
        vfeat = self.fc1(vfeat)
        vfeat = F.dropout(vfeat, self.dropnum)
        # vfeat = self.bn_v(vfeat)
        vfeat = self.fc2(vfeat)

        #afeat = afeat.transpose(2,1).contiguous()
        afeat = self.fc_a(afeat)
        afeat1 = afeat
        afeat = self.lk(afeat)
        afeat = F.dropout(afeat, self.dropnum)
        afeat = self.ap(afeat.transpose(2, 1).contiguous())
        afeat = afeat.view(afeat.size(0), -1)
        afeat1 = afeat
        afeat = self.fc_a(afeat)
        afeat = self.lk(afeat)
        afeat = afeat + afeat1
        afeat = self.fc1(afeat)
        afeat = F.dropout(afeat, self.dropnum)
        # afeat = self.bn_a(afeat)
        afeat = self.fc2(afeat)

        return vfeat, afeat

# Visual-audio multimodal metric learning:
# MaxPool + FC ---> Conv1d + AvgPool + FC
class VAMetric2(nn.Module):
    def __init__(self, framenum=120):
        super(VAMetric2, self).__init__()
#       self.mp = nn.MaxPool1d(framenum)
        self.ap = nn.AvgPool1d(framenum)
        self.convv = nn.Conv1d(1024, 128, 1, 1, 0)
        self.conva = nn.Conv1d(128 , 128, 1, 1, 0)
        self.fc0 = nn.Linear(128, 96)
        self.fc1 = nn.Linear(96 , 64)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        # aggregate the visual features
        vfeat = vfeat.transpose(2, 1)
        vfeat = self.convv(vfeat)
        vfeat = F.tanh(vfeat)
        vfeat = self.ap(vfeat)
#       vfeat = self.mp(vfeat)
        vfeat = vfeat.view(-1, 128)
        vfeat = F.tanh(self.fc0(vfeat))
        vfeat = self.fc1(vfeat)

        # aggregate the auditory features
        afeat = afeat.transpose(2, 1)
        afeat = self.conva(afeat)
        afeat = F.tanh(afeat)
        afeat = self.ap(afeat)
#       afeat = self.mp(afeat)
        afeat = afeat.view(-1, 128)
        afeat = F.tanh(self.fc0(afeat))
        afeat = self.fc1(afeat)
        return F.pairwise_distance(vfeat, afeat)
	
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        dist = dist.view(-1)
        loss = torch.mean((1-label) * torch.pow(dist, 2) +
                (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss
	"""
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        
    def newloss(self, vfeat, afeat):
        batchsize = vfeat.size(0)
        margin = int(sys.argv[1])
        p = 2
        eps = 1e-6
        swap = False
        sample_number = 5
        assert(sample_number < batchsize)
        for k in range(batchsize):
            vfeat0 = vfeat[k, :]
            afeat0 = afeat[k, :]
            vfeat0 = vfeat0.view(-1, vfeat0.size(0))
            afeat0 = afeat0.view(-1, afeat0.size(0))
            for m in range(sample_number):
                if m == 0:
                    anchorv0 = vfeat0
                    anchora0 = afeat0
                else:
                    anchorv0 = torch.cat((anchorv0, vfeat0), 0)
                    anchora0 = torch.cat((anchora0, afeat0), 0)
                if m == k:
                    vfeat1 = vfeat[(m+1)%batchsize, :]
                    vfeat1 = vfeat1.view(-1, vfeat1.size(0))
                    afeat1 = afeat[(m+1)%batchsize, :]
                    afeat1 = afeat1.view(-1, afeat1.size(0))
                    if k == 0:
                        negative1 = vfeat1
                        negative2 = afeat1
                    else:
                        negative1 = torch.cat((negative1, vfeat1), 0)
                        negative2 = torch.cat((negative2, afeat1), 0)
                else:
                    vfeat2 = vfeat[(m+k+1)%batchsize, :]
                    vfeat2 = vfeat2.view(-1, vfeat2.size(0))
                    negative1 = torch.cat((negative1,vfeat2),0)
                    afeat2 = afeat[(m+k+1)%batchsize, :]
                    afeat2 = afeat2.view(-1, afeat2.size(0))
                    negative2 = torch.cat((negative2,afeat2),0)
            if k == 0:
                anchor1 = anchorv0
                anchor2 = anchora0
            else:
                anchor1 = torch.cat((anchor1, anchorv0), 0)
                anchor2 = torch.cat((anchor2, anchora0), 0)

        # print(anchor1, negative1, negative2, anchor2)
        d_p = F.pairwise_distance(anchor1, anchor2, p, eps)
        d_n1 = F.pairwise_distance(anchor1, negative2, p, eps)
        d_n2 = F.pairwise_distance(anchor2, negative1, p, eps)

        for k in range(int(d_p.size(0)/sample_number)):
            d_p_max_temp = torch.topk(d_p[k*sample_number:(k+1)*sample_number, :], 1, dim = 0)[0]
            d_n1_min_temp = torch.topk(d_n1[k*sample_number:(k+1)*sample_number, :], 1, dim = 0,largest = False)[0]
            d_n2_min_temp = torch.topk(d_n2[k*sample_number:(k+1)*sample_number, :], 1, dim = 0,largest = False)[0]
            if k == 0:
                d_p_max = d_p_max_temp
                d_n1_min = d_n1_min_temp
                d_n2_min = d_n2_min_temp
            else:
                d_p_max = torch.cat((d_p_max, d_p_max_temp),0)
                d_n1_min = torch.cat((d_n1_min, d_n1_min_temp), 0)
                d_n2_min = torch.cat((d_n2_min, d_n2_min_temp), 0)
        if swap:
            d_s = F.pairwise_distance(positive, negative, p, eps)
            d_n = torch.min(d_n, d_s)
        #dist_hinge = torch.clamp(margin + 2*d_p - d_n1 - d_n2, min=0.0)
        dist_hinge = torch.clamp(margin + 2*d_p_max - d_n1_min - d_n2_min, min=0.0)
        #print(dist_hinge)
        loss = torch.mean(dist_hinge)
        return loss
		
    def forward(self, vfeat, afeat):
        loss = self.newloss(vfeat, afeat)
        return loss
