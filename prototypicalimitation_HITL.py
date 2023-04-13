
import numpy as np
import time
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from ExTrajectory_nooption import ExpertTraj
import random

from Feedback import OptionFeedback 
import scipy.stats as S 


# data preprocessing
d1 = pd.read_csv('trajectoryfile.csv')
d1_day = pd.to_datetime(d1['BinTimeStamp']).dt.day
d1_hour = pd.to_datetime(d1['BinTimeStamp']).dt.hour
d1['hour'] = (d1_day-13)*24 + d1_hour

list_2 = [ 'hour1','nodeCoreUsage', 'nodeMemoryUsage', 'LifeTimebyHour',  'CpuRequest', 'CpuAllocate',  'VMMemory']
list_3 = [ 'hour1','nodeCoreUsage', 'nodeMemoryUsage',  'LifeTimebyHour',  'CpuRequest', 'CpuAllocate',  'VMMemory','cpuUsage_label','cpuUsage']

d1 = d1.fillna(method="ffill")
d1 = d1.fillna(method="bfill")
d1 = d1.fillna(0)
d1 = d1.dropna()
d1['hour1'] = d1['hour'].values%24
for i in list_2:
    d1[i] = (d1[i]-min(d1[i]))/(max(d1[i])-min(d1[i])+0.0001)
d1['cpuUsage'] = d1['cpuUsage']/100
list_2.append('cpuUsage')
d1['cpuUsage_label'] = d1['cpuUsage'].values
ServiceId = list(d1['ServiceId'].drop_duplicates())
for i in ServiceId:
    value = d1['cpuUsage'][d1['ServiceId'] == i].values
    value_1 = value.copy()
    value_1[0]=0
    value_1[1:]=value[:-1]
    d1['cpuUsage_label'][d1['cpuUsage_label']==i]=value_1
otherFeatures = d1[['ServiceId','hour','BinTimeStamp']]
d1 = d1[list_3]


random_seed = 1024
if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

class OptionEMB(nn.Module):
    def __init__(self, emb_dim, num_option):
        super(OptionEMB, self).__init__()
        self.emb_dim = emb_dim
        self.num_option = num_option
        self.o = []
        self.o_all = []
        # \uniform_(-1, 1), requires_grad = True) for i in range(num_option)]
        for i in range(self.num_option):
            a_t = np.zeros((1,6))
            a_t[0,i] = 1
            self.o_all.append(torch.autograd.Variable(torch.FloatTensor(a_t), requires_grad = True))

    def get_emb(self):
        return (self.o_all)
    #----------------- hitl---------------
    def addOption(self, emb = None):
        print("Here")
        a_new = np.zeros((1,6))
        if emb is None:
            a_new[0,self.num_option] = 1
        else:
            a_new = [emb]
        self.o_all.append(torch.autograd.Variable(torch.FloatTensor(a_new), requires_grad = True))
        self.num_option +=1
        print(self.o_all)

    def splitOption(self,optionid,cluster,pointEntropy, policy, trainState):
        
        points = cluster[optionid]
        EntList = []
        for p in points:
            e = pointEntropy[p]
            EntList.append(e)
        topentK=np.argpartition(EntList, -4)[-4:]
        splitPoints = np.array(points)[topentK]
        state = torch.FloatTensor(np.array(trainState)[splitPoints])
        # print('state',state)
        split_o_state = policy.forward_emb(state)
        print(split_o_state.data.numpy())
        embMean = np.mean(split_o_state.data.numpy(), axis = 0)
        print(embMean)
        self.addOption(emb=embMean)
        return 

    def mergeOptions(self,ops):
        to_merge = []
        for o in ops: 
            to_merge.append(self.o_all[o].data.numpy())
        merge = np.mean(np.array(to_merge),axis=0)
        print("MERGED",merge)
        temp_o_all = []
        for i,o in enumerate(self.o_all):
            if i not in ops:
                temp_o_all.append(o)
        o_new = torch.autograd.Variable(torch.FloatTensor(merge), requires_grad = True)
        temp_o_all.append(o_new)
        self.o_all = temp_o_all
        self.num_option = len(self.o_all)
        print(self.o_all)
    #---------------------hitl ---------------


##---------------------- For HITL -----------
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
def entropy(input):
    input = 1/np.absolute(input)
    base = len(input)
    #print(input)

    if base <= 1:
        return 0
    
    #norm = np.sum(np.exp(input))
    #probs = np.exp(input) / norm
    probs = (input - np.min(input))/np.ptp(input) 
    probs[np.where(probs==0.0)]=0.00001
    #probs = input
    #print(probs)
    #print((probs * np.log(probs))) 

    if base <= 1:
        return 0
    return S.entropy(probs,base=base)

def simOpt(o_state, option_embs, pretrain):
    #reverse distances for given option
    optionEnt=[]
    optionDist=[]
    pointEnt = []
    cluster = {}
    o_state = o_state.data.numpy()
    for i in range(o_state.shape[0]):
        distance4givenPoint = [np.linalg.norm(o_state[i] - option_embs[j].data.numpy()) for j in range(len(option_embs))]
        distance4givenPoint = np.array(distance4givenPoint)
        #distance4givenPointNorm = NormalizeData(distance4givenPoint)
        ent4point = entropy(distance4givenPoint)
        pointEnt.append(ent4point)


    for j in range(len(option_embs)):
        #optionEntDist[j] = {}
        distance4givenOption = np.absolute([np.linalg.norm(o_state[i] - option_embs[j].data.numpy()) for i in range(o_state.shape[0])])
        invd = np.array(distance4givenOption)
        indk = np.argpartition(invd, 10)[:10]
        cluster[j] = indk
        topk = np.array(distance4givenOption)[indk]
        entO = entropy(topk)
        optionEnt.append(entO)
        avgDist = np.mean(topk)
        optionDist.append(avgDist)
        # ind = np.argpartition(-np.absolute(distances4givenOption), -30)[-30:]
        # topk = np.array(distances4givenOption)[ind]
        # entO = entropy(topk)
        #print("For option: ",j," = ", entO)
    return optionEnt, optionDist, pointEnt, cluster

def simCrossOpt(o_state, option_embs, pretrain): 
    D = []
    #print(D)
    for j in range(len(option_embs)):
        d = []
        for k in range(len(option_embs)):
            #print("J,K === ", j,k )
            if j is not k:
                distanceBetweenOption = np.absolute([np.linalg.norm(option_embs[j].data.numpy() - option_embs[k].data.numpy())])
                d.append(distanceBetweenOption[0])
                #print(distanceBetweenOption[0])
            else:
                d.append(0.0)
                #print(0)
        D.append(d)
    return D

def weightedMSE(x,y,signgrad):
    sqterms = (x - y)**2
    penalty = torch.Tensor(np.exp(np.array(signgrad)))
    sqterms = torch.mul(sqterms,penalty)
    loss = torch.mean(sqterms)
    return loss

##-------------------------------------------------------------------------

def sim(o_state, option_embs, pretrain):


    chosed_options = []
    chosed_options_tensor = []
    chosed_options_state_list = [[] for i in range(len(option_embs))]
    op_ids = []
    o_state = o_state.data.numpy()
    op_2_ids = []

    list_all = []



    if pretrain:


        for i in range(o_state.shape[0]):
            # print(i)
            sim_emb = [np.linalg.norm(o_state[i] - option_embs[j].data.numpy()) for j in range(len(option_embs))]

            op = random.randint(0, len(option_embs)-1)
            op_1 = np.argmin(sim_emb)
            chosed_options.append(option_embs[op].data.numpy())
            chosed_options_tensor.append(option_embs[op][0])
            chosed_options_state_list[op].append(o_state[i])
            op_ids.append(op)
            op_2_ids.append(op_1)
    else:
        for i in range(o_state.shape[0]):
            sim_emb = [np.linalg.norm(o_state[i] - option_embs[j].data.numpy()) for j in range(len(option_embs))]
            op = np.argmin(sim_emb)
            chosed_options.append(option_embs[op].data.numpy())
            chosed_options_tensor.append(option_embs[op][0])
            chosed_options_state_list[op].append(o_state[i])
            op_ids.append(op)


    chosed_options = np.array(chosed_options)
    chosed_options = np.squeeze(chosed_options, 1)

    chosed_options_tensor = torch.stack(chosed_options_tensor)
    op_ids = np.array(op_ids)
    #print(op_ids)
    #print(op_2_ids)

    return np.array(chosed_options), chosed_options_state_list, chosed_options_tensor, op_ids, op_2_ids


class Policy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_emb, emb_dim, num_option):
        super(Policy, self).__init__()

        self.hidden_dim = hidden_dim
        self.l1 = nn.Linear(state_dim, hidden_dim)
        torch.nn.init.xavier_uniform(self.l1.weight)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_uniform(self.l2.weight)
        self.l3 = nn.Linear(hidden_dim, emb_dim)
        torch.nn.init.xavier_uniform(self.l3.weight)
        self.l4 = nn.Linear(num_option, action_emb)
        torch.nn.init.xavier_uniform(self.l4.weight)
        self.tahn = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmiod = nn.Sigmoid()
        self.softmax = nn.Softmax()


    def forward(self, state, option):
        output = self.relu(self.l1(state))
        #print(output.shape)
        output = self.relu(self.l2(output))
        #print(output.shape)
        output = self.tahn( self.l3(output))
        #print(output.shape,option.shape)
        output = torch.matmul(output, option)
        #print(output.shape)
        output = self.l4(output)

        return output

    def forward_emb(self, state):
        output = self.relu(self.l1(state))
        # output = self.relu(self.l2(output))
        output = self.tahn( self.l3(output))

        return output


state_dim = len(list_3)-1
Rep_hidden_dim = 128
action_dim = 1
emb_dim = 6
num_option = 6
betas = (0.5, 0.999)
use_hitl =  True 

adhocPretrainCount = 0 

policy = Policy(state_dim, Rep_hidden_dim, action_dim, emb_dim, num_option)
optim_policy = torch.optim.Adam(policy.parameters(), lr=0.01)
optim_policy_emb = torch.optim.Adam(policy.parameters(), lr=0.00001)

option_net = OptionEMB(emb_dim, num_option)
optim_option = torch.optim.Adam(option_net.o_all, lr=0.0001, betas=betas)
loss_emb = nn.MSELoss()
expert = ExpertTraj(d1.values, 0.8)
pretrain = False
opCountPos = [0]*num_option
opCountNeg = [0]*num_option

feedback = None 
if use_hitl == True: 
    feedback = OptionFeedback(num_options=num_option)


for t in range(300):
   
    train_state, train_action= expert.sample(128)
    state = torch.FloatTensor(train_state)
    # print('state',state)
    action = torch.FloatTensor(train_action)
    o_state = policy.forward_emb(state)

    if t > 60:
        pretrain = False
    else:
        pretrain = True

    #-------------hitl -----------
    if adhocPretrainCount>0 and adhocPretrainCount < 30:
        pretrain = True
        adhocPretrainCount += 1
    elif adhocPretrainCount >= 30:
        pretrain = False
        adhocPretrainCount = 0
    #--------------hitl------------

    ops1, ops_state_lists, ops_tensor, op_ids, op_2_ids = sim(o_state, option_net.get_emb(), pretrain)
    #print(op_ids)
    merge = []
    splits = []
    

    if t > 60:

        #--------------for HITL ------
        if use_hitl==True:
            oent,odist, pointEnt, optionCLuster = simOpt(o_state, option_net.get_emb(),pretrain)
            D = np.array(simCrossOpt(o_state, option_net.get_emb(),pretrain))
            #print(pointEnt)
            #print(D, np.unravel_index(np.argmax(D),D.shape))
            advise, splits, interadvice, merge = feedback.getOptionFeedback(oent,odist,all=False,step=t,D=D)
            #if frequent_unsafe_option != -1:
                #advise[frequent_unsafe_option] +=-1
            #ops1, ops_state_lists, ops_tensor, op_ids, op_2_ids = sim(o_state, option_net.get_emb(), pretrain)
            for i in range(len(opCountPos)):
                p = opCountPos[i]
                n = opCountNeg[i]
                if p==0 and n ==0:
                    continue
                opCountPos[i] = float(p)/float(p+n)
                opCountNeg[i] = float(n)/float(p+n)
            print(opCountNeg)
        #-----------------------------
        ##### emb loss #####
        o_state = torch.squeeze(o_state, 0)
        ops1 = torch.tensor(ops1)
        optim_policy.zero_grad()
        emb_loss = loss_emb(o_state, ops1)
        # print('emb_loss',emb_loss)
        emb_loss.backward(retain_graph=True)
        optim_policy_emb.step()

        #### option loss #####
        o_state = o_state.detach()
        states_for_o = [[] for i in range(option_net.num_option)]
        for k in range(op_ids.shape[0]):
            states_for_o[op_ids[k]].append(o_state[k])
        loss_option = 0

        for k in range(option_net.num_option):
            ko = option_net.o_all[k]
            o_state_ind = torch.argmin(torch.cdist(o_state , ko,p=2))
            option_loss = loss_emb(o_state[o_state_ind], option_net.o_all[k].squeeze(0))
            #print(option_net.o_all)
            #-------------------hitl-----------
            if use_hitl is True:
                max = np.max(np.absolute(advise))
                if max != 0:
                    a = advise/max
                else:                                        
                    a = advise
                #print("ADVICE", -a)
                option_loss = option_loss * np.exp(-a)[k] * np.exp(np.array(opCountPos))[k] # advice attention e^Advice
            ####-------------------------------
            
            loss_option += option_loss
            #print(loss_option.shape)
        loss_option = loss_option



        for i in range(option_net.num_option):
            option_loss = 0
            for j in range(i + 1, option_net.num_option):
                a_l = -loss_emb(option_net.o_all[i][0], option_net.o_all[j][0])
                
                if use_hitl is True: ####------ HITL
                    max = np.max(np.absolute(interadvice))
                    if max != 0:
                        a = interadvice/max
                    else:
                        a = interadvice
                    #print("ADVICE", -a)
                    
                    a_l = a_l * np.exp(-a)[i][j] # advice attention
                    #--------------------
                option_loss += a_l
            loss_option += option_loss

        loss_option = loss_option / option_net.num_option
        all_option = loss_option
        option_net.zero_grad()
        loss_option.backward(retain_graph=True)
        optim_option.step()

    ops = torch.stack(option_net.o_all)
    ops = torch.squeeze(ops, 1)
    ops = torch.transpose(ops,1,0)
    agent_action = policy(state,ops)
    optim_policy.zero_grad()
    agent_action=agent_action.view(-1,)
    #emb_loss = loss_emb(action, agent_action)
    
    penaltygrad = np.sign(action.data.numpy() - agent_action.data.numpy())
    #penaltygrad[np.where(penaltygrad==1)]=1.5
    pos = np.where(penaltygrad>0)
    neg = np.where(penaltygrad<0)
    opCountPos = [0]*option_net.num_option
    opCountNeg = [0]*option_net.num_option
    bincountPos = np.bincount(np.array(op_ids)[pos])
    for i, x in enumerate(bincountPos):
        opCountPos[i] = x
    bincountNeg = np.bincount(np.array(op_ids)[neg])
    for i, x in enumerate(bincountNeg):
        opCountNeg[i] = x
    #frequent_unsafe_option = S.mode(np.array(op_ids)[pos])[0][0] if len(S.mode(np.array(op_ids)[pos])[0]) > 0 else -1 
    emb_loss = weightedMSE(action, agent_action, penaltygrad)
    print("EMB LOSS:", pos, np.array(op_ids)[pos])
    
    
    emb_loss.backward(retain_graph=True)
        
    optim_policy.step()

    #### option loss #####
    if t %10==0:
        eval_state, eval_action, Tslice = expert.eval()
        otherFeaturesVal = otherFeatures[Tslice:]
        state = torch.FloatTensor(eval_state)
        action = torch.FloatTensor(eval_action)
        agent_action = policy(state,ops)
        #ops1, ops_state_lists, ops_tensor, op_ids, op_2_ids = sim(policy.forward_emb(state), option_net.get_emb(), False)
        #emb_loss = loss_emb(action, agent_action.squeeze(1))
        agent_action = agent_action.data.numpy()
        # print("AGENT ACTION: ",agent_action, op_ids)
        # with open("hitl_resultX"+str(t)+".csv","w") as file:
        #     header = 'ServiceId,hour,cpuUsage_label,BinTimeStamp,prototype'+'\n'
        #     file.write(header)
        #     for i,action in enumerate(agent_action):
        #         print(action[0])
        #         s = otherFeatures['ServiceId'][Tslice+i] + ','+str(otherFeatures['hour'][Tslice+i])+','+str(action[0])+','+str(otherFeatures['BinTimeStamp'][Tslice+i])+','+str(op_ids[i])+'\n'
        #         file.write(s)
        plt.plot(eval_action,label = 'Usage Rate',alpha = 0.4,color = 'b')
        plt.plot(agent_action,label = 'Subscription rate',alpha = 0.4,color = 'r')
        plt.legend()
        plt.show()
        foo_fig = plt.gcf() # 'get current figure'
        foo_fig.savefig(f'./imittestHITLx.png', format='png', dpi=200)
        plt.close()

        #for saving------------------
        full_state, full_action = expert.full()
        state = torch.FloatTensor(full_state)
        action = torch.FloatTensor(full_action)
        agent_action = policy(state,ops)
        ops1, ops_state_lists, ops_tensor, op_ids, op_2_ids = sim(policy.forward_emb(state), option_net.get_emb(), False)
        agent_action = agent_action.data.numpy()
        print("AGENT ACTION: ", op_ids)
        with open("hitl_resultX"+str(t)+".csv","w") as file:
            header = 'ServiceId,hour,cpuUsage_label,BinTimeStamp,prototype'+'\n'
            file.write(header)
            for i,action in enumerate(agent_action):
                #print(action[0])
                s = otherFeatures['ServiceId'][i] + ','+str(otherFeatures['hour'][i])+','+str(action[0])+','+str(otherFeatures['BinTimeStamp'][i])+','+str(op_ids[i])+'\n'
                file.write(s)
        #--------------------------

    #=---- changing option structure based on --- feedback
    if use_hitl:
        mergesplit = False
        if len(merge)>0:
            option_net.mergeOptions(merge)
            mergesplit = True
        elif len(splits)>0:
            option_net.splitOption(optionid=splits[0],cluster=optionCLuster,pointEntropy=pointEnt, policy=policy, trainState=train_state)
            mergesplit = True
        if mergesplit:
            feedback.updateOptions(num_options=option_net.num_option)
            policy = Policy(state_dim, Rep_hidden_dim, action_dim, emb_dim, option_net.num_option)
            optim_policy = torch.optim.Adam(policy.parameters(), lr=0.01)
            optim_policy_emb = torch.optim.Adam(policy.parameters(), lr=0.00001)
            optim_option = torch.optim.Adam(option_net.o_all, lr=0.0001, betas=betas)
            pretrain = True
            adhocPretrainCount +=1
            mergesplit = False
        
    #-----------------------------------------------------------
#print(advise)

