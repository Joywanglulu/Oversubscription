from sklearn import ensemble
from torch.utils.data import Dataset
import pandas as pd
from math import floor, isnan
import random
from scipy.stats import truncnorm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data.sampler import SubsetRandomSampler
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def columnidx2time(step, latest_time):
    start_year=latest_time[0]
    start_quarter=latest_time[1]
    year=start_year+floor((start_quarter-step)/4)
    quarter=(start_quarter-step)%4
    return [year, quarter]

def demand_sim(ongate, year_norm, prob_showup=0.9):
    prob_showup_year=prob_showup*(1-0.2*year_norm**2)
    demand_mean=ongate/prob_showup_year
    demand_sigma=1000000
    demand = random.gauss(demand_mean, demand_sigma) 
    return demand

def oversale_sim(year_norm, lower_oversale=0.03, upper_oversale=0.05, mean=0, sd=1):
    oversale = truncnorm((lower_oversale - mean) / sd, (upper_oversale - mean) / sd, loc=mean, scale=sd)
    oversale = oversale.rvs()*(1+0.01*year_norm**2)
    return oversale

def build_airline_dict(df_airline):
    # current use idx as tag
    l_airline=pd.Series(df_airline.index.values.tolist())
    d_airline={}
    for idx, airline in enumerate(l_airline):
        d_airline[airline]=idx
    return d_airline

def collate_fn_reg(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_batch = torch.as_tensor(np.array([feature for (feature, target) in batch])).float().to(device)
    target_batch = torch.as_tensor(np.array([target for (feature, target) in batch])).float().to(device)
    return (feature_batch, target_batch)


class AirplaneDataset(Dataset):
    def __init__(self, processed_data_path, latest_time, ancient_time):
        super(AirplaneDataset, self).__init__()
        df=pd.read_csv(processed_data_path,header=None, index_col=0)
        # df.drop(df.tail(1).index, inplace=True)
        self.raw_df=df
        self.latest_time=latest_time
        self.ancient_time=ancient_time
        data=[]
        self.d_airline=build_airline_dict(df)
        n_time=int(df.shape[1]/3)
        n_airline=len(df)
        self.feature=[]
        self.target=[]
        for col_step in range(int(df.shape[1]/3)):
            col_time=columnidx2time(col_step, latest_time)
            for row_idx, row in df.iterrows():
                col_idx=col_step*3
                if not isnan(row.iloc[col_idx]):
                    vol_bump=int(row.iloc[col_idx])
                    invol_bump=int(row.iloc[col_idx+1])
                    onboard=int(row.iloc[col_idx+2])
                    bump=vol_bump+invol_bump
                    year_normalized=(latest_time[0]-col_time[0])/(latest_time[0]-ancient_time[0])
                    ongate=onboard+bump                    
                    quarter=col_time[1]
                    demand=demand_sim(ongate,year_normalized)
                    oversale_rate=oversale_sim(year_normalized)
                    ticket_sold=int(onboard/(1+oversale_rate))
                    record=[demand, ongate, onboard, ticket_sold, year_normalized, quarter, self.d_airline[row_idx], n_time-col_step,oversale_rate, ]
                    data.append(record)
                else:
                    data.append('nodata')

        for i in range(n_time-1):
            for j in range(n_airline):
                if data[j+n_airline*i]=='nodata' or data[j+n_airline*(i+1)]=='nodata':
                    continue
                self.feature.append(data[j+n_airline*i])
                self.target.append(data[j+n_airline*(i+1)][:4])

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        return (self.feature[index], self.target[index])

    def _plot(self):
        save_path='img/'
        df_feature=pd.DataFrame(self.feature, columns=['demand', 'ongate', 'ticket_sold', 'year_norm', 'quarter', 'oversale_rate', 'airline_id', 'time'])
        plt.figure(figsize = (20,8))
        plot_demand= sns.lineplot(data=df_feature, x='time', y='demand', hue='airline_id', palette=sns.color_palette("husl", n_colors=len(self.d_airline))).get_figure()
        plot_demand.savefig(save_path+'demand.png')
        plt.figure(figsize = (20,8))
        plot_ongate = sns.lineplot(data=df_feature, x='time', y='ongate', hue='airline_id', palette=sns.color_palette("husl", n_colors=len(self.d_airline))).get_figure()
        plot_ongate.savefig(save_path+'ongate.png')
        plt.figure(figsize = (20,8))
        plot_oversale=sns.lineplot(data=df_feature, x='time', y='oversale_rate', hue='airline_id', palette=sns.color_palette("husl", n_colors=len(self.d_airline))).get_figure()
        plot_oversale.savefig(save_path+'oversale_rate.png')
        plt.figure(figsize = (20,8))
        plot_ticket_sold=sns.lineplot(data=df_feature, x='time', y='ticket_sold', hue='airline_id', palette=sns.color_palette("husl", n_colors=len(self.d_airline))).get_figure()
        plot_ticket_sold.savefig(save_path+'ticket_sold.png')


if __name__ == "__main__":
    data_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/airplane.csv')  
    ancient_time=[1998,0]
    latest_time=[2021,2]
    valid_split=0.2 
    shuffle_dataset=False
    batch_size=1
    seed=0

    dataset=AirplaneDataset(data_path,latest_time,ancient_time)
    # dataset._plot()

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split*dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]
    train_indices, val_indices = indices, indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, collate_fn=collate_fn_reg)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, collate_fn=collate_fn_reg)
    tot_train_feature=[]
    tot_train_feature_targets=[]
    criterion = nn.MSELoss()

    for batch_index, batch_data in enumerate(train_loader):
        (feature_batch, target_batch) = batch_data
        tot_train_feature.append(feature_batch)
        tot_train_feature_targets.append(target_batch)

    tot_train_feature = torch.cat(tot_train_feature, dim=0).detach().cpu().numpy()
    tot_train_feature_targets = torch.cat(tot_train_feature_targets, dim=0).detach().cpu().numpy()

    gbdt_model_save_path = 'save/'+f'GBDT_{seed}.pth'
    gbdt = MultiOutputRegressor(ensemble.GradientBoostingRegressor(random_state=seed, max_depth=4, min_samples_leaf=2, n_estimators=5000))
    print('start gbdt')
    gbdt.fit(tot_train_feature, tot_train_feature_targets)
    pickle.dump(gbdt, open(gbdt_model_save_path, 'wb'))

    print('start val')
    val_loss_gbdt = 0.
    outputs = []
    targets = []
    for batch_index, batch_data in enumerate(validation_loader):
        (feature_batch, target_batch) = batch_data
        output = gbdt.predict(feature_batch.detach().cpu().numpy())
        outputs.append(output[0])
        targets.append(target_batch.detach().cpu().numpy()[0])
    val_loss_gbdt = mean_squared_error(outputs, targets, squared=False)
    # val_loss_gbdt = evaluate_node(gbdt, validation_loader, predict_window, t)
    print(f'val_loss gbdt {val_loss_gbdt}')
