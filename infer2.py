import torch
import pandas as pd
column_names=['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3', 's_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 's_20', 's_21']
x=pd.read_csv("train_FD001.txt",sep='\s+',header=None,names=column_names)
drop_columns=['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
x.drop(drop_columns, axis=1, inplace=True)
x = (x - x.mean()) / x.std()
x = x.iloc[2:32, :]
x=torch.Tensor(x.to_numpy()).T

from models2.GCN import GCN
model=GCN(30,1,'EdgePool')
state_dict=torch.load("./checkpoint/FD001/GCN_EdgePool_0309-222048/14-297.7675-best_model.pth")
state_dict.pop("pool.lin.weight", None)
state_dict.pop("pool.lin.bias", None)
model.load_state_dict(state_dict)
model.eval()

class input_data:
    def __init__(self,x,edge_index,batch):
        self.x          = x
        self.edge_index = edge_index
        self.batch      = batch
        
edge_index=torch.LongTensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,
          2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  4,  4,  4,  4,
          4,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,
          7,  7,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12,
         12, 13, 13, 13],
        [ 1,  2,  4,  5,  6,  8,  9, 10, 11,  0,  2,  4,  5,  6,  8,  9, 10, 11,
          0,  1,  4,  5,  6,  8,  9, 10, 11,  7, 12, 13,  0,  1,  2,  6,  8, 10,
         11,  0,  1,  2,  6,  9, 10, 11,  0,  1,  2,  4,  5,  8,  9, 10, 11,  3,
         12, 13,  0,  1,  2,  4,  6, 10, 11,  0,  1,  2,  5,  6, 10, 11,  0,  1,
          2,  4,  5,  6,  8,  9, 11,  0,  1,  2,  4,  5,  6,  8,  9, 10,  3,  7,
         13,  3,  7, 12]])
batch=torch.LongTensor([0 for _ in range(14)])
a=input_data(x,edge_index,batch)
res=torch.round(torch.squeeze(model(a,'EdgePool')))
print(int(res))