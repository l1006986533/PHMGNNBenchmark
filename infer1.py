import torch
import pandas as pd
a=pd.read_pickle("./data/XJTUGearbox/XJTUGearboxRadius.pkl")
x=torch.FloatTensor(a[114].x)

from models.GCN import GCN
model=GCN(1024,9)
model.load_state_dict(torch.load("./checkpoint/Node_GCN_XJTUGearboxRadius_TD_0309-223044/25-0.8278-best_model.pth"))
model.eval()

class input_data:
    def __init__(self,x,edge_index,edge_attr):
        self.x          = x
        self.edge_index = edge_index
        self.edge_attr  = edge_attr

edge_index=torch.LongTensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                              2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                              6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
                              8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9], 
                             [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 3, 4, 5, 6, 7, 8, 9, 
                              0, 1, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 4, 5, 6, 7, 8, 9, 
                              0, 1, 2, 3, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 6, 7, 8, 9, 
                              0, 1, 2, 3, 4, 5, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 8, 9, 
                              0, 1, 2, 3, 4, 5, 6, 7, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8]])
edge_attr=torch.FloatTensor([1 for _ in range(90)])
a=input_data(x,edge_index,edge_attr)
res=model(a)[0].argmax()
print(int(res))