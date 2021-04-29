from __future__ import division
from __future__ import print_function

import os.path as osp
import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from model import MXMNet, Config
from utils import EMA, TUDataset, InMemoryDataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
parser.add_argument('--seed', type=int, default=920, help='Random seed.')
parser.add_argument('--epochs', type=int, default=900, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay value.')
parser.add_argument('--n_layer', type=int, default=6, help='Number of hidden layers.')
parser.add_argument('--dim', type=int, default=128, help='Size of input hidden units.')
parser.add_argument('--dataset', type=str, default="QM9", help='Dataset')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--target', type=int, default="7", help='Index of target (0~11) for prediction')
parser.add_argument('--cutoff', type=float, default=5.0, help='Distance cutoff used in the global layer')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpu)
torch.cuda.empty_cache()

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class TableDataset(InMemoryDataset):
    # Params: input file (TSV or CSV) path
    #         seperator (default = tab)
    #         header line exist (default = True)
    #         Column index (0-based) or name for SMILES
    #         Column index (0-based) or name for Y value
    def __init__(self, root, input_file_path, col_smiles, col_y, sep='\t', header=True, transform=None, pre_transform=None):
        self.input_file_path = input_file_path
        self.col_smiles = col_smiles
        self.col_y = col_y
        self.sep = sep
        self.header = header
        self.data_list = []

        super(TableDataset, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        pass

    @property
    def raw_file_names(self):
        return ['raw.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        with open(self.input_file_path, 'r') as fin:
            smiles_list = fin.readlines()

            if self.header:
                header_line = smiles_list.pop(0).strip().split(self.sep)
                smiles_idx = find_str_or_int_from_list(self.col_smiles, header_line)
                y_idx = find_str_or_int_from_list(self.col_y, header_line)
                assert 0 <= smiles_idx < len(header_line), f'Cannot find SMILES in header: {self.col_smiles}'
                assert 0 <= y_idx < len(header_line), f'Cannot find Y in header: {self.col_y}'
            else:
                assert isinstance(self.col_smiles, int), f'Column info must be an integer when no header line.'
                assert isinstance(self.col_y, int), f'Column info must be an integer when no header line.'
                assert 0 <= self.col_smiles < len(smiles_list[0].split(self.sep)), f'Invalid column info: {self.col_smiles}'
                assert 0 <= self.col_y < len(smiles_list[0].split(self.sep)), f'Invalid column info: {self.col_y}'

            for line in smiles_list:
                token = line.strip().split('\t')
                print(token[smiles_idx])
                mol = Chem.MolFromSmiles(token[smiles_idx])
                node_attr, edge_index, edge_attr = mol2graph(mol)

                self.data_list.append(Data(x=torch.tensor(node_attr, dtype=torch.float),
                                           edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                                           edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                                           y=torch.tensor([float(token[y_idx])], dtype=torch.float)))

        self.data, self.slices = self.collate(self.data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])


target = args.target
set_seed(args.seed)

targets = ['mu (D)', 'a (a^3_0)', 'e_HOMO (eV)', 'e_LUMO (eV)', 'delta e (eV)', 'R^2 (a^2_0)', 'ZPVE (eV)', 'U_0 (eV)', 'U (eV)', 'H (eV)', 'G (eV)', 'c_v (cal/mol.K)', ]
scale = 1.0
#Change the unit from meV to eV for energy-related targets
if target in [2, 3, 4, 6, 7, 8, 9, 10]:
    scale = 1000.0

def test(loader):
    error = 0
    ema.assign(model)

    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        error += (output - data.y).abs().sum().item()
    ema.resume(model)
    return error / len(loader.dataset)

class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, target] / scale
        return data

# Load dataset
path = osp.join('.', 'data')
dataset = TUDataset(path, name=args.dataset, transform=MyTransform(), use_node_attr=True, use_edge_attr=True).shuffle()
#path = osp.join('.', 'data', 'oled_dataset_ref_292_V_Raw')
#oled_292_dataset = TableDataset(path, None, None, None)

# Split dataset
train_dataset = dataset[:110000]
val_dataset = dataset[110000:120000]
test_dataset = dataset[120000:]

#train_dataset = oled_292_dataset[:52]
#val_dataset = oled_292_dataset[52:64]
#test_dataset = oled_292_dataset[64:]

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=args.seed)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

print('Loaded the QM9 dataset. Target property: ', targets[target])

# Load model
config = Config(dim=args.dim, n_layer=args.n_layer, cutoff=args.cutoff)

model = MXMNet(config).to(device)
print('Loaded the MXMNet.')

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

ema = EMA(model, decay=0.999)

print('===================================================================================')
print('                                Start training:')
print('===================================================================================')

best_epoch = None
best_val_loss = None

for epoch in range(args.epochs):
    loss_all = 0
    step = 0
    model.train()

    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()       # Init grad

        output = model(data.x, data.edge_index, data.batch)
        loss = F.l1_loss(output, data.y)        # calculate loss
        loss_all += loss.item() * data.num_graphs
        loss.backward()         # calculate gradients
        clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2) #
        optimizer.step()        # update parameters
        
        curr_epoch = epoch + float(step) / (len(train_dataset) / args.batch_size)
        scheduler_warmup.step(curr_epoch)   # warmup LR??

        ema(model)  # Exponential moving avaerage
        step += 1

    train_loss = loss_all / len(train_loader.dataset)

    val_loss = test(val_loader)

    if best_val_loss is None or val_loss <= best_val_loss:
        test_loss = test(test_loader)
        best_epoch = epoch
        best_val_loss = val_loss

    print('Epoch: {:03d}, Train MAE: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f}'.format(epoch+1, train_loss, val_loss, test_loss))

print('===================================================================================')
print('Best Epoch:', best_epoch)
print('Best Test MAE:', test_loss)

torch.save({'state_dict': model.state_dict(),
            'epochs': args.epochs,
            'lr': args.lr,
            'wd': args.wd,
            'n_layer': args.n_layer,
            'dim': args.dim,
            'batch_size': args.batch_size,
            'cutoff': args.cutoff},
           f'QM9_{targets[target]}.pt')
