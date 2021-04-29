from __future__ import division
from __future__ import print_function



import os
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
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay value.')
parser.add_argument('--n_layer', type=int, default=6, help='Number of hidden layers.')
parser.add_argument('--dim', type=int, default=128, help='Size of input hidden units.')
parser.add_argument('--dataset', type=str, default="QM9", help='Dataset')
parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
parser.add_argument('--target', type=int, default="7", help='Index of target (0~11) for prediction')
parser.add_argument('--cutoff', type=float, default=5.0, help='Distance cutoff used in the global layer')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpu)


def find_str_or_int_from_list(target, str_list):
    target_idx = -1
    if isinstance(target, str):
        for i, token in enumerate(str_list):
            if token.lower() == target.lower():
                target_idx = i
                break
    else:
        assert 0 <= target < str_list.len(), f'Invalid column info: {target}'
        target_idx = target

    return target_idx


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
        pass


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def test(model, loader, verbose=False):
    error = 0
    ema.assign(model)

    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        if verbose:
            print('True y:', data.y)
            print('Model output:', output)
        error += (output - data.y).abs().sum().item()
    ema.resume(model)
    return error / len(loader.dataset)

target = args.target
set_seed(args.seed)

#targets = ['mu (D)', 'a (a^3_0)', 'e_HOMO (eV)', 'e_LUMO (eV)', 'delta e (eV)', 'R^2 (a^2_0)', 'ZPVE (eV)', 'U_0 (eV)', 'U (eV)', 'H (eV)', 'G (eV)', 'c_v (cal/mol.K)', ]
#scale = 1.0
#Change the unit from meV to eV for energy-related targets
#if target in [2, 3, 4, 6, 7, 8, 9, 10]:
#    scale = 1000.0


# Load dataset
#path = osp.join('.', 'data', 'oled_dataset_ref_292_V_Raw')
#oled_292_dataset = TableDataset(path, None, None, None)
#path = osp.join('.', 'data', 'soled_dataset_HIL_Homo')
path = osp.join('.', 'data', 'soled_dataset_HIL_V')
soled_hil_dataset = TableDataset(path, None, None, None)

import sklearn
from sklearn.model_selection import KFold

n_folds = 6
earlystopping_patience = 50

y_true = []
y_pred = []

config = Config(dim=args.dim, n_layer=args.n_layer, cutoff=args.cutoff)

model = MXMNet(config).to(device)
print('Loaded the MXMNet.')

#init_checkpoint = model.state_dict()

no_improve_cnt = 0
for i in range(n_folds):
    test_start = int(soled_hil_dataset.len() / n_folds) * i
    test_end = int(soled_hil_dataset.len() / n_folds) * (i + 1)
    if i == n_folds - 1:
        test_end = soled_hil_dataset.len() - 1
        val_start = int(soled_hil_dataset.len() / n_folds) * (i - 1)
        val_end = test_start
        train_dataset = soled_hil_dataset[:val_start]
        val_dataset = soled_hil_dataset[val_start:val_end]
        test_dataset = soled_hil_dataset[test_start:test_end]
    else:
        val_start = test_end
        val_end = int(soled_hil_dataset.len() / n_folds) * (i + 2)
        train_dataset = soled_hil_dataset[:test_start] + soled_hil_dataset[val_end:]
        val_dataset = soled_hil_dataset[val_start:val_end]
        test_dataset = soled_hil_dataset[test_start:test_end]

    print(f'Fold {i}: Train dataset ({len(train_dataset)}), Val dataset ({len(val_dataset)}), Test dataset ({len(test_dataset)})')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=args.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    #soled_train_dataset = soled_hil_dataset[:50]
    #soled_val_dataset = soled_hil_dataset[50:60]
    #soled_test_dataset = soled_hil_dataset[60:]

    #soled_train_loader = DataLoader(soled_train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=args.seed)
    #soled_val_loader = DataLoader(soled_val_dataset, batch_size=args.batch_size, shuffle=False)
    #soled_test_loader = DataLoader(soled_test_dataset, batch_size=args.batch_size, shuffle=False)

    #model.load_state_dict(init_checkpoint)

    #### Load pretrained model
    qm9_checkpoint = torch.load('QM9_e_HOMO (eV).pt')
    model.load_state_dict(qm9_checkpoint['state_dict'])
    args.cutoff = qm9_checkpoint['cutoff']

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

    ema = EMA(model, decay=0.999)

    torch.cuda.empty_cache()

    #model_soled = MXMNet(config).to(device)
    #optimizer_soled = optim.Adam(model_soled.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
    #scheduler_soled = torch.optim.lr_scheduler.ExponentialLR(optimizer_soled, gamma=0.9961697)
    #scheduler_warmup_soled = GradualWarmupScheduler(optimizer_soled, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

    #ema_soled = EMA(model_soled, decay=0.999)

    best_epoch = None
    best_val_loss = None
    best_hil_val_loss = None

    for epoch in range(args.epochs):
        loss_all = 0
        step = 0
        model.train()
        """model_soled.train()
    
        #model_soled.load_state_dict(model.state_dict())
    
        #for data in soled_train_loader:
            data = data.to(device)
    
            optimizer_soled.zero_grad()  # Init grad
    
            output = model_soled(data.x, data.edge_index, data.batch)
    
            loss = F.l1_loss(output, data.y)  # calculate loss
            loss_all += loss.item() * data.num_graphs
            loss.backward()  # calculate gradients
            clip_grad_norm_(model_soled.parameters(), max_norm=1000, norm_type=2)  #
            optimizer_soled.step()  # update parameters
    
            curr_epoch = epoch + float(step) / (len(soled_train_dataset) / args.batch_size)
            scheduler_warmup_soled.step(curr_epoch)  # warmup LR??
    
            ema(model)  # Exponential moving avaerage
            step += 1
    
        train_loss = loss_all / len(soled_train_loader.dataset)
    
        val_loss = test(model_soled, soled_val_loader, verbose = False)
    
        if best_hil_val_loss is None or val_loss <= best_hil_val_loss:
            test_hil_loss = test(model_soled, test_loader, verbose=False)
            best_epoch = epoch
            best_hil_val_loss = val_loss
    
        print('Epoch: {:03d}, HIL Train MAE: {:.7f}, HIL Validation MAE: {:.7f}, '
              'HIL Test MAE: {:.7f}'.format(epoch + 1, train_loss, val_loss, test_hil_loss))
    
        model.load_state_dict(model_soled.state_dict())
        """
        for data in train_loader:
            data = data.to(device)
            #print(data.x)

            optimizer.zero_grad()  # Init grad

            output = model(data.x, data.edge_index, data.batch)

            loss = F.l1_loss(output, data.y)  # calculate loss
            loss_all += loss.item() * data.num_graphs
            loss.backward()  # calculate gradients
            clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)  #
            optimizer.step()  # update parameters

            curr_epoch = epoch + float(step) / (len(train_dataset) / args.batch_size)
            scheduler_warmup.step(curr_epoch)  # warmup LR??

            ema(model)  # Exponential moving avaerage
            step += 1

        train_loss = loss_all / len(train_loader.dataset)

        val_loss = test(model, val_loader, verbose = False)

        temp = ''

        if best_val_loss is None or val_loss <= best_val_loss:
            error = 0
            ema.assign(model)
            y_true_temp = []
            y_pred_temp = []
            for data in test_loader:
                data = data.to(device)
                output = model(data.x, data.edge_index, data.batch)
                y_true_temp += data.y
                y_pred_temp += output
                error += (output - data.y).abs().sum().item()

            ema.resume(model)
            test_loss = error / len(test_loader.dataset)

            best_epoch = epoch
            best_val_loss = val_loss
            temp = ' <= best'
            no_improve_cnt = 0
        else:
            if epoch > 100:
                no_improve_cnt += 1
                if no_improve_cnt > earlystopping_patience:
                    print(f'Epoch: {epoch + 1:03d}, Train MAE: {train_loss:.7f}, Validation MAE: {val_loss:.7f}, Test MAE: {test_loss:.7f}, Early Stopped')
                    break

        print(f'Epoch: {epoch + 1:03d}, Train MAE: {train_loss:.7f}, Validation MAE: {val_loss:.7f}, Test MAE: {test_loss:.7f} {temp}')

    print('===================================================================================')
    print(f'Best Epoch for Fold {i}: {best_epoch}')
    print(f'Best Test MAE for Fold {i}: {test_loss}')
    y_true += y_true_temp
    y_pred += y_pred_temp

cv_result = open("cv_results.tsv", 'w')
cv_result.write(y_true, '\n')
cv_result.write(y_pred, '\n')
cv_result.close()
# 292 only. Epoch: 219, Train MAE: 0.0586225, Validation MAE: 0.0432742, Test MAE: 0.1075440
# with HIL. Epoch: 149, HIL Train MAE: 0.7478834, HIL Validation MAE: 0.3716734, HIL Test MAE: 0.0997167
#           Epoch: 149, Train MAE: 0.9897208, Validation MAE: 0.0549521, Test MAE: 0.0997167


#python soled_test.py --gpu 1 --epochs 300 --n_layer 6 (292 only)
#Best Epoch: 247
#Best Test MAE: 0.11774170398712158
#Epoch: 247, Train MAE: 0.0467853, Validation MAE: 0.0444346, Test MAE: 0.1183466

#python soled_test.py --gpu 1 --epochs 1500 --n_layer 6 (QM9 HOMO > 292)
#Epoch: 607, Train MAE: 0.0149887, Validation MAE: 0.0592838, Test MAE: 0.0924670
