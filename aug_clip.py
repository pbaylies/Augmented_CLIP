# Augmenting CLIP, 2021, by Peter Baylies (@pbaylies)
# Simple MLPs for training against LAION400m embeddings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import shutil

def save_ckp(state, is_best, prefix=''):
    f_path = prefix + '_checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = prefix + '_best_model.pt'
        shutil.copyfile(f_path, best_fpath)

in_size = 512
out_size = 512
device = torch.device('cuda')
model1 = nn.Sequential(
    nn.Linear(in_size, in_size, True),
    nn.ELU(),
    nn.Linear(in_size, in_size, True),
    nn.ELU(),
    nn.Linear(in_size, in_size, True),
    nn.ELU(),
    nn.Linear(in_size, out_size, True),
    nn.Tanh(),
).to(device)
model2 = nn.Sequential(
    nn.Linear(in_size, in_size, True),
    nn.ELU(),
    nn.Linear(in_size, in_size, True),
    nn.ELU(),
    nn.Linear(in_size, in_size, True),
    nn.ELU(),
    nn.Linear(in_size, out_size, True),
    nn.Tanh(),
).to(device)
#model = torch.load("checkpoint.pt")

batch_size = 256

lr = 1e-3
optim1 = torch.optim.AdamW(model1.parameters(), lr)
optim2 = torch.optim.AdamW(model2.parameters(), lr)
smoothl1 = nn.SmoothL1Loss()

count = 0
min_loss1 = 1000000
loss1 = min_loss1 - 1
min_loss2 = 1000000
loss2 = min_loss2 - 1
steps = 1000
for d in range(410):
    inputs = torch.from_numpy(np.load('images/img_emb_%d.npy' % d)).float().to(device)
    outputs = torch.from_numpy(np.load('texts/text_emb_%d.npy' % d)).float().to(device)
    inputs /= inputs.norm(dim=-1, keepdim=True)
    outputs /= outputs.norm(dim=-1, keepdim=True)
    dataset = TensorDataset(inputs,outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for input, target in dataloader:
        count += 1
        optim1.zero_grad()
        output = model1(input)
        loss1 = 1000*smoothl1(output, target)
        loss1.backward()
        optim1.step()
        if count % steps == 0:
            print(count)
            is_best = loss1.sum() < min_loss1
            save_ckp(model1, is_best, prefix='i2t')
            if is_best:
                min_loss1 = loss1
                print('i2t: ' + str(min_loss1))

        input, target = target, input
        optim2.zero_grad()
        output = model2(input)
        loss2 = 1000*smoothl1(output, target)
        loss2.backward()
        optim2.step()
        if count % steps == 0:
            is_best = loss2.sum() < min_loss2
            save_ckp(model2, is_best, prefix='t2i')
            if is_best:
                min_loss2 = loss2
                print('t2i: ' + str(min_loss2))
if count % steps != 0:
    print(count)
    is_best = loss1 < min_loss1
    save_ckp(model1, is_best, prefix='i2t')
    if is_best:
        min_loss1 = loss1
        print('i2t: ' + str(min_loss1))
    is_best = loss2 < min_loss2
    save_ckp(model2, is_best, prefix='t2i')
    if is_best:
        min_loss2 = loss2
        print('t2i: ' + str(min_loss2))
