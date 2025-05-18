import numpy as np
import pickle
import os
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


def set_up_data(H):
    shift_loss = -127.5
    scale_loss = 1. / 127.5
    if H.dataset == 'cifar10':
        (trX, _), (vaX, _), (teX, _) = cifar10(H.data_root, one_hot=False)
        H.image_size = 32
        H.image_channels = 3
        shift = -120.63838
        scale = 1. / 64.16736
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    if H.test_eval:
        print('DOING TEST')
        eval_dataset = teX
    else:
        eval_dataset = vaX

    shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
    scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)
    
    train_data = TensorDataset(torch.as_tensor(trX))
    valid_data = TensorDataset(torch.as_tensor(eval_dataset))
    
    def preprocess_func(x):
        nonlocal shift
        nonlocal scale
        nonlocal shift_loss
        nonlocal scale_loss
        inp = x[0].cuda(non_blocking=True).float()
        out = inp.clone()
        inp.add_(shift).mul_(scale)
        out.add_(shift_loss).mul_(scale_loss)
        return inp, out
    
    return H, train_data, valid_data, preprocess_func


def unpickle_cifar10(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    data = dict(zip([k.decode() for k in data.keys()], data.values()))
    return data


def cifar10(data_root, one_hot=True):
    root = os.path.join(data_root, 'cifar-10-batches-py')
    
    # load training batches
    data_list, label_list = [], []
    for i in range(1, 6):
        batch_path = os.path.join(root, f'data_batch_{i}')
        batch = unpickle_cifar10(batch_path)
        data_list.append(batch['data'])
        label_list.append(batch['labels'])
    trX = np.concatenate(data_list, axis=0).astype(np.float32)               # (50000, 3072)
    trY = np.concatenate(label_list, axis=0).astype(np.int64)                # (50000,)
    
    # load test batches
    test_batch = unpickle_cifar10(os.path.join(root, 'test_batch'))
    teX = np.array(test_batch['data'], dtype=np.uint8).astype(np.float32)    # (10000, 3072)
    teY = np.array(test_batch['labels'], dtype=np.int64)                     # (10000,)
    
    trX = trX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    teX = teX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    trX, vaX, trY, vaY = train_test_split(trX, trY, test_size=5000, random_state=11172018)
    
    if one_hot:
        trY = np.eye(10, dtype=np.float32)[trY]
        vaY = np.eye(10, dtype=np.float32)[vaY]
        teY = np.eye(10, dtype=np.float32)[teY]
    else:
        trY = np.reshape(trY, [-1, 1])
        vaY = np.reshape(vaY, [-1, 1])
        teY = np.reshape(teY, [-1, 1])
    return (trX, trY), (vaX, vaY), (teX, teY)