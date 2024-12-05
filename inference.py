# coding: utf-8
import json
import os
import numpy as np
import random
import subprocess
from datetime import datetime
import logging
import sys,signal
from torch.utils import data
from tensorboardX import SummaryWriter
import datetime
import time
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt

from torch.autograd import Variable
from collections import OrderedDict
from tqdm import tqdm
from torch import nn
from torchsummary import summary
import torch
from torchvision import models, utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import linalg as LA

source_path = os.path.join('./sequence')
sys.path.append(source_path)
from multi_label_dataloader_v2 import get_dataloader
from models.test_model_v3 import ResNetBackbone3 as ResNetBackbone
from torch_utils import eval_red_model,display_red_eval_tb,train_logging,save_inter_graph,dump_to_txt
from runjobs_utils import init_logger,Saver,DataConfig,torch_load_model
from hyperparameter import *
from ground_truth import folder_lst

logger = init_logger(__name__)
logger.setLevel(logging.INFO)

starting_time = datetime.datetime.now()

## Deterministic training
_seed_id = 100
torch.backends.cudnn.deterministic = True
torch.manual_seed(_seed_id)

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def tensor_boardcast(tensor, batch_size):
    """wrap numpy in tensor."""
    tensor = torch.from_numpy(tensor[np.newaxis,:].astype(np.float32))
    tensor = tensor.repeat(batch_size,1,1)
    var_tensor = Variable(tensor.cuda())    # GX: what if I do not have the Variable?
    return var_tensor

def load_coarse_matrix(batch_size):
    """
        loads the pseduo label for the matching matrix and correlation matrix.
    """
    coarse_m_1 = './coarse_matrix_2.npy'
    coarse_m_2 = './coarse_matrix_3.npy'

    cm_1 = np.load(coarse_m_1)  
    cm_2 = np.load(coarse_m_2)

    cm_21 = softmax(cm_1, axis=0)
    cm_32 = softmax(cm_2, axis=0)

    cm_12 = softmax(cm_1.T, axis=0)
    cm_23 = softmax(cm_2.T, axis=0)

    cm_12 = tensor_boardcast(cm_12, batch_size) 
    cm_23 = tensor_boardcast(cm_23, batch_size)
    cm_21 = tensor_boardcast(cm_21, batch_size) 
    cm_32 = tensor_boardcast(cm_32, batch_size)
    cm_1  = tensor_boardcast(cm_1, batch_size)
    cm_2  = torch.permute(cm_1, (0,2,1))

    cm_12 = torch.permute(cm_12, (0,2,1))
    cm_21 = torch.permute(cm_21, (0,2,1))
    return Variable(cm_12.cuda()), Variable(cm_21.cuda()), Variable(cm_1.cuda()), Variable(cm_2.cuda())

def initialize_adjecent(task_num, batch_size, adj_file='./adj_matrix.npy', 
                        self_loop=True):
    #### initialize the adjcent matrix ####
    logger.info(f'################################')
    if os.path.isfile(adj_file):
        logger.info(f'Loading the pre-defined adjcent matrix from {adj_file}...')
        A = np.load(adj_file)
        A = A[:task_num][:task_num]
    else:
        logger.info(f'Does not have the valid adjcent matrix.')
        np.random.seed(0)
        A = np.ones((task_num, task_num))
        if not isinstance(A, (np.ndarray)):
            raise ValueError
        A_sum = np.sum(A, axis=1)
        A = A/A_sum[:, np.newaxis]
    # if self_loop:
    #     A_I = np.eye(task_num, dtype=int)
    #     A = A - A_I
    #     logger.info(f'Self-loop can be applied on this adjcent_matrix.')
    logger.info(f'The final adj_matrix has {np.sum(A)} edge...')
    # import sys;sys.exit(0)
    A = torch.from_numpy(A[np.newaxis,:].astype(np.float32))
    A = A.repeat(batch_size,1,1)
    adj = Variable(A.cuda())    
    return adj

def dataloader_return(args, label_list, normalize, batch_size):
    manipulations_dict = None
    datasets = ['ce', 'non-ce']

    ## we have to re-init this to use all the samples to init the center c
    balanced_minibatch_opt = True
    val_batch_size = 256
    train_generator, train_dataset = get_dataloader(args.img_path, datasets, label_list, folder_lst,
                                                    manipulations_dict, normalize, 'train', 
                                                    batch_size, workers=8, cv=args.cross_val
                                                    )
    val_generator, val_dataset     = get_dataloader(args.img_path, datasets, label_list, folder_lst,
                                                    manipulations_dict, normalize, 'val', 
                                                    val_batch_size, workers=8, cv=args.cross_val
                                                    )
    adjcent_matrix = initialize_adjecent(args.task_num, batch_size=batch_size)
    mm_1_GT, mm_2_GT, cm_1_GT, cm_2_GT = load_coarse_matrix(args.batch_size)
    ## Centers are computed we can del the dataloader to free up gpu.
    del train_dataset
    del val_dataset

    return train_generator, val_generator, adjcent_matrix, mm_1_GT, mm_2_GT, cm_1_GT, cm_2_GT

def config_setup(args):
    hparams['epochs'] = args.epoch
    hparams['valid_epoch'] = args.val_epoch
    hparams['basic_lr'] = basic_lr = args.lr
    hparams['batch_size'] = batch_size = args.batch_size
    pair_dist = torch.nn.PairwiseDistance(p=2)
    exp_num = 'camera-ready'    ## camera-ready version
    model_name = exp_num + f'_{args.task_num}_bs_{args.batch_size}_lr_{args.lr}_cv_{args.cross_val}'
    args.att_graph_folder = exp_num + f'_att_graph_cv_{args.cross_val}_lr_{args.lr}'
    os.makedirs(args.att_graph_folder, exist_ok=True)
    model_path = os.path.join(args.model_folder, model_name)
    txt_file   = os.path.join(model_path, 'test_inference_v3.txt')
    tb_folder  = os.path.join(f'./{args.model_folder}/tb_logs',model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(tb_folder)
    log_string_config = '  '.join([k+':'+str(v) for k,v in hparams.items()])
    writer.add_text('config : %s' % model_name, log_string_config, 0)
    label_list = []

    # Create the model path if doesn't exists
    if not os.path.exists(model_path):
        subprocess.call(f"mkdir -p {model_path}", shell=True)
    args.device = device
    # args.txt_file = txt_file
    if not os.path.exists(txt_file):
        args.txt_handler = open(txt_file, 'w')
    else:
        args.txt_handler = open(txt_file, 'a')
    logger.info(f'Set up the configuration for {model_name}...')
    return pair_dist, model_name, label_list, model_path, tb_folder, device, writer

def main(args):
    ## Configuration
    pair_dist, model_name, label_list, model_path, tb_folder, device, writer = config_setup(args)
    ## Dataloader
    train_generator, val_generator, adjcent_matrix, mm_1_GT, mm_2_GT, cm_1_GT, cm_2_GT = \
            dataloader_return(args, label_list, normalize, args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNetBackbone(
                            class_num=2,
                            task_num=args.task_num,
                            feat_dim=512,
                            )
    model = model.to(device)
    model = nn.DataParallel(model)

    ## Fine-tuning functions
    params_to_optimize = model.parameters()
    optimizer = torch.optim.Adam(params_to_optimize, lr=basic_lr, weight_decay=weight_decay)
    lr_scheduler = ReduceLROnPlateau(
                                    optimizer, 
                                    mode='min', 
                                    factor=step_factor, 
                                    min_lr=1e-06, 
                                    patience=patience, 
                                    verbose=True
                                    )
    pair_dist = torch.nn.PairwiseDistance(p=2)

    ## Re-loading the model in case
    epoch_init=epoch=ib=ib_off=before_train=0
    load_model_path = os.path.join(model_path,'current_model.pth')
    # load_model_path = os.path.join(model_path,'best_model.pth')
    val_loss = np.inf
    if os.path.exists(load_model_path):
        logger.info(f'Loading weights, optimizer and scheduler from {load_model_path}...')
        ib_off, epoch_init, scheduler, val_loss = torch_load_model(model, optimizer, load_model_path)
    else:
        logger.info(f'Training from the scratch.')

    ## Saver object and data config
    data_config = DataConfig(model_path, model_name)
    saver = Saver(model, optimizer, lr_scheduler, data_config, starting_time, hours_limit=23, mins_limit=0)

    if epoch_init == 0:
        model.zero_grad()

    ## Start training
    epoch = 'INFERENCE'
    criterion_2 = nn.CrossEntropyLoss().to(device)
    logger.info(f'Begin the inference.')
    res_f1_1, res_f1_2, accur, val_loss_avg = eval_red_model(
                                                            model,
                                                            val_generator,
                                                            criterion_2,
                                                            device,
                                                            adjcent_matrix[:256],
                                                            task_num=args.task_num,
                                                            desc='valid',
                                                            # debug=True
                                                            )
    dump_to_txt(args, epoch, res_f1_1, res_f1_2, accur, val_loss_avg)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  ## Configuration
  parser.add_argument('--img_path', type=str, 
                        default="./RED140",
                        help="where to store the image.")
  parser.add_argument('--model_folder', type=str, default='./expts', help='Output result to folder.')
  parser.add_argument('--cross_val', type=int, default=1, choices=[1,2,3,4], help='Cross dataset')

  ## Train hyper-parameters
  parser.add_argument('--epoch', type=int, default=5, help='How many epochs to train.')
  parser.add_argument('--val_epoch', type=int, default=10, help='How many epochs to val.')
  parser.add_argument('--display_step', type=int, default=10, help='The display epcoh.')
  parser.add_argument('--batch_size', type=int, default=256, help='The batch size.')
  parser.add_argument('--lr', type=float, default=0.1, help='The starting learning rate.')
  parser.add_argument('--protocol', type=str, default='None', help='What parameter to reverse?')
  parser.add_argument('--task_num', type=int, default=55, help='How hyper-parameter to parse.')
  args = parser.parse_args()
  main(args)