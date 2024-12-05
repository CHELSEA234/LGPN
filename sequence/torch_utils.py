import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
import numpy as np
from runjobs_utils import init_logger
import logging
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

logger = init_logger(__name__)
logger.setLevel(logging.INFO)

def eval_red_model(model,val_generator,criterion,device,adj,task_num=29,desc='valid',debug=False):
    # https://stackoverflow.com/a/67064979
    # pred_lst  = [[]]*task_num
    # label_lst = [[]]*task_num
    pred_lst  = [[] for _ in range(task_num)]
    label_lst = [[] for _ in range(task_num)]
    val_loss_total = 0
    model.eval()

    for jb, val_batch in enumerate(tqdm(val_generator),1):
        if debug and jb == 5:
            break
        
        val_img_batch_mmodal, val_true_label, val_real_fake_label, image_name = val_batch
        val_img_batch_mmodal = val_img_batch_mmodal.float().to(device)
        val_true_label = val_true_label.long().to(device)

        if adj.size()[0] != val_img_batch_mmodal.size()[0]:
            break
            
        _, pred_ce_val, pred_gcn_val, _ = model(val_img_batch_mmodal, adj)
        loss_lst = [criterion(pred_ce_val[:,_,:], val_true_label[:,_]) for _ in range(task_num)]
        loss = sum(loss_lst)
        val_loss_total += loss.item()
        for _ in range(task_num):
            log_probs = F.softmax(pred_ce_val[:,_,:],dim=-1)
            res_probs = torch.argmax(log_probs, dim=-1)
            res_probs = list(res_probs.cpu().numpy())
            val_true  = list(val_true_label[:,_].cpu().numpy())
            pred_lst[_].extend(res_probs)
            label_lst[_].extend(val_true)

    ma_f1_lst, mi_f1_lst, acc_lst = [], [], []
    for _ in range(task_num):   
        # print("_ is: ", pred_lst[_][:5])
        res_f1_1 = f1_score(pred_lst[_], label_lst[_], average='macro')
        res_f1_2 = f1_score(pred_lst[_], label_lst[_], average='weighted')
        accur = accuracy_score(label_lst[_], pred_lst[_])
        ma_f1_lst.append(res_f1_1)
        mi_f1_lst.append(res_f1_2)
        acc_lst.append(accur)

    val_loss_avg = val_loss_total / (jb+1)
    model.train()
    return ma_f1_lst, mi_f1_lst, acc_lst, val_loss_avg

def eval_red_FID_model(model,val_generator,criterion,device,adj,task_num=29,desc='valid',debug=False):
    ## GX: https://stackoverflow.com/a/67064979
    pred_lst  = []
    label_lst = []
    val_loss_total = 0
    model.eval()
    for jb, val_batch in enumerate(tqdm(val_generator),1):
        if debug and jb == 5:
            break

        val_img_batch_mmodal, val_true_label, image_name = val_batch
        val_img_batch_mmodal = val_img_batch_mmodal.float().to(device)
        val_true_label = val_true_label.long().to(device)

        if adj.size()[0] != val_img_batch_mmodal.size()[0]:
            break
            
        pred_ce_val = model(val_img_batch_mmodal)
        loss = criterion(pred_ce_val, val_true_label)
        val_loss_total += loss.item()

        log_probs = F.log_softmax(pred_ce_val, dim=-1)
        res_probs = torch.argmax(log_probs, dim=-1)
        res_probs = list(res_probs.cpu().numpy())
        val_true  = list(val_true_label.cpu().numpy())
        pred_lst.extend(res_probs)
        label_lst.extend(val_true)

    res_f1_1 = f1_score(pred_lst, label_lst, average='macro')
    res_f1_2 = f1_score(pred_lst, label_lst, average='weighted')
    accur = accuracy_score(label_lst, pred_lst)
    val_loss_avg = val_loss_total / (jb+1)
    return res_f1_1, res_f1_2, accur, val_loss_avg

def format_str(input_list):
    res = ''
    for value in input_list:
        res += "%.3f" % value
        res += ' '
    return res

def dump_to_txt_helper(args, res_list):
    obj_lst = res_list[:10]
    con_arc = res_list[10:37]
    dis_arc = res_list[37:]
    obj_res, con_res, dis_res = np.mean(obj_lst), np.mean(con_arc), np.mean(dis_arc)
    res_str = f"{obj_res:.3f}, {con_res:.3f}, {dis_res:.3f}."
    
    obj_lst = format_str(obj_lst)
    con_arc = format_str(con_arc)
    dis_arc = format_str(dis_arc)

    args.txt_handler.write(dis_arc+'\n')
    args.txt_handler.write('Mean: '+res_str+'\n')

def dump_to_txt(args, epoch, res_f1_1, res_f1_2, accur, val_loss_avg):
    args.txt_handler.write('epoch: '+str(epoch))
    args.txt_handler.write('\n')
    args.txt_handler.write('macro f1: '+str(epoch))
    args.txt_handler.write('\n')
    dump_to_txt_helper(args, res_f1_1)
    args.txt_handler.write('weighted f1: '+str(epoch))
    args.txt_handler.write('\n')
    dump_to_txt_helper(args, res_f1_2)
    args.txt_handler.write('accuracy: '+str(epoch))
    args.txt_handler.write('\n')
    dump_to_txt_helper(args, accur)
    args.txt_handler.flush()

def tensor_to_numpy(ten_gpu, trans=False):
    res = ten_gpu.cpu().detach().numpy()
    return res if not trans else np.transpose(res)

def save_inter_graph(args, epoch, pred_fea, gcn_out, image_name, 
                    mm_1, mm_2, cg_1, cg_2, debug=False):
    idx = 0
    pred_ = pred_fea[idx]
    adj_0 = gcn_out[0][idx]
    adj_1 = gcn_out[1][idx]
    adj_2 = gcn_out[2][idx]
    name_ = image_name[idx].split('/')[-1]

    mm_1 = tensor_to_numpy(mm_1[idx])
    mm_2 = tensor_to_numpy(mm_2[idx], True)
    cg_1 = tensor_to_numpy(cg_1[idx])
    cg_2 = tensor_to_numpy(cg_2[idx])
    adj_0 = tensor_to_numpy(adj_0)
    adj_1 = tensor_to_numpy(adj_1)
    adj_2 = tensor_to_numpy(adj_2)

    np.save(f"{args.att_graph_folder}/mm1_{epoch}.npy", mm_1)
    np.save(f"{args.att_graph_folder}/mm2_{epoch}.npy", mm_2)
    np.save(f"{args.att_graph_folder}/cm1_{epoch}.npy", cg_1)
    np.save(f"{args.att_graph_folder}/cm2_{epoch}.npy", cg_2)
    np.save(f"{args.att_graph_folder}/adj0_{epoch}.npy", adj_0)
    np.save(f"{args.att_graph_folder}/adj1_{epoch}.npy", adj_1)
    np.save(f"{args.att_graph_folder}/adj2_{epoch}.npy", adj_2)

    ax0 = plt.subplot(2, 3, 1)
    ax0.imshow(mm_1, cmap='hot', interpolation='nearest')
    ax0.set_title("matching 1")
    ax1 = plt.subplot(2, 3, 2)
    ax1.imshow(mm_2, cmap='hot', interpolation='nearest')
    ax1.set_title("matching 2")
    ax2 = plt.subplot(2, 3, 3)
    ax2.imshow(adj_0, cmap='hot', interpolation='nearest')
    ax2.set_title("graph 1")
    ax3 = plt.subplot(2, 3, 4)
    ax3.imshow(adj_1, cmap='hot', interpolation='nearest')
    ax3.set_title("graph 2")
    ax4 = plt.subplot(2, 3, 5)
    ax4.imshow(adj_2, cmap='hot', interpolation='nearest')
    ax4.set_title("graph 3")
    plt.savefig(f"{args.att_graph_folder}/mm_{epoch}.png")

def display_red_eval_tb(writer,macro_f1,micro_f1,accu,val_loss_avg,tot_iter,desc='valid'):
    writer.add_scalar('%s/loss'%desc, val_loss_avg, tot_iter)
    writer.add_scalar('%s/accu'%desc, accu, tot_iter)
    writer.add_scalar('%s/macro_f1'%desc, macro_f1, tot_iter)
    writer.add_scalar('%s/weighted_f1'%desc, micro_f1, tot_iter)

def display_eval_tb(writer,metrics,tot_iter,desc='valid',old_metrics=False):
    avg_loss = metrics.get_avg_loss()
    acc = metrics.get_acc()
    ## in case of test we report the accuracy
    ## with the best thrs from the validation
    if desc != 'valid':
        acc = metrics.tuned_acc_thrs[0]
        thrs = metrics.tuned_acc_thrs[1]        
    auc = metrics.roc.auc
    writer.add_scalar('%s/loss'%desc, avg_loss, tot_iter)
    writer.add_scalar('%s/acc'%desc, acc, tot_iter)
    ## we also write the best thrs found in the validation 
    if desc != 'valid':
        writer.add_scalar('%s/thrs_acc'%desc, thrs, tot_iter)                            
    writer.add_scalar('%s/auc'%desc, auc, tot_iter)
    writer.add_scalar('%s/precision_orig'%desc, metrics.roc.ap_0, tot_iter)
    writer.add_scalar('%s/precision_manip'%desc, metrics.roc.ap_1, tot_iter)
    if old_metrics:
        fpr_values = [0.1]
    else:
        fpr_values = [0.1,0.01]    
    for fpr_value in fpr_values:
        t_auc = metrics.roc.get_trunc_auc(fpr_value)
        t_auc_proba = metrics.roc.get_trunc_auc_proba(fpr_value)
        tpr_fpr, score_for_tpr_fpr = metrics.roc.get_tpr_at_fpr(fpr_value)
        if old_metrics:
            writer.add_scalar('%s/tauc'%(desc), t_auc, tot_iter)
            writer.add_scalar('%s/tpr_fpr'%(desc), tpr_fpr, tot_iter)
            writer.add_scalar('%s/score_tpr_fpr'%(desc), score_for_tpr_fpr, tot_iter)
        else:
            writer.add_scalar('%s/tauc_%.0f'%(desc,(fpr_value*100.0)), t_auc, tot_iter)
            writer.add_scalar('%s/tauc_proba_%.0f'%(desc,(fpr_value*100.0)), t_auc_proba, tot_iter)
            writer.add_scalar('%s/tpr_fpr_%.0f'%(desc,(fpr_value*100.0)), tpr_fpr, tot_iter)
    
def train_logging(
                string, writer, logger, epoch, saver, 
                tot_iter, loss, loss_m, loss_ce, lr_scheduler
                ):
    _, hours, mins = saver.check_time()
    logger.info("[Epoch %d] | h:%d m:%d | iteration: %d, loss: %f, loss_m: %f, loss_ce: %f", epoch, hours, 
                mins, tot_iter, loss, loss_m, loss_ce)    
    writer.add_scalar(string, loss, tot_iter)

def get_lr_blocks(lr_basic=2e-05,gamma=2.0):
    ## These values specify the indexing for Densenet 121
    ## it will access the first conv block, then each DenseBlock+Transition.
    ## In total we have 5 blocks (the classification layer is outside of this)
    ## Note: this is model specific. We might need a dictionary with the model name
    ## if we want to do it model agnostic
    idx_blocks = [[0,4],[4,6],[6,8],[8,10],[10,12]]
    lr_list = [None]*(len(idx_blocks)+1)
    for count, l in enumerate(reversed(range(len(idx_blocks)+1))):
        scale_factor = 1/(gamma**l)
        lr_list[count] = 0.5*lr_basic*scale_factor
    lr_list.append(lr_basic)
    print('lr rate for each densenet block:')
    print(lr_list)
    return idx_blocks, lr_list

def associate_param_with_lr(model_lp,idx_blocks,lr_list,
                            offset=6,lp_lr_multiplier=1.0):
    count = 0
    params_dict_list = []
    if torch.cuda.device_count() > 1:
        ## LP branch ########################################################
        ## Optimizing fast the LP branch
        params_dict_list.append({'params' : model_lp.module.lp_branch.parameters(), 'lr' : lr_list[-1]*lp_lr_multiplier})
        print('******* lr block %d, [laplacian] c_lr: %.10f'% (count,lr_list[-1])) 
        print(model_lp.module.lp_branch)
        ## Optimizing fast merging of features
        params_dict_list.append({'params' : model_lp.module.conv_1x1_merge.parameters(), 'lr' : lr_list[-1]*lp_lr_multiplier})
        print('******* lr block %d, [conv_1x1_merge] c_lr: %.10f'% (count,lr_list[-1])) 
        print(model_lp.module.conv_1x1_merge)
        ## RGB branch ########################################################
        ## Optimizing 1st conv layer very very slowly
        params_dict_list.append({'params' : model_lp.module.rgb_branch[0][:4].parameters(), 'lr' : lr_list[0]})
        print('******* lr block %d, [rgb_conv] c_lr: %.10f'% (count,lr_list[0])) 
        print(model_lp.module.rgb_branch[0][:4])
        ## Optimizing 1st densnet block very slowly
        params_dict_list.append({'params' : model_lp.module.rgb_branch[0][4:6].parameters(), 'lr' : lr_list[1]})
        print('******* lr block %d, [rgb_dense_block] c_lr: %.10f'% (count,lr_list[1])) 
        print(model_lp.module.rgb_branch[0][4:6])
        ## Now deceding the optimizer in the backbone
        mod_feat = model_lp.module.backbone
        for count, (idx, c_lr) in enumerate(zip(idx_blocks[2:],lr_list[2:])):
            print('******* lr block %d, [%d,%d] c_lr: %.10f'% (count,idx[0],idx[1],c_lr))    
            sliced_model = mod_feat[idx[0]-offset:idx[-1]-offset]                    
            print(sliced_model)
            param_dict = {'params' : sliced_model.parameters(), 'lr' : c_lr}
            params_dict_list.append(param_dict)
        ## Adding the flatten  
        c_lr = lr_list[-1]
        print('******* lr block %d, [%s] c_lr: %.10f'% (count+1,'flatten',c_lr)) 
        print(model_lp.module.flatten)       
        params_dict_list.append({'params' : model_lp.module.flatten.parameters(), 'lr' : c_lr})

        ## Finally adding the RNN
        c_lr = lr_list[-1]
        print('******* lr block %d, [%s] c_lr: %.10f'% (count+2,'RNN',c_lr)) 
        print(model_lp.module.rnn)       
        params_dict_list.append({'params' : model_lp.module.rnn.parameters(), 'lr' : c_lr})
    else:
        ## LP branch ########################################################
        ## Optimizing fast the LP branch
        params_dict_list.append({'params' : model_lp.lp_branch.parameters(), 'lr' : lr_list[-1]*lp_lr_multiplier})
        print('******* lr block %d, [laplacian] c_lr: %.10f'% (count,lr_list[-1])) 
        print(model_lp.lp_branch)
        ## Optimizing fast merging of features
        params_dict_list.append({'params' : model_lp.conv_1x1_merge.parameters(), 'lr' : lr_list[-1]*lp_lr_multiplier})
        print('******* lr block %d, [conv_1x1_merge] c_lr: %.10f'% (count,lr_list[-1])) 
        print(model_lp.conv_1x1_merge)
        ## RGB branch ########################################################
        ## Optimizing 1st conv layer very very slowly
        params_dict_list.append({'params' : model_lp.rgb_branch[0][:4].parameters(), 'lr' : lr_list[0]})
        print('******* lr block %d, [rgb_conv] c_lr: %.10f'% (count,lr_list[0])) 
        print(model_lp.rgb_branch[0][:4])
        ## Optimizing 1st densnet block very slowly
        params_dict_list.append({'params' : model_lp.rgb_branch[0][4:6].parameters(), 'lr' : lr_list[1]})
        print('******* lr block %d, [rgb_dense_block] c_lr: %.10f'% (count,lr_list[1])) 
        print(model_lp.rgb_branch[0][4:6])
        ## Now deceding the optimizer in the backbone
        mod_feat = model_lp.backbone
        for count, (idx, c_lr) in enumerate(zip(idx_blocks[2:],lr_list[2:])):
            print('******* lr block %d, [%d,%d] c_lr: %.10f'% (count,idx[0],idx[1],c_lr))    
            sliced_model = mod_feat[idx[0]-offset:idx[-1]-offset]                    
            print(sliced_model)
            param_dict = {'params' : sliced_model.parameters(), 'lr' : c_lr}
            params_dict_list.append(param_dict)
        ## Adding the flatten  
        c_lr = lr_list[-1]
        print('******* lr block %d, [%s] c_lr: %.10f'% (count+1,'flatten',c_lr)) 
        print(model_lp.flatten)       
        params_dict_list.append({'params' : model_lp.flatten.parameters(), 'lr' : c_lr})

        ## Finally adding the RNN
        c_lr = lr_list[-1]
        print('******* lr block %d, [%s] c_lr: %.10f'% (count+2,'RNN',c_lr)) 
        print(model_lp.rnn)       
        params_dict_list.append({'params' : model_lp.rnn.parameters(), 'lr' : c_lr})
    return params_dict_list

class lrSched_monitor(object):
    """
    This class is used to monitor the learning rate scheduler's behavior
    during training. If the learning rate decreases then this class re-initializes
    the last best state of the model and starts training from that point of time.
    
    Parameters
    ----------
    model : torch model
    scheduler : learning rate scheduler object from training
    data_config : this object holds model_path and model_name, used to load the last best model.
    """
    def __init__(self, model, scheduler, data_config):
        self.model = model
        self.scheduler = scheduler
        self.model_name = data_config.model_name
        self.model_path = data_config.model_path
        self._last_lr = [0]*len(scheduler.optimizer.param_groups)
        self.prev_lr_mean = self.get_lr_mean()
    
    ## Get the current mean learning rate from the optimizer
    def get_lr_mean(self):
        lr_mean = 0
        for i, grp in enumerate(self.scheduler.optimizer.param_groups):
            if 'lr' in grp.keys():
                lr_mean += grp['lr']
                self._last_lr[i] = grp['lr']
        return lr_mean/(i+1)       
        
    ## This is the function that is to be called right after lr_scheduler.step(val_loss)    
    def monitor(self):
        ## When self.num_bad_epochs > self.patience, lr will be decreased
        if self.scheduler.num_bad_epochs == self.scheduler.patience:
            self.prev_lr_mean = self.get_lr_mean()      ## to keep the best lr in the last time
        elif self.get_lr_mean() < self.prev_lr_mean:    ## this means scheduler/ReduceLROnPlateau effects, please load the last best model
            self.load_best_model()        ## lr is reduced one, but the rest is the last best one.
            self.prev_lr_mean = self.get_lr_mean()
    
    ## This function loads the last best model once the learning rate decreases
    def load_best_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            ckpt = torch.load(os.path.join(self.model_path,'best_model.pth'))
            self.model.load_state_dict(ckpt['model_state_dict'], strict=True)
            self.scheduler.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print(f'Loading the best model from {self.model_path}')
            if device.type == 'cpu':
                ckpt = torch.load(os.path.join(self.model_path,'best_model.pth'), map_location='cpu')
            else:
                ckpt = torch.load(os.path.join(self.model_path,'best_model.pth'))
            ## Model State Dict
            state_dict = ckpt['model_state_dict']
            ## Since the model files are saved on dataparallel we use the below hack to load the weights on a model in cpu or a model on single gpu.
            keys = state_dict.keys()
            values = state_dict.values()
            new_keys = []
            for key in keys:
                new_key = key.replace('module.','')    # remove the 'module.'
                new_keys.append(new_key)

            new_state_dict = OrderedDict(list(zip(new_keys, values))) # create a new OrderedDict with (key, value) pairs
            self.model.load_state_dict(new_state_dict, strict=True)
            
            ## Optimizer State Dict
            optim_state_dict = ckpt['optimizer_state_dict']
            # Since the model files are saved on dataparallel we use the below hack to load the optimizer state in cpu or a model on single gpu.
            keys = optim_state_dict.keys()
            values = optim_state_dict.values()
            new_keys = []
            for key in keys:
                new_key = key.replace('module.','')    # remove the 'module.'
                new_keys.append(new_key)

            new_optim_state_dict = OrderedDict(list(zip(new_keys, values))) # create a new OrderedDict with (key, value) pairs
            self.scheduler.optimizer.load_state_dict(new_optim_state_dict)
        
        ## Reduce the learning rate
        for i, grp in enumerate(self.scheduler.optimizer.param_groups):
            grp['lr'] = self._last_lr[i]
            # self._last_lr[i] is the new one, decreased by ReduceLROnPlateau