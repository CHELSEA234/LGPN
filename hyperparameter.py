## Hyper-params #######################
hparams = {
		'epochs': 500, 'batch_size': 64, 'basic_lr': 0.01, 'fine_tune': True, 'use_laplacian': True, 'step_factor': 0.1, 
		'patience': 5, 'weight_decay': 1e-06, 'lr_gamma': 2.0, 'use_magic_loss': True, 'feat_dim': 1024, 'drop_rate': 0.2, 
		'skip_valid': False, 'rnn_type': 'LSTM', 'rnn_hidden_size': 128, 'num_rnn_layers': 1, 'rnn_drop_rate': 0.2, 
		'bidir': True, 'merge_mode': 'concat', 'perc_margin_1': 0.95, 'perc_margin_2': 0.95, 'soft_boundary': False, 
		'dist_p': 2, 'radius_param': 0.84, 'strat_sampling': True, 'normalize': True, 'window_size': 10, 'hop': 1, 
		'valid_epoch': 100, 'use_sched_monitor': True
		}
batch_size = hparams['batch_size']
basic_lr = hparams['basic_lr']
fine_tune = hparams['fine_tune']
use_laplacian = hparams['use_laplacian']
step_factor = hparams['step_factor']
patience = hparams['patience']
weight_decay = hparams['weight_decay']
lr_gamma = hparams['lr_gamma']
use_magic_loss = hparams['use_magic_loss']
feat_dim = hparams['feat_dim']
drop_rate = hparams['drop_rate']
rnn_type = hparams['rnn_type']
rnn_hidden_size = hparams['rnn_hidden_size']
num_rnn_layers = hparams['num_rnn_layers']
rnn_drop_rate = hparams['rnn_drop_rate']
bidir = hparams['bidir']
merge_mode = hparams['merge_mode']
perc_margin_1 = hparams['perc_margin_1']
perc_margin_2 = hparams['perc_margin_2']
dist_p = hparams['dist_p']
radius_param = hparams['radius_param']
strat_sampling = hparams['strat_sampling']
normalize = hparams['normalize']
window_size = hparams['window_size']
hop = hparams['hop']
soft_boundary = hparams['soft_boundary']
use_sched_monitor = hparams['use_sched_monitor']
########################################