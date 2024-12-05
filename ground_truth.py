"""
	generate ground truth for 55 hyperparatmers.
"""
from glob import glob
from operator import add

import os
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

ADJ_GEN = True
VIZ_FLAG  = False
THRESHOLD = 0.4
GM_NUM = 140
Dis_NET_NUM = 18
Con_NET_NUM = 27
OBJ_FUN_NUM = 10

ground_truth_text_dir = './'

set_1 = ["ADV_FACES", "BETA_B", "BETA_TCVAE", "BIGGAN_128", "DAGAN_C", "DRGAN", "FGAN", 
			"PIXELCNN", "PIXELCNN_PP", "RSGAN_HALF", "STARGAN", "VAE_GAN", "DDPM_256", 
			"IDDPM_64", "DDiFFGAN_32"]
set_2 = ["AAE", "ADAGAN_C", "BEGAN", "BETA_H", "BIGGAN_256", "COCO_GAN", "CRAMER_GAN", 
			"DEEPFOOL", "DRIT", "FASTPIXEL", "FVBN", "SRFLOW", "ADM_G_64", "DDPM_32",
			"GLIDE"]
set_3 = ["BICYCLE_GAN", "BIGGAN_512", "CRGAN_C", "FACTOR_VAE", "FGSM", "ICRGAN_C", "LOGAN", 
			"MUNIT", "PIXELSNAIL", "STARGAN_2", "SURVAE_FLOW_MAXPOOL", "VAE_FIELD",
			"LDM", "CONTROLNET", "STABLE_DM_15"]
set_4 = ["GFLM", "IMAGE_GPT", "LSGAN", "MADE", "PIX2PIX", "PROG_GAN", "RSGAN_REG", "SEAN", 
			"STYLEGAN", "SURVAE_FLOW_NONPOOL", "WGAN_DRA", "YLG",
			"SDEDIT", "ADM_G_128", "STABLE_DM_XL"]

def viz_func(input_list):
	'''
		visualize the correlation matrix.
	'''
	array = input_list
	total_length = len(input_list)
	df_cm = pd.DataFrame(
						array, 
						index = [i for i in range(total_length)],
	                  	columns = [i for i in range(total_length)]
	                  	)
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True, cmap="crest")
	plt.show()

def list_return(obj_list, archi_list, idx):
	'''
	return:
		res_list: the combine list, which perserve the order of hyper-parameters.
		res_flag: indicates whether the order has been maintained.
	'''
	res_list = obj_list[idx] + archi_list[idx]
	res_flag = (len(obj_list[idx]) == 10 and len(archi_list[idx]) == 21)
	return res_list, res_flag

# step 0: loading images from the train directory.
folder_complete_tmp  = glob(f"{ground_truth_text_dir}/*") ## it has 127 GMs here.
folder_complete_list = []
for _ in folder_complete_tmp:
	GM_name = _.split('/')[-1]
	folder_complete_list.append(GM_name)
folder_lst = []

## step 1: loading the architecture parameters.
archi_fun_list = os.path.join(ground_truth_text_dir, 'copy_list_archi.txt')
f = open(archi_fun_list, "r")
lines = f.readlines()
label_archi_list = []
for idx, _ in enumerate(lines):
	cur_list = []
	line = _.strip()
	par_lst = line.split(' ')
	GM_name = par_lst[0]
	for par in par_lst:
		if par.isdigit(): ## in case STARGAN 2, STYLEGAN 2
			par_val = int(par)
			cur_list.append(par_val)
	folder_lst.append(GM_name)
	label_archi_list.append(cur_list)
assert len(label_archi_list) == GM_NUM
assert len(label_archi_list[0]) == 15
f.close()

## step 2: refine the architecture parameters.
## do not have F2-F5, divide others into 3 blocks.
refine_archi_list = []
max_value_list = [717,289,185,46,235,8365,9,16,94008488]
for cur_list in label_archi_list:
	refine_cur_list = []
	cur_list = cur_list[-15:]
	for idx, ele in enumerate(cur_list):
		# if idx in [1,2,3,4]:
		# 	# refine_archi_list.append()
		# 	continue
		if idx == 0:	# layer
			if ele < 300: 
				# tmp = [float(ele/30),0,0]
				tmp = [1,0,0]
			elif 300 <= ele and ele < 600:
				# tmp = [0,float((ele-30)/30),0]
				tmp = [0,1,0]
			else:
				# tmp = [0,0,float((ele-60)/(97-60))]
				tmp = [0,0,1]
		elif idx == 1:	# conv layer.
			if ele < 100:
				tmp = [1,0,0]
			elif 100 <= ele and ele < 200:
				tmp = [0,1,0]
			else:
				tmp = [0,0,1]
		elif idx == 2:	# linear layer.
			if ele < 60:
				tmp = [1,0,0]
			elif 60 <= ele and ele < 120:
				tmp = [0,1,0]
			else:
				tmp = [0,0,1]
		elif idx == 3:	# pool layer.
			if ele < 10:
				tmp = [1,0,0]
			elif 10 <= ele and ele < 30:
				tmp = [0,1,0]
			else:
				tmp = [0,0,1]
		elif idx == 4:	# norm layer.
			if ele < 80:
				tmp = [1,0,0]
			elif 80 <= ele and ele < 160:
				tmp = [0,1,0]
			else:
				tmp = [0,0,1]
		elif idx == 5: # num of filter
			if ele < 3000:
				tmp = [1,0,0]
			elif 3000 <= ele and ele < 6000:
				tmp = [0,1,0]
			else:
				tmp = [0,0,1]
		elif idx == 6:	# num of block
			if ele < 3:
				tmp = [1,0,0]
			elif 3 <= ele and ele < 6:
				tmp = [0,1,0]
			else:
				tmp = [0,0,1]
		elif idx == 7:	# layer per block
			if ele < 5:
				tmp = [1,0,0]
			elif 5 <= ele and ele < 10:
				tmp = [0,1,0]
			else:
				tmp = [0,0,1]
		elif idx == 8:	# para. num.
			if ele < 10000000:
				tmp = [1,0,0]
			elif 10000000 <= ele and ele < 60000000:
				# tmp = [0,float((ele-10000000)/50000000),0]
				tmp = [0,1,0]
			else:
				tmp = [0,0,1]
		elif idx == 9:	# normalization
			if ele == 0:
				tmp = [1,0,0,0]
			elif ele == 1:
				tmp = [0,1,0,0]
			elif ele == 2:
				tmp = [0,0,1,0]
			elif ele == 3:	# this is due to the inconsistency between the new and old definition tables.
				tmp = [0,0,0,0]
			elif ele == 4:
				tmp = [0,0,0,1]
		elif idx == 10:	# last layer nonlinear
			if ele == 0:
				tmp = [1,0,0,0,0]
			elif ele == 1:
				tmp = [0,1,0,0,0]
			elif ele == 2:
				tmp = [0,0,1,0,0]
			elif ele == 3:	
				tmp = [0,0,0,1,0]
			elif ele == 4:
				tmp = [0,0,0,0,1]
			elif ele == 5:
				tmp = [0,0,0,0,0]
		elif idx == 11:	# last block nonlinear
			if ele == 0:
				tmp = [1,0,0,0,0]
			elif ele == 1:
				tmp = [0,1,0,0,0]
			elif ele == 2:
				tmp = [0,0,1,0,0]
			elif ele == 3:	
				tmp = [0,0,0,1,0]
			elif ele == 4:
				tmp = [0,0,0,0,1]
			elif ele == 5:
				tmp = [0,0,0,0,0]
			else:
				raise ValueError
		elif idx == 12: # up-sampling
			if ele == 0:
				tmp = [1,0]
			elif ele == 1:
				tmp = [0,1]
			else:
				raise ValueError
		elif idx == 13:
			if ele == 0:
				tmp = [0]
			else:
				tmp = [1]
		elif idx == 14:
			if ele == 0:
				tmp = [0]
			else:
				tmp = [1]
		refine_cur_list.extend(tmp)
	assert len(refine_cur_list) == (Con_NET_NUM + Dis_NET_NUM)
	refine_archi_list.append(refine_cur_list)

## step 3: loading the objective functions.
object_fun_list = os.path.join(ground_truth_text_dir, "copy_list.txt")
f = open(object_fun_list, "r")
lines = f.readlines()
label_obj_list = []
for idx, _ in enumerate(lines):
	cur_list = []
	line = _.strip()
	line = line.replace(" ", "")
	for char in line:
		cur_list.append(int(char))
	label_obj_list.append(cur_list)
assert len(label_obj_list) == GM_NUM
assert len(label_obj_list[0]) == OBJ_FUN_NUM
f.close()

## step 5: making the ground truth file.
label_dict = dict()
for idx, GM_name in enumerate(folder_lst):
	final_label_list, _ = list_return(label_obj_list, refine_archi_list, idx)
	assert len(label_obj_list[idx]) == OBJ_FUN_NUM
	assert len(refine_archi_list[idx]) == (Con_NET_NUM + Dis_NET_NUM), len(refine_archi_list[idx])
	GM_name = GM_name.split('/')[-1]
	label_dict[GM_name] = final_label_list
	# import sys;sys.exit(0)