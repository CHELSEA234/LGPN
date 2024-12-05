source ~/.bashrc
conda activate RED
CUDA_NUM=5
CUDA_VISIBLE_DEVICES=$CUDA_NUM python main.py --cross_val=1 --batch_size=256 --protocol=Non --lr=0.1
CUDA_VISIBLE_DEVICES=$CUDA_NUM python main.py --cross_val=2 --batch_size=256 --protocol=Non --lr=0.1
CUDA_VISIBLE_DEVICES=$CUDA_NUM python main.py --cross_val=3 --batch_size=256 --protocol=Non --lr=0.1
CUDA_VISIBLE_DEVICES=$CUDA_NUM python main.py --cross_val=4 --batch_size=256 --protocol=Non --lr=0.1