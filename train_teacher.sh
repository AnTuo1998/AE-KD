CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
    train_teacher.py \
    --model ResNet50 \
    --dataset cifar100 \
    --nesterov \
    --epochs 240 \
    --seed $1 \
    --trial 1