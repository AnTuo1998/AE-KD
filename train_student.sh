CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
    train_student.py \
    --distribution \
    --master_port 29534 \
    --model_s ResNet18 \
    --distill kd \
    --dataset cifar100 \
    --nesterov \
    --epochs 240 \
    --teacher_num 3 \
    --trial 0 \
    --ensemble_method AEKD