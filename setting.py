cifar100_teacher_model_name = [
    'cifar100-ResNet50-0', 'cifar100-ResNet50-1', 'cifar100-ResNet50-2',
]


cifar10_teacher_model_name = [
    'cifar10-resnet56-35'
]


# ------------- teacher net --------------------#
teacher_model_path_dict = {
    "cifar100-ResNet50-0": "./save/teacher_models/cifar100/ResNet50_cifar100_lr_0.05_decay_0.0005_trial_1_nesterov_True_step_150-180-210_bs_64_seed_0_ep_240/ResNet50_best.pth",
    "cifar100-ResNet50-1": "./save/teacher_models/cifar100/ResNet50_cifar100_lr_0.05_decay_0.0005_trial_1_nesterov_True_step_150-180-210_bs_64_seed_4_ep_240/ResNet50_best.pth",
    "cifar100-ResNet50-2": "./save/teacher_models/cifar100/ResNet50_cifar100_lr_0.05_decay_0.0005_trial_1_nesterov_True_step_150-180-210_bs_64_seed_9_ep_240/ResNet50_best.pth",
}
