# AE-KD

This repo covers the implementation of the following NeurIPS 2020 paper:

Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space 

## Installation

This repo was tested with Python 3.5, PyTorch 1.0.0, and CUDA 9.0.

## Usage

1. Train multiple teacher models by:

    ```sh
    sh train_cifar100.sh [seed]
    ```
   where specifies a random seed.
   
   Write your teacher directory in `setting.py`.
   
2. Run distillation by following commands

    ```sh
    python3 train_student.py --distill --dataset --model_s \
    	-r -a -b -C --trial --teacher_num --ensemble_method
    ```
    where the flags are explained as:
    - `--distill`: specify the  distillation method, e.g. `kd`, `hint`
    - `--dataset`: specify the dataset, e.g. `cifar10`, `cifar100`
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `0.9`
    - `-b`: the weight of other distillation losses, default: `100`
    - `-C`: specify the tolerance parameter, default: `0.6`
    - `--trial`: specify the experimental id to differentiate between multiple runs, default: `1`
    - `--teacher_num`: specify the ensemble size (number of teacher models)
    - `--ensemble_method`: specify the ensemble_method, e.g. `AVERAGE_LOSS`,`AEKD`


Therefore, the command for running AE-KD for student model `ResNet18` is something like:

```sh
python train_multiTeacher.py --distill kd --dataset cifar100 \
	--model_s ResNet18 -r 1 -a 0.9 -b 0 \
  	--teacher_num 5 --ensemble_method AEKD
```
## Acknowledgement

This repo is built upon [Repdistiller](https://github.com/HobbitLong/RepDistiller).

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{du2020agree,
  title={Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space},
  author={Du, Shangchen and You, Shan and Li, Xiaojie and Wu, Jianlong and Wang, Fei and Qian, Chen and Zhang, Changshui},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

