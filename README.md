# FR-Train: A Mutual Information-Based Approach to Fair and Robust Training

#### Authors: Yuji Roh, Kangwook Lee, Steven Euijong Whang, and Changho Suh
#### In Proceedings of the 37th International Conference on Machine Learning (ICML), 2020

This directory is for simulating FR-Train 
[https://arxiv.org/abs/2002.10234, ICML 2020] on synthetic dataset.
The program needs PyTorch and Jupyter Notebook.

The directory contains total 8 files: 1 README, 1 python file, 
2 jupyter notebooks, and 4 data files (3 numpy files for synthetic data, 
1 text file for poisoning index)

To simulate FR-Train, please use the jupyter notebooks in the directory.
FRTrain_clean.ipynb and FRTrain_poisoned.ipynb contain clean mode and
poisoned mode, respectively.

The jupyter notebooks will load the data and put the arranged dataset 
into train_model(). The variable 'y_train' contains different data 
depending on whether it is a clean or poisoned mode.

The train_model() will train FR-Train by using the classes in FRTrain_arch.py.
After the training, train_model() will return the test accuracy and
disparate impact to the caller.

The python file is FRTrain_arch.py contains the defined structures 
of FR-Train: generator and two discriminators (for fairness and 
robustness each). 

The detailed explanations about each component have been written 
in the codes as comments.
Thanks!
