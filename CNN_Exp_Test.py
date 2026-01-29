
from CNN_test import CNN_exp_test

exp_npz_20220510='./Dataset_Filted/Experiment/2022/0510/combined_Exp_20220510_1e9_V01_dataset.npz'
exp_npz_20241231='./Dataset_Filted/Experiment/2024/1231/combined_Exp_20241231_1e9_V01_dataset.npz'

CNN_exp_test(exp_npz_20220510,'CNN2_ExpTest_20220510')
CNN_exp_test(exp_npz_20241231,'CNN2_ExpTest_20241231')

