#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-

import os
import numpy as np
from numba import jit
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")


class feature_extract(Dataset):
    def __init__(self, nucle_name, result_dir, win_size_1D=25, win_size_min_2D=9,
                 win_size_max_2D=31):
        super(feature_extract, self).__init__()
        self.nucle_name = nucle_name
        self.fa_path = os.path.join(result_dir, nucle_name + ".fasta")
        self.seq = np.loadtxt(self.fa_path, dtype=str)[1]
        self.result_dir = result_dir
        self.psfm_path = os.path.join(self.result_dir, self.nucle_name + ".psfm")
        self.ss_path = os.path.join(self.result_dir, self.nucle_name + ".ss")
        self.one_hot_path = os.path.join(self.result_dir, self.nucle_name)

        self.stride_1D = int(win_size_1D / 2)
        self.win_size_1D = win_size_1D
        self.stride_min_2D = int(win_size_min_2D / 2)
        self.win_size_min_2D = win_size_min_2D
        self.stride_max_2D = int(win_size_max_2D / 2)
        self.win_size_max_2D = win_size_max_2D
        self.dataInfo = self.feas_labs()

    def __len__(self):
        return len(self.dataInfo)

    @jit()
    def fusion_feature(self):

        one_hot = np.loadtxt(self.one_hot_path, dtype=float)
        psfm = np.loadtxt(self.psfm_path, dtype=float)
        prss = np.expand_dims(np.loadtxt(self.ss_path, dtype=float), 1)

        one_hot_psfm = np.append(one_hot, psfm, axis=1)
        one_hot_psfm_prss = np.append(one_hot_psfm, prss, axis=1)

        paddingheader = one_hot_psfm_prss[:self.stride_1D, :]
        paddingfooter = one_hot_psfm_prss[-self.stride_1D:, :]
        one_hot_psfm_prss = np.append(paddingheader, one_hot_psfm_prss, axis=0)
        one_hot_psfm_prss = np.append(one_hot_psfm_prss, paddingfooter, axis=0)

        return one_hot_psfm_prss

    @jit()
    def build3D_feature(self):
        one_hot_psfm_prss, lab = self.fusion_feature()
        nucle_length, one_hot_psfm_prss_dim = one_hot_psfm_prss.shape
        nucle_length = int(nucle_length - int(self.stride_1D * 2))
        one_hot_psfm_prss_3D = np.zeros((nucle_length, self.win_size_1D, one_hot_psfm_prss_dim))

        for i in range(self.stride_1D, nucle_length + self.stride_1D):
            one_hot_psfm_prss_3D[i - self.stride_1D, :, :] = one_hot_psfm_prss[
                                                             i - self.stride_1D:i + self.stride_1D + 1, :]

        paddingheader = one_hot_psfm_prss_3D[:self.stride_max_2D, :, :]
        paddingfooter = one_hot_psfm_prss_3D[-self.stride_max_2D:, :, :]
        one_hot_psfm_prss_3D = np.append(paddingheader, one_hot_psfm_prss_3D, axis=0)
        one_hot_psfm_prss_3D = np.append(one_hot_psfm_prss_3D, paddingfooter, axis=0)

        return one_hot_psfm_prss_3D

    @jit()
    def feas_labs(self):

        self.feas_labs = []

        one_hot_psfm_prss_3D, lab = self.build3D_feature()
        nucle_length = one_hot_psfm_prss_3D.shape[0]
        for i in range(self.stride_max_2D, nucle_length - self.stride_max_2D):
            fea_min_1D = one_hot_psfm_prss_3D[i, :, :]
            fea_min_2D = one_hot_psfm_prss_3D[i - self.stride_min_2D:i + self.stride_min_2D + 1, :, :]
            fea_max_2D = one_hot_psfm_prss_3D[i - self.stride_max_2D:i + self.stride_max_2D + 1, :, :]
            label = lab[i - self.stride_max_2D]

            dim_0, dim_1, dim_2 = fea_min_2D.shape
            sides = int(np.sqrt(dim_0))
            fea_max_1D = np.zeros((int(sides * dim_1), int(sides * dim_2)))
            for i in range(sides):
                for j in range(sides):
                    fea_max_1D[i * dim_1:(i + 1) * dim_1, j * dim_2:(j + 1) * dim_2] = fea_min_2D[j + (i * sides),
                                                                                       :, :]
            fea_min_1D = np.expand_dims(fea_min_1D, axis=0)
            fea_max_1D = np.expand_dims(fea_max_1D, axis=0)

            self.feas_labs.append((fea_min_1D, fea_max_1D, fea_min_2D, fea_max_2D))

        return self.feas_labs
