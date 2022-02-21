#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：FWorks -> main
@IDE    ：PyCharm
@Author  : Xue-Qiang Fan
@Date   ：2022/2/21 13:10
=================================================='''
import os
import numpy as np
import torch
from bin.multi_scale_context_feature_extract import feature_extract
from MSNN import MSNN
from bin.WriteFile import appendWrite
from torch.utils.data import DataLoader
from bin.feature_generate import FeaturesGeneration
import warnings

warnings.filterwarnings('ignore')


def MAXASAValue(residue_type):
    nucle_ASA = {"A": 400,
                 "C": 350,
                 "G": 400,
                 "U": 350}
    return nucle_ASA[residue_type]


def tester(nucle_name, result_dir):
    fa_path = os.path.join(result_dir, nucle_name+".fasta")
    save_model = "./saved_model/"
    model = MSNN()
    saved_model = save_model + 'MSNN'
    model.load_state_dict(torch.load(saved_model, map_location="cpu"))
    optimizer = torch.optim.Adam(model.parameters())
    saved_model = save_model + 'MSNNopt'
    optimizer.load_state_dict(torch.load(saved_model, map_location="cpu"))

    model.eval()
    with torch.no_grad():
        Data = feature_extract(nucle_name, result_dir)
        batch_size = Data.__len__()
        test_loader = DataLoader(dataset=Data, batch_size=batch_size, shuffle=False, drop_last=False)
        for i, data in enumerate(test_loader):
            feature = data
            fea_min_1D, fea_max_1D = feature[0], feature[1]
            fea_min_2D, fea_max_2D = feature[2], feature[3]
            fea_min_1D, fea_max_1D = torch.FloatTensor(fea_min_1D.float()), torch.FloatTensor(fea_max_1D.float())
            fea_min_2D, fea_max_2D = torch.FloatTensor(fea_min_2D.float()), torch.FloatTensor(fea_max_2D.float())
            predict = model(fea_min_1D, fea_max_1D, fea_min_2D, fea_max_2D)

            seq = np.loadtxt(fa_path, dtype=str)[1]
            nucle_length = len(seq)
            filename = nucle_name + ".sa"
            file_path = os.path.join(result_dir, filename)
            if os.path.exists(file_path):
                pass
            else:
                appendWrite(file_path, '{:>4}\n\n'.format("# M2pred VFORMAT (M2pred V1.0)"))
                appendWrite(file_path, '{:>1}  {:>1}  {:>4}  {:>4}\t\n'.format("NO.", "AA", "RSA", "ASA"))
                for i in range(nucle_length):
                    index, residue, RSA = i + 1, seq[i], predict[i, 0]
                    SA = MAXASAValue(seq[i]) * predict[i, 0]
                    appendWrite(file_path, '{:>4}  {:>1}  {:>.3f}  {:>.3f}\t\n'.format(index, residue, RSA, SA))
                appendWrite(file_path, '{:>8} \t'.format("END"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M2pred Predicts RNA Solvent Accessibility")
    parser.add_argument("-n", "--nucle_name", required=True, type=str, help="nucleotide name")
    parser.add_argument("-s", "--sequence", required=True, type=str, help="AA sequence ")
    parser.add_argument("-o", "--result_path", required=True, type=str, help="save result path")
    args = parser.parse_args()
    features_generation = FeaturesGeneration(args.nucle_name, args.sequence, args.result_path)
    features_generation.One_Hot_Encoding()
    features_generation.LinearParitition_SS()
    features_generation.PSFM_generation()
    tester(args.nucle_name, args.result_path)


if __name__ == '__main__':
    main()
