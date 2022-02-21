#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：RNASolventAccessibility -> processing_msa_to_psfm
@IDE    ：PyCharm
@Date   ：2021/5/9 15:57
=================================================='''
import numpy as np
import re
from numba import jit

np.set_printoptions(threshold=np.inf)


class Processing_Aln_To_PSFM():
    def __init__(self, aln_path):
        self.aln_path = aln_path

    @jit
    def __nucle_dict__(self, nucle_type):
        nucle_numeric = {"A": 0,
                         "C": 1,
                         "G": 2,
                         "U": 3,
                         "-": 4,
                         "W": 4,
                         "M": 4,
                         ">": 4,
                         "B": 4,
                         "D": 4,
                         "V": 4,
                         "H": 4,
                         "N": 4,
                         }
        return nucle_numeric[nucle_type]

    # @jit
    def transform_Aln_to_numeric(self):

        with open(self.aln_path, "r") as fr:
            aln = fr.readlines()
        N = len(aln)
        L = len(aln[0].strip())
        numeric_msa = np.zeros([N, L], dtype=int)

        for i in range(N):
            aline = aln[i].strip()
            for j in range(int(L)):
                numeric_msa[i, j] = self.__nucle_dict__(aline[j])

        return numeric_msa

    @jit
    def transform_numeric_MSA_to_PSFM(self):
        numeric_msa = self.transform_Aln_to_numeric()
        N, nucle_length = numeric_msa.shape
        query = numeric_msa[0, :]
        PSFM = np.zeros((nucle_length, 4))
        for i in range(nucle_length):
            for j in range(0, 4):
                total = 0
                for k in range(N):
                    if j == numeric_msa[k, i]: total += 1
                # PSFM[i, j] = total
                if j == query[i]:
                    PSFM[i, j] = total + 9
                else:
                    PSFM[i, j] = total + 0.3
            fenmu = PSFM[i, 0] + PSFM[i, 1] + PSFM[i, 2] + PSFM[i, 3]
            for l in range(0, 4):
                PSFM[i, l] = np.round(PSFM[i, l] / (fenmu), 3)

        return PSFM


class Processing_MSA_To_PSFM():
    def __init__(self, query_path, msa_path, aln_path):
        self.query_path = query_path
        self.msa_path = msa_path
        self.aln_path = aln_path

    @jit
    def __nucle_dict__(self, nucle_type):

        nucle_numeric = {"A": 0,
                         "C": 1,
                         "G": 2,
                         "U": 3,
                         "-": 4,
                         "W": 4,
                         "M": 4,
                         ">": 4,
                         "B": 4,
                         "D": 4,
                         "V": 4,
                         "H": 4,
                         "N": 4,
                         }
        return nucle_numeric[nucle_type]

    @jit
    def MAXASAValue(self, residue_type):
        nucle_ASA = {"A": 400,
                     "C": 350,
                     "G": 400,
                     "U": 350}
        return nucle_ASA[residue_type]

    def __readCmsearchMSA__(self):
        f = open(self.msa_path)
        if len(f.readlines()) <= 43:
            MSA = []
            query_seq = np.loadtxt(self.query_path, dtype=str)[1]
            MSA.append(query_seq)
        else:
            f.seek(0)
            line = f.readline()
            while line.strip() != "Hit alignments:":
                line = f.readline()
            line = f.readline()
            MSA = []
            query_seq = np.loadtxt(self.query_path, dtype=str)[1]
            MSA.append(query_seq)
            line = line.strip().split(" ")
            while line[0] != "Internal":
                line = f.readline()
                hit_seq = ""
                key_word = line.strip().split(" ")[0]
                while key_word != ">>":
                    line = line.strip().split(" ")
                    if line[0] == "Internal":
                        break
                    if line[-1] == "CS":
                        f.readline()
                        f.readline()
                        line = f.readline().strip().split(" ")
                        while "" in line:
                            line.remove("")
                        line = line[2]
                        line = "".join([re.sub(r'[a-z''\n]', '', x) for x in line])
                        line = "".join([re.sub(r'[NYRKS*><\[\]0-9]', '-', x) for x in line])
                        hit_seq += line
                    line = f.readline()
                    key_word = line.strip().split(" ")[0]
                print(hit_seq)
                MSA.append(hit_seq)
        f.close()
        return MSA

    # @jit
    def non_redundant_nucle_MSA(self):
        MSA = self.__readCmsearchMSA__()
        non_redundant_MSA = []
        for i in range(len(MSA)):
            if MSA[i] in non_redundant_MSA:
                pass
            elif len(MSA[i]) == len(MSA[0]):
                non_redundant_MSA.append(MSA[i])
        with open(self.aln_path, "w") as fw:
            for i in range(len(non_redundant_MSA)):
                fw.write(non_redundant_MSA[i].strip() + "\n")
        print(non_redundant_MSA)
        return non_redundant_MSA

    @jit
    def transform_non_redundant_MSA_to_numeric(self):
        non_redundant_MSA = self.non_redundant_nucle_MSA()
        N = len(non_redundant_MSA)
        L = len(non_redundant_MSA[0])
        numeric_msa = np.zeros([N, L], dtype=int)
        for i in range(N):
            aline = non_redundant_MSA[i]
            for j in range(L):
                numeric_msa[i, j] = self.__nucle_dict__(aline[j])

        return numeric_msa

    @jit
    def transform_numeric_MSA_to_PSFM(self):
        numeric_msa = self.transform_non_redundant_MSA_to_numeric()
        N, nucle_length = numeric_msa.shape
        query = numeric_msa[0, :]
        PSFM = np.zeros((nucle_length, 4))
        for i in range(nucle_length):
            for j in range(0, 4):
                total = 0
                for k in range(N):
                    if j == numeric_msa[k, i]: total += 1
                # PSFM[i, j] = total
                if j == query[i]:
                    PSFM[i, j] = total + 9
                else:
                    PSFM[i, j] = total + 0.3
            fenmu = PSFM[i, 0] + PSFM[i, 1] + PSFM[i, 2] + PSFM[i, 3]
            for l in range(0, 4):
                PSFM[i, l] = np.round(PSFM[i, l] / (fenmu), 3)

        return PSFM

# if __name__ == '__main__':
#     query_path = r"E:\RNA-PDB-20210702\cd85-blastclust30-length(30,1000)-na-bio-RNA20210702\RNAsol_no_seq\6ifn_N.fasta"
#     msa_path = r"E:\RNA Solvent accessibility\HCD-CNN-Multi-Head-Attiation\result\buffer\6ifn_N.msa"
#     b= r".\6ifn_N.aln"
#     a = Processing_MSA_To_PSFM(query_path, msa_path,b)
#     psfm = a.transform_numeric_MSA_to_PSFM()
#     np.set_printoptions(threshold=np.inf)
#     print(psfm)
