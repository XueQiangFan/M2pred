# M2pred
Predicting RNA solvent accessibility from multi-scale context feature via multi-shot neural network

## Pre-requisite:  
   - Linux system
   - [Python3.7](https://docs.python-guide.org/starting/install3/linux/)
   - [pytorch version 1.3.1](https://pytorch.org/)
   - [infernal-1.1.3](http://eddylab.org/infernal/infernal-1.1.3.tar.gz)
   - [ViennaRNA-2.4.18](https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_4_x/ViennaRNA-2.4.18.tar.gz)
   - [LinearPartition](https://github.com/LinearFold/LinearPartition.git/)
   - [nt](https://ftp.ncbi.nih.gov/blast/db/)  
    

## Installation:

*Install and configure the softwares of python3.7, Pytorch, Infernal, RNAfold, LinearPartition, and nt in your Linux system. Please make sure that python3 includes the modules of 'os', 'math', 'numpy', 'configparser', 'numba', 'random', 'subprocess', 'sys', and 'shutil'. If any one modules does not exist, please using 'pip install xxx' command install the python revelant module. Here, "xxx" is one module name.

*Download this repository at https://github.com/XueQiangFan/M2pred. Then, uncompress it and run the following command lines on Linux System.

~~~
  $ jar xvf M2pred-main.zip
  $ chmod -R 777 ./I-RNAsol-main.zip
  $ cd ./M2pred-main
  $ unzip save_model.zip 
~~~
Here, you will see one configuration files.   
*Configure the following tools or databases in M2pred.config  
  The file of "M2pred.config" should be set as follows:
- Infernal
- LinearPartition
- RNAfold
- nt
~~~
  For example:  
  [Infernal]
  Infernal = /iobio/fxq/software/infernal-1.1.3-linux-intel-gcc/binaries
  cmsearch_DB = /iobio/fxq/library/database/nt.fa/nt
  [RNAfold]
  RNAfold_EXE = /iobio/fxq/software/ViennaRNA-2.4.17/bin/RNAfold
  [LinearPartition]
  LinearPartition_EXE = /iobio/fxq/software/LinearPartition-master/linearpartition
~~~
Note: Make sure there is enough space on the system as NCBI's nt database is of size around 333 GB after extraction and it can take couple of hours to download depending on the internet speed. In case of any issue, please rerfer to https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download

Either follow **conda** column steps to create virtual environment and to install M2pred dependencies given in table below:<br />

|  | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; conda |
| :- | :--- |
| 1. |  `conda create -n venv python=3.7` |
| 2. |  `conda activate venv` | 
| 3. |  *To run M2pred on CPU:*<br /> <br /> `conda install pytorch torchvision torchaudio cpu only -c pytorch` <br /> <br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  |
| 4. | `while read p; do conda install --yes $p; done < requirements.txt` | 

## Run M2pred 
### run: python main.py -n RNA name -s RNA sequence -o result path
~~~
    For example:
    python main.py -n 1f1t_A -s GGACCCGACGGCGAGAGCCAGGAACGAAGGACC -o ./
~~~

## The RNA solvent accessibility result

*The RNA solvent accessibility result of each rsidue should be found in the outputted file, i.e., " RNA name +.rsa". In each result file, where "NO" is the position of each residue in your RNA, where "AA" is the name of each residue in your RNA, where "RSA" is the predicted relative accessible surface area of each residue in your RNA, and where "ASA" is the predicted accessible surface area of each nucleotide in your RNA.

## Update History:

First release 2022-05-27

## References

[1] Xue-Qiang Fan, Jun Hu*, Yu-Xuan Tang, Ning-Xin Jia, Dong-Jun Yu*, and Gui-Jun Zhang*. Predicting RNA solvent accessibility from multi-scale context feature via multi-shot neural network.
