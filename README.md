# Distilled Protein Backbone Generation

This repository contains code for our paper [Distilled Protein Backbone Generation](https://arxiv.org/abs/2510.03095). 

*Abstract*: Diffusion- and flow-based generative models have recently demonstrated strong performance in protein backbone generation tasks, offering unprecedented capabilities for de novo protein design. However, while achieving notable performance in generation quality, these models are limited by their generating speed, often requiring hundreds of iterative steps in the reverse-diffusion process. This computational bottleneck limits their practical utility in large-scale protein discovery, where thousands to millions of candidate structures are needed. To address this challenge, we explore the techniques of score distillation, which has shown great success in reducing the number of sampling steps in the vision domain while maintaining high generation quality. However, a straightforward adaptation of these methods results in unacceptably low designability. Through extensive study, we have identified how to appropriately adapt Score identity Distillation (SiD), a state-of-the-art score distillation strategy, to train few-step protein backbone generators which significantly reduce sampling time, while maintaining comparable performance to their pretrained teacher model. In particular, multistep generation combined with inference time noise modulation is key to the success. We demonstrate that our distilled few-step generators achieve more than a 20-fold improvement in sampling speed, while achieving similar levels of designability, diversity, and novelty as the Proteina teacher model. This reduction in inference cost enables large-scale in silico protein design, thereby bringing diffusion-based models closer to real-world protein engineering applications.  

Below are some protein structures sampled from our 16-step distilled generator. 
![](./assets/combined_grid.png)

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed the latest version of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
* You have a `Windows/Linux/Mac` machine.

## Installation

To install the necessary packages and set up the environment, follow these steps:

### Clone the repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/LY-Xie/SiD_Protein.git
cd SiD_Protein
```

### Installation

To create the Conda environment with all the required dependencies, run:

```bash
conda env create -f environment.yaml
conda activate sid_protein_env
pip install -e training/proteina
```

This command will create an environment according to the `environment.yaml` file in the repository, activate the environment, and install Proteina from their source code stored in `training/proteina`.

## Prepare pretrained model
### Create directory for checkpoints
```bash
mkdir pretrained_checkpoints
```
Make sure to update the `ckpt_path` in `training/proteina/configs/experiment_config/inference_base.yaml` to `pretrained_checkpoints`.

### Download pretrained model checkpoints
Available pretrained model checkpoints are listed in the repository of [Proteina](https://github.com/NVIDIA-Digital-Bio/proteina/tree/main). Here we provide an example with the $\mathcal{M}^{\textrm{no-tri}}_{\textrm{FS}}$ model. 
```bash
curl -L 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/proteina_v1.2_dfs_200m_notri/1.0/files?redirect=true&path=proteina_v1.2_DFS_200M_notri.ckpt' -o 'pretrained_checkpoints/proteina_v1.2_DFS_200M_notri.ckpt'
```

### Download additional files
Although SiD training is data-free for unconditional generation, conditional generation and the fold-class-based evaluation require information about the CATH codes. Therefore, we still need to set up a data directory following the instructions in [Proteina](https://github.com/NVIDIA-Digital-Bio/proteina/tree/main).

In short, first create a file `.env` in the root directory of the repository with the single line:
```bash
DATA_PATH=/directory/where/you/store/files
```
Then download [proteina_additional_files.zip](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/proteina_additional_files/files) and decompress into the `$DATA_PATH` folder. 

Eventually, the directory should look like:
```
$DATA_PATH
    - metric_factory
        - features
            - D_FS_eval_ca_features.pth
            - D_FS_afdb_cath_codes.pth
            - pdb_eval_ca_features.pth
            - fold_class_mappings_C_selected_A_T_cath_codes.pth
        - model_weights
            - gearnet_ca.pth
    - pdb_raw
        - cath_label_mapping.pt
```

### Download foldseek databases
For novelty calculations, Foldseek databases for the PDB and AFDB are used as reference. To facilitate relative paths in our evaluation scripts, we suggest downloading the databases under `$DATA_PATH/foldseek_databases/pdb` and `$DATA_PATH/foldseek_databases/afdb` using commands:
```bash
foldseek databases PDB pdb tmp
foldseek databases Alphafold/Proteome afdb tmp
```

The eventual `$DATA_PATH` directory should look like:
```
$DATA_PATH
    - metric_factory
        - features
            - D_FS_eval_ca_features.pth
            - D_FS_afdb_cath_codes.pth
            - pdb_eval_ca_features.pth
            - fold_class_mappings_C_selected_A_T_cath_codes.pth
        - model_weights
            - gearnet_ca.pth
    - pdb_raw
        - cath_label_mapping.pt
    - foldseek_databases
        - pdb
            - pdb
            - pdb.lookup
            - other related files
        - afdb
            - afdb
            - afdb.lookup
            - other related files
```

## Usage

### Training
After activating the environment, you can run the scripts or use the modules provided in the repository. Example:

```bash
bash run_sid_proteina.sh
```

Adjust the --batch-gpu parameter according to your GPU memory limitations. Setting it to 2 consumes about 27 GB of memory per GPU but will vary based on the lengths of the proteins.

### Inference
To generate protein structures unconditionally with specified lengths (number of residues), run
```bash
bash run_generate_distilled_proteina.sh 0
```
Replace 0 with the desired gpu device to use. If not specified, gpu0 will be used.

By default, the 16-step generator is used for generation. If a different model is desired, please make sure to change the `model_path`, `nstep`, and other relevant parameters in `run_generate_distilled_proteina.sh` as needed. Unless otherwise specified, the generated structures will be saved in `protein_out/proteina_(un)cond_distilled_{nstep}step_noise{noise_scale}_output` based on the parameters `nstep` and `noise_scale` as well as the `conditional` flag.

For fold-class-conditional generation, simply uncomment the `--conditional` flag in `run_generate_distilled_proteina.sh`. The lengths and CATH codes will be sampled from the empirical joint distribution as specified in `training/proteina/configs/experiment_config/inference_cond_sampling_specific_codes.yaml`. 

### Evaluation
To run the evaluation pipeline on the generated samples, run
```bash
python run_protein_evaluation.py --eval_input_dir path_to_samples
```

For example, if the samples are generated using our inference script with default parameters, the `--eval_input_dir` should be `protein_out/proteina_uncond_distilled_16step_noise0.45_output`. The resulting evaluation metrics are saved in `combined_results.csv` within the same folder.

## Checkpoints for our distilled generators
The few-step generators distilled from Proteina are provided [here](https://huggingface.co/LY-Xie/DistilledProteina/tree/main). 

## Citation
If you find this open source release useful, please cite in your paper:
```
@misc{xie2025distilledproteinbackbonegeneration,
      title={Distilled Protein Backbone Generation}, 
      author={Liyang Xie and Haoran Zhang and Zhendong Wang and Wesley Tansey and Mingyuan Zhou},
      year={2025},
      eprint={2510.03095},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.03095}, 
}
```

## Acknowledgement
This repo is heavily built upon [EDM](https://github.com/NVlabs/edm), [SiD](https://github.com/mingyuanzhou/SiD), and [Proteina](https://github.com/NVIDIA-Digital-Bio/proteina/). We thank the authors for their excellent work and for making their code publicly available.