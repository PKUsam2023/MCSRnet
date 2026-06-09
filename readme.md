# CrySTARNet

<div align='center'>
 
<!-- [![preprint](https://img.shields.io/static/v1?label=arXiv&message=2310.12508&color=B31B1B)](https://www.google.com/) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**Title** - CrySTARNet: X‑ray Diffraction Refinement for Disordered Inorganic Crystals via Generative Artificial Intelligence

---

## Table of Contents

- [CrySTARNet](#CrySTARNet)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Model Files](#Model-Files)
    - [CLIP Model](#CLIP Model)
	- [Diffusion Model](#Diffusion Model)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

---

## Introduction
The autonomous interpretation of diffraction data remains a major bottleneck in high-throughput materials discovery and automated laboratories. Although powder X-ray diffraction (XRD) is the most widely used technique for crystal structure determination, the projection of three-dimensional atomic arrangements into one-dimensional diffraction patterns inevitably leads to substantial information loss, making accurate structure refinement particularly challenging for complex and disordered materials. While recent machine-learning methods have achieved promising results in phase identification and property prediction, the autonomous generation of complete crystal structures from characterization data remains largely unexplored.

To address this challenge, we present the Crystal Structure Autonomous Refinement Network (CrySTARNet), a generative framework for end-to-end crystal structure refinement. CrySTARNet primarily utilizes powder XRD patterns and compositional information, while optionally incorporating transmission electron microscopy (TEM) data to further enhance structural discrimination. Through a contrastive multimodal representation learning strategy, XRD and TEM information are projected into a shared latent space, enabling effective extraction of complementary structural features when available. Guided by these representations, a conditional diffusion model reconstructs complete three-dimensional crystal structures, including lattice parameters, atomic coordinates, elemental species, and fractional site occupancies.

Unlike conventional refinement approaches that assume fixed stoichiometry and fully occupied crystallographic sites, CrySTARNet directly predicts continuous occupancy distributions, allowing native treatment of non-stoichiometric compounds, doped materials, and solid-solution systems. Evaluated across six representative crystal structure families, CrySTARNet achieves over 85% match accuracy within the top ten generated candidates, with several families exceeding 95% accuracy. These results demonstrate the capability of CrySTARNet to perform accurate and robust crystal structure refinement from diffraction data, providing a scalable solution for autonomous materials characterization and AI-driven materials discovery.

> **Keywords**: Deep learning, Multimodal learning, Automated Laboratories, Crystal structure refinement, Diffusion model

## Model Architecture
As illustrated in Figure 1, CrySTARNet consists of three sequential stages: multimodal encoding, crystal structure generation, and structure validation. The input data comprise powder XRD patterns, TEM images, and elemental composition information. During multimodal encoding, a contrastively pretrained XRD–TEM representation model projects diffraction and imaging data into a unified latent space, enabling effective fusion of complementary structural features from both modalities.

The fused representation is subsequently provided to a conditional diffusion-based crystal structure generator. Starting from random structural noise, the diffusion model progressively reconstructs the target crystal through iterative denoising, ultimately predicting lattice parameters, atomic positions, elemental species, and fractional occupancies. To improve robustness and account for the stochastic nature of diffusion generation, CrySTARNet produces multiple candidate structures for each sample. Generating ten candidates requires approximately 580 seconds for a typical perovskite system and 3720 seconds for more complex spinel structures, representing an effective balance between computational cost and refinement accuracy.

In the validation stage, each generated candidate undergoes structural and diffraction-based screening. A candidate is considered valid only if it satisfies predefined crystallographic similarity criteria and achieves a cosine similarity greater than 70% between its simulated and target XRD patterns. Among all valid candidates, the structure with the highest overall similarity score is selected as the final refinement result. The resulting crystal structure is exported as a crystallographic information file (CIF), accompanied by a simulated diffraction pattern for direct comparison with experimental measurements. This closed-loop workflow enables fully automated crystal structure refinement while ensuring consistency between atomic geometry and diffraction characteristics.

![Model Architecture](pic1.png)

Further explain the details in the [paper](https://github.com/PKUsam2023/CrySTARNet), providing context and additional information about the architecture and its components.

## Getting Started

### prerequisites
The code in this repo has been tested with the following software versions:
python==3.8.13
torch==1.9.0
torch-geometric==1.7.2
pytorch_lightning==1.3.8
pymatgen==2023.8.10

The installation can be done quickly with the following statement.
```
pip install -r requirements.txt
```

We recommend using the Anaconda Python distribution, which is available for Windows, MacOS, and Linux. Installation for all required packages (listed above) has been tested using the standard instructions from the providers of each package.

## Model-Files

### CLIP Model
For XRD-TEM CLIP Model:

Training: python ./XRD-TEM CLIP/train.py

Validation / Inference: python ./XRD-TEM CLIP/apply.py

For XRD CLIP Model:

Training: python ./XRD CLIP/train.py

Validation / Inference: python ./XRD CLIP/apply.py

### Diffusion Model
Training: python ./Diffusion/main/run.py data=<dataset> expname=<expname>

The <dataset> tag can be selected from perov_CaTiO3, perov_GdFeO3, perov_NdAlO3 and , and the <expname> tag can be an arbitrary name to identify each experiment.

Please download the dataset and model checkpoints from Zenodo:[10.5281/zenodo.17896706](https://zenodo.org/records/17900555). Please place them into the correct directories after downloading: rename best_xrd.pt by removing the _xrd suffix and put it into ./XRD CLIP/; rename best_xrd_TEM.pt by removing the _xrd_TEM suffix and put it into ./XRD-TEM CLIP/; place epoch=699-step=246399.ckpt into ./Diffusion/output/HYDRA/2025-03-12/perov_CaTiO3/; and after extracting the data archive, move the resulting folder into ./Diffusion/.

Evaluation:

One sample:

python ./Diffusion/scripts/evaluate.py --model_path <model_path> --dataset <dataset>

python ./Diffusion/scripts/compute_metrics.py --root_path <model_path> --tasks csp --gt_file data/<dataset>/test.csv 

Multiple samples:

python ./Diffusion/scripts/evaluate.py --model_path <model_path> --dataset <dataset> --num_evals 10

python ./Diffusion/scripts/compute_metrics.py --root_path <model_path> --tasks csp --gt_file data/<dataset>/test.csv --multi_eval

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this code or the pre-trained models in your work, please cite our work. 

- "CrySTARNet: X‑ray Diffraction Refinement for Disordered Inorganic Crystals via Generative Artificial Intelligence"

## Acknowledgments
The main framework of the Diffusion part is build upon DiffCSP. All entries in the database were extracted and curated from the ICSD.
