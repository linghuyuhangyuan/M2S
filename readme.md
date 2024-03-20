# Minutes to Seconds: Speeded-up DDPM-based Image Inpainting with Coarse-to-Fine Sampling (ICME2024)

Lintao Zhang, Xiangcheng Du, LeoWu TomyEnrique, Yiqun Wang, Yingbin Zheng, Cheng Jin

Fudan University, Videt Technology

<div style="text-align:center">
    <img src="assets/fig.png" alt="image" />
</div>

---



## Installation

1. Clone our repository

   ```
   git clone https://github.com/linghuyuhangyuan/M2S.git
   cd M2S
   ```

2. Make conda environment

   ```
   conda create -n M2S python=3.8
   conda activate M2S
   ```

   ```
   pip install -r requirements.txt
   ```



## Data Preparation

The inputs of Image Inpainting include original images and binary masks.

1. Image

   We conduct experiements on two datasets: [CelebA-HQ](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)  and [ImageNet](https://www.kaggle.com/datasets/dimensi0n/imagenet-256) at 256×256 pixels.

2. Mask

   We use the mask test sets of [RePaint](https://github.com/andreas128/RePaint), which include 6 types: Wide, Narrow, Half, Expand, Alternating Lines and Super-Resolve 2×. You can download these datasets from their provided [Google Drive link](https://drive.google.com/uc?id=1Q_dxuyI41AAmSv9ti3780BwaJQqwvwMv).



## Training

We employ a pretrained Denoising Diffusion Probabilistic Model (DDPM) as the generative prior. For speeding up, we use a Light-Weight Diffusion Model from [P2-weighting](https://github.com/jychoi118/P2-weighting), substituting the large-parameter DDPM from [guided-diffusion](https://github.com/openai/guided-diffusion).

Training code can be found in the repository [P2-weighting](https://github.com/jychoi118/P2-weighting). our trained models of 64×64 resolution for the coarse stage and 256×256 resolution for the refinement stage are accessible in this [Google Drive link](https://drive.google.com/drive/folders/1PR87Kt5WmmFzutSvFS99UVtBNgXomTI1?usp=sharing).



## Evaluation

Download pretrained model from [Google Drive](https://drive.google.com/drive/folders/1PR87Kt5WmmFzutSvFS99UVtBNgXomTI1?usp=sharing) and place them within the `models` directory.

1. First, set PYTHONPATH variable to point to the root of the repository.

   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

2. Run demo.

   ```
   sh run.sh
   ```

   The visualized outputs will be gererated in ```results/celebahq/thick```.

   The quantified metric results are displayed in ```results/celebahq/thick/metrics_log.txt```. 


   If you want to try other images and different mask types, please modify `--base_samples` and `--mask_path` in `run.sh`.

**Note:** For special mask types: Alternating Lines and Super-Resolve 2×, please ensure to set `--special_mask True` in the `run.sh` script.



## Acknowledgment

This code is based on the [RePaint](https://github.com/andreas128/RePaint),  [P2-weighting](https://github.com/jychoi118/P2-weighting) and [guided-diffusion](https://github.com/openai/guided-diffusion). Thanks for their awesome works.



## Contact

If you have any question or suggestion, please contact ltzhang21@m.fudan.edu.cn.