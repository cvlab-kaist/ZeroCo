<div align="center">
<h1>Cross-View Completion Models are <br> Zero-shot Correspondence Estimators</h1>
<h3>CVPR 2025</h3>

[**Honggyu An**](https://hg010303.github.io)<sup>1\*</sup> · [**Jinhyeon Kim**](https://github.com/jinlovespho)<sup>2\*</sup> · [**Seonghoon Park**](https://github.com/seong0905)<sup>3</sup> · [**Jaewoo Jung**](https://crepejung00.github.io/)<sup>1</sup> <br>
[**Jisang Han**](https://onground-korea.github.io)<sup>1</sup> . [**Sunghwan Hong**](https://sunghwanhong.github.io/)<sup>2</sup> . [**Seungryong Kim**](https://cvlab.korea.ac.kr)<sup>1&dagger;</sup>

<sup>1</sup>KAIST&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Korea University&emsp;&emsp;&emsp;&emsp;<sup>3</sup>Samgsung Electronics <br>
*: Equal Contribution &emsp;&emsp;&emsp;&emsp; &dagger;: Corresponding Author

<a href="https://arxiv.org/abs/2412.09072"><img src="https://img.shields.io/badge/arXiv-ZeroCo-red"></a>
<a href="https://cvlab-kaist.github.io/ZeroCo/"><img src="https://img.shields.io/badge/Project%20Page-ZeroCo-brightgreen"></a>
</div>

## Prerequisites

- Environment setup
  ```bash
  conda create -n CVC python=3.10.15
  conda activate CVC
  pip install -r requirements.txt
  ```

- Dataset preparation
  ```bash
  bash download_ETH3D.sh
  bash download_hpatches.sh
  ```

- CVC model checkpoint preparation
  - [CroCo](https://github.com/naver/croco)
  - [DUSt3R](https://github.com/naver/dust3r)  
  - [MASt3R](https://github.com/naver/mast3r)

- Create admin/local by running the following command and update the paths to the dataset
```
python -c "from admin.environment import create_default_local_file; create_default_local_file()"
```


## Zero-shot Evaluation
The `scripts` folder contains multiple bash files for evaluating models on either the HPatches or ETH3D datasets. Most experiments were conducted on HPatches. For each model, you can perform zero-shot evaluation of geometric matching performance using one of three methods:

### Available Methods
1. **Encoder Correlation**: Uses encoder features to build a correlation
2. **Decoder Correlation**: Uses decoder features to build a correlation
3. **Cross-Attention Maps**: Uses cross-attention maps for correlation

For detailed explanations of each method, please refer to our paper.

### Example Commands

```bash
# HPatches (Original Resolution) - CroCov2
bash scripts/run_hp_crocov2_Largebase.sh

# HPatches (240 Resolution) - CroCov2
bash scripts/run_hp240_crocov2_LargeBase.sh

# ETH3D - CroCov2 
bash scripts/run_eth3d_crocov2_LargeBase.sh
```

### Script Configuration Details
Each evaluation script contains several key parameters that can be customized:

```bash
# Example evaluation script
CUDA=3  # Specify GPU device rank
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 2024                     # Random Seed Selection
    --dataset hp                    # Dataset (hp: HPatches, hp-240: HPatches (240x240), eth3d: ETH3D)
    --model_img_size 224 224        # CVC model's Input image dimensions
    --model crocov2                 # Model type [crocov1, crocov2, dust3r, mast3r]
    --pre_trained_models croco      # Pre-trained model type
    --croco_ckpt /path/to/croco/ckpts/CroCo_V2_ViTLarge_BaseDecoder.pth
    --output_mode ca_map            # Correlation method choose from [enc_feat, dec_feat, ca_map]
    --output_ca_map                 # Enable cross-attention map output
    --reciprocity                   # Enable reciprocal cross-attention map
    --save_dir /path/to/save/images/for/visualisation/  
```

### Acknowledgements
This code is heavily based on [DenseMatching](https://github.com/PruneTruong/DenseMatching), We highly appreciate the authors for their great work.

### Citation
If you found this code useful, please consider citing our paper.

```
@article{an2024cross,
  title={Cross-View Completion Models are Zero-shot Correspondence Estimators},
  author={An, Honggyu and Kim, Jinhyeon and Park, Seonghoon and Jung, Jaewoo and Han, Jisang and Hong, Sunghwan and Kim, Seungryong},
  journal={arXiv preprint arXiv:2412.09072},
  year={2024}
}
```
