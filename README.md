# RoofMapNet: Geometric Primitive-based Planar Roof Structure Reconstruction

## 📖 Introduction
This paper introduces RoofMapNet, an end-to-end deep learning framework that significantly improves the robustness and accuracy of roof structure extraction in complex scenarios. The framework incorporates an innovative progressive node extraction strategy and an adaptive occlusion-aware module to address challenges such as structural heterogeneity and occlusions. Furthermore, the authors have developed RoofMapSet, a large-scale and diverse remote sensing image dataset, to enable comprehensive evaluation of roof structure extraction performance.

**Paper**:  
[RoofMapNet: Utilizing Geometric Primitives for Depicting Planar Building Roof Structure from High-Resolution Remote Sensing Imagery](https://www.sciencedirect.com/science/article/pii/S1569843225002778)  
*International Journal of Applied Earth Observation and Geoinformation 2025*

<p align="center">
  <img src="assets/teaser.png" alt="RoofMapNet" width="80%">
  <br>
  <em>Figure : Architecture of RoofMapNet Framework.</em>
</p>

## 🚀 Quick Start

Get started with RoofMapNet in 3 steps:

### ⚙️ Prerequisites
- **Dependencies**:
  ```bash
  Python 3.7+
  PyTorch
  Numpy
  Matplotlib
  Skimage
  ```
### 📥 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/CVEO/RoofMapNet.git
   cd RoofMapNet
   ```
2. Create conda environment:
    ```bash
    conda create -n roofmapnet python=3.8
    conda activate roofmapnet
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 🔍 Evaluation
1. Download the pre-trained models:
    The pretrained weights on the RoofMapSet dataset can be downloaded [here](https://drive.google.com/file/d/1x0Ids8n48vT1fUgCG3vXEWladJlgcNo0/view?usp=sharing).

2. Dataset Preparation:
    Download the dataset from [data](https://github.com/whu-v2/RoofMapSet) and extract it to the `data` folder. 

3. Run the evaluation script:
    ```bash
    python inference.py configs/config.yaml pretrained_models/roofmapnet.pth data
    output/results
    ```
4. Calculation of the sAP metric
    ```bash
    python eval-sAP.py
    ```
## 📊 Results & Performance

### 🏆 Benchmark Performance (RoofMapSet Dataset)

| Model          | sAP<sup>5</sup> | sAP<sup>10</sup> | sAP<sup>15</sup> | mAP<sup>J</sup> | AP<sup>H</sup> | F<sup>H</sup> |
|----------------|:---------------:|:----------------:|:----------------:|:---------------:|:--------------:|:-------------:|
| LCNN           |      67.96      |      72.02       |      73.65       |      52.00      |     81.60      |     84.77     |
| F-CLIP         |      67.44      |      74.25       |      76.73       |      35.00      |     85.06      |     82.07     |
| HAWP           |      66.60      |      71.90       |      73.80       |      27.50      |     83.20      |     81.00     |
| HT-LCNN        |      69.04      |      74.31       |      76.11       |      53.40      |     84.95      |     86.18     |
| M-LSD          |      64.72      |      72.30       |      75.58       |      33.70      |     83.65      |     82.66     |
| ULSD           |      70.90      |      75.20       |      77.00       |      52.60      |     84.16      |     81.93     |
| **RoofMapNet** |    **73.47**    |    **78.19**     |    **79.99**     |    **57.20**    |   **88.73**    |   **87.40**   |
<!-- <span style="color:green">▲</span> | <span style="color:green">▲ +2.57</span> | <span style="color:green">▲ +2.99</span> | <span style="color:green">▲ +2.99</span> | <span style="color:green">▲ +3.80</span> | <span style="color:green">▲ +3.78</span> | <span style="color:green">▲ +1.22</span> | -->

<!-- **Key**:
- <span style="color:green">▲</span> : Improvement over previous SOTA (ULSD)
- **sAP<sup>θ</sup>**: Structural AP at tolerance θ pixels
- **mAP<sup>J</sup>**: Mean AP for junction detection
- **AP<sup>H</sup>/F<sup>H</sup>**: Polygon accuracy/F-score -->

## 🏢 RoofMapSet Dataset

The RoofMapSet dataset is specifically designed for extracting building outlines and roof structure information from remote sensing imagery. This dataset carefully selected 9,576 building instances from the [WHU Building Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html) and the [Inria Dataset](https://project.inria.fr/aerialimagelabeling/), covering diverse architectural styles, imaging conditions, and geographic locations. The data can be downloaded through this [link](https://drive.google.com/drive/folders/1l9LKZg8z6oarYiERQf6_1KS2MkdFlixZ?usp=sharing).

## 📚 Citation

If you find RoofMapNet useful in your research, please consider citing our paper:

```bibtex
@article{wang2025roofmapnet,
  title={RoofMapNet: Utilizing geometric primitives for depicting planar building roof structure from high-resolution remote sensing imagery},
  author={Wang, Jiaqi and Chen, Guanzhou and Zhang, Xiaodong and Wang, Tong and Tan, Xiaoliang and Yang, Qingyuan and Zhou, Wenlin and Zhu, Kun},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={141},
  pages={104630},
  year={2025},
  publisher={Elsevier}
}
```
### 📜 License
This project is released under the Non-Commercial Academic License. For commercial use, please contact the authors.

### 🤝 Acknowledgements and Reference
- This project includes code from [LCNN](https://github.com/zhou13/lcnn?tab=readme-ov-file)，Copyright (c) 2019-2020 Yichao Zhou，MIT Licensed.
- [WHU Building Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html)
- [Inria Dataset](https://project.inria.fr/aerialimagelabeling/)
