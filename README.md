# MMSNet: Multi-Modal scene recognition using multi-scale encoded features
This repository provides the implementation of the following paper:<br/>
<br/>
**MMSNet: Multi-Modal scene recognition using multi-scale encoded features**<br/>
<a href="https://github.com/acaglayan" target="_blank">Ali Caglayan *</a>, <a href="https://scholar.google.com/citations?hl=en&user=VJgx61MAAAAJ&view_op=list_works&sortby=pubdate" target="_blank">Nevrez Imamoglu *</a>, <a href="https://www.airc.aist.go.jp/en/gsrt/" target="_blank">Ryosuke Nakamura</a>  
[<a href="https://deliverypdf.ssrn.com/delivery.php?ID=580124069074127013026064019068093126084045031036095011024026100020003031060111038068121028114008004079028048042114056099030016111091093127049009044098084099064020122077092082092084037000111047054002044068049036094021097003112067071099018126099087064127006125020080111069087099107098113025103&EXT=pdf&INDEX=TRUE" target="_blank">Paper</a>]


<br/>

![Graphical abstract](https://github.com/acaglayan/MMSNet/blob/main/figures/graph_abs.png)

## Requirements
Before starting, it is required to install the following libraries. Note that the package versions might need to be changed depending on the system:
```
conda create -n mmsnet python=3.7
conda activate mmsnet
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -U scikit-learn
pip install opencv-python
pip install psutil
pip install h5py
pip install seaborn
```
Also, source code path might need to be included to the PYTHONPATH (e.g. `export PYTHONPATH=$PYTHONPATH:/path_to_project/MMSNet/src/utils`).
## Data Preparation
### SUN RGB-D Scene
<a href="http://rgbd.cs.princeton.edu/" target="_blank">SUN RGB-D Scene</a> dataset is available <a href="http://rgbd.cs.princeton.edu/data/SUNRGBD.zip" target="_blank">here</a>. Keep the file structure as is after extracting the files. In addition, `allsplit.mat` and `SUNRGBDMeta.mat` files need to be downloaded from <a href="http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip" target="_blank">the SUN RGB-D toolbox</a>. `allsplit.mat` file is under `SUNRGBDtoolbox/traintestSUNRGBD` and  `SUNRGBDMeta.mat` is under `SUNRGBDtoolbox/Metada`. Both files should be placed under the root folder of SUN RGB-D dataset. E.g. :
<pre>
SUNRGBD ROOT PATH
├── SUNRGBD
│   ├── kv1 ...
│   ├── kv2 ...
│   ├── realsense ...
│   ├── xtion ...
├── allsplit.mat
├── SUNRGBDMeta.mat
</pre>
The dataset is presented in a complex hierarchy. Therefore, it's adopted to the local system as follows: 

```
python utils/organize_sunrgb_scene.py --dataset-path <SUNRGBD ROOT PATH>
```
This creates train/eval splits, copies RGB and depth files together with camera calibration parameters files for depth data under the corresponding split structure. Then, depth colorization is applied as below, which takes a couple of hours.
```
python utils/depth_colorize.py --dataset "sunrgbd" --dataset-path <SUNRGBD ROOT PATH>
```

### NYUV2 RGB-D Scene
<a href="https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html" target="_blank">NYUV2 RGB-D Scene</a> dataset is available <a href="http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat" target="_blank">here</a>. In addition, `splits.mat` file needs to be downloaded from <a href="http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat" target="_blank">here</a> together with `sceneTypes.txt` from <a href="https://github.com/acaglayan/MMSNet/blob/main/data/sceneTypes.txt" target="_blank">here</a>. The dataset structure should be something like below:
<pre>
NYUV2 ROOT PATH
├── nyu_depth_v2_labeled.mat
├── splits.mat
├── sceneTypes.txt
</pre>
Unlike other datasets, NYUV2 dataset is provided as a Matlab .mat file in `nyu_depth_v2_labeled.mat`. This work uses the provided in-painted depth maps and RGB images. In order to prepare depth data offline, depth colorization can be applied as follows:
```
python utils/depth_colorize.py --dataset "nyuv2" --dataset-path <NYUV2 ROOT PATH>
```
### Fukuoka RGB-D Scene
<a href="http://robotics.ait.kyushu-u.ac.jp/kyushu_datasets/indoor_rgbd.html" target="_blank">Fukuoka RGB-D Indoor Scene</a> dataset is used for the first time in the literature for benchmarking in this work. There are 6 categories: <a href="http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/corridors.tar.gz" target="_blank">Corridor</a>, <a href="http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/kitchens.tar.gz" target="_blank">kitchen</a>, <a href="http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/labs.tar.gz" target="_blank">lab</a>, <a href="http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/offices.tar.gz" target="_blank">office</a>, <a href="http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/studyrooms.tar.gz" target="_blank">study_room</a>, and <a href="http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/toilets.tar.gz" target="_blank">toilet</a>. The files should be extracted in a folder (e.g. `fukuoka`). The dataset structure should be something like below:
<pre>
Fukuoka ROOT PATH
├── fukuoka
│   ├── corridors ...
│   ├── kitchens ...
│   ├── labs ...
│   ├── offices ...
│   ├── studyrooms ...
│   ├── toilets ...
</pre> 
The dataset is organized using the following command, which creates `eval-set` under the root `fukuoka` path:
```
python utils/organize_fukuoka_scene.py --dataset-path <Fukuoka ROOT PATH> 
```
Then, depth colorization is applied similar to the other dataset usages.
```
python utils/depth_colorize.py --dataset "fukuoka" --dataset-path <Fukuoka ROOT PATH>
```
## Evaluation
### Trained Models
Trained models that give the results in the paper are provided as follows in a tree hierarchy. Download the models to run the evaluation code. Note that we share the used random weights here. However, it's possible to generate new random weights using the param `--reuse-randoms 0` (default 1). The results might change slightly (could be higher or lower). We discuss the effect of randomness in our previous paper <a href="https://authors.elsevier.com/a/1eXMb3qy-3WuW5" target="_blank">here</a>.
<pre>
ROOT PATH TO MODELS
├── models
│   ├── <a href="https://drive.google.com/file/d/1O_Jj9PH2id07SCPFkpRF5UQKr_YAWFCL/view?usp=sharing" target="_blank">resnet101_sun_rgb_best_checkpoint.pth</a>
│   ├── <a href="https://drive.google.com/file/d/1OjPGjxZW4lUdOucJ2Pix80HaNYtOajv9/view?usp=sharing" target="_blank">resnet101_sun_depth_best_checkpoint.pth</a>
│   ├── <a href="https://drive.google.com/file/d/1DZm4l5kP03AtWlyGvy6IXhZI1cf6tzeN/view?usp=sharing" target="_blank">sunrgbd_mms_best_checkpoint.pth</a>
│   ├── <a href="https://drive.google.com/file/d/1sM7owsRVi_6r0VdT2JU7gX8v7qH1ugEZ/view?usp=sharing" target="_blank">nyuv2_mms_best_checkpoint.pth</a>
│   ├── <a href="https://drive.google.com/file/d/1EtgJsWDXr1QslHqkOlLBfukiP3Sf8bfW/view?usp=sharing" target="_blank">fukuoka_mms_best_checkpoint.pth</a>
├── random_weights
│   ├── <a href="https://drive.google.com/file/d/19_tV1bWwfyN4q3NOLm67MWlSPoEXaLRJ/view?usp=sharing" target="_blank">resnet101_reduction_random_weights.pkl</a>
│   ├── <a href="https://drive.google.com/file/d/1UeZduyD8jo8aB_lLLOje2DVJfIN6VY9C/view?usp=sharing" target="_blank">resnet101_rnn_random_weights.pkl</a>
</pre> 
### Evaluation
After data preparation and downloading the models, to evaluate to models on SUN RGB-D, NYUV2, and Fukuoka RGB-D, run the following commands:
```
python eval_models.py --dataset "sunrgbd" --dataset-path <SUNRGBD ROOT PATH> --models-path <ROOT PATH TO MODELS>
python eval_models.py --dataset "nyuv2" --dataset-path <NYUV2 ROOT PATH> --models-path <ROOT PATH TO MODELS>
python eval_models.py --dataset "fukuoka" --dataset-path <Fukuoka ROOT PATH> --models-path <ROOT PATH TO MODELS>
```
## Results
Multi-modal performance comparison of this work (MMSNet) with the related methods on SUN RGB-D, NYUV2 RGB-D, and Fukuoka RGB-D Scene datasets in terms of accuracy (%).
Method | Paper | SUN RGB-D |  NYUV2 RGB-D | Fukuoka RGB-D |
:--------|:--------|:-------:|:-------:|:-------:|
Places CNN-RBF SVM | <a href="https://papers.nips.cc/paper/2014/hash/3fe94a002317b5f9259f82690aeea4cd-Abstract.html" target="_blank">NeurIPS’14</a> | 39.0 | - | -
SS-CNN-R6 | <a href="https://ieeexplore.ieee.org/abstract/document/7487381" target="_blank">ICRA’16</a> | 41.3 | - | -
DMFF | <a href="https://openaccess.thecvf.com/content_cvpr_2016/html/Zhu_Discriminative_Multi-Modal_Feature_CVPR_2016_paper.html" target="_blank">CVPR’16</a> | 41.5 | - | -
Places CNN-RCNN | <a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Wang_Modality_and_Component_CVPR_2016_paper.html" target="_blank">CVPR’16</a> | 48.1 | 63.9 | -
MSMM | <a href="https://www.ijcai.org/proceedings/2017/0631.pdf" target="_blank">IJCAI’17</a> | 52.3 | 66.7 | -
RGB-D-CNN  | <a href="https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14695" target="_blank">AAAI’17</a> | 52.4 | 65.8 | -
D-BCNN | <a href="https://www.sciencedirect.com/science/article/abs/pii/S0921889016304225" target="_blank">RAS’17</a> | 55.5 | 64.1 | -
MDSI-CNN | <a href="https://ieeexplore.ieee.org/abstract/document/8022892" target="_blank">TPAMI’18</a> | 45.2 | 50.1 | -
DF2Net | <a href="https://ojs.aaai.org/index.php/AAAI/article/view/12292" target="_blank">AAAI’18</a> | 54.6 | 65.4 | -
HP-CNN-T | <a href="https://link.springer.com/article/10.1007/s10514-018-9776-8" target="_blank">Auton.’19</a> | 42.2 | - | -
LM-CNN | <a href="https://link.springer.com/article/10.1007/s12559-018-9580-y" target="_blank">Cogn. Comput.’19</a> | 48.7 | - | -
RGB-D-OB | <a href="https://ieeexplore.ieee.org/abstract/document/8476560" target="_blank">TIP’19</a> | 53.8 | 67.5 | -
Cross-Modal Graph | <a href="https://ojs.aaai.org/index.php/AAAI/article/view/4952" target="_blank">AAAI’19</a> | 55.1 | 67.4 | -
RAGC | <a href="https://openaccess.thecvf.com/content_ICCVW_2019/html/GMDL/Mosella-Montoro_Residual_Attention_Graph_Convolutional_Network_for_Geometric_3D_Scene_Classification_ICCVW_2019_paper.html" target="_blank">ICCVW’19</a> | 42.1 | - | -
MAPNet | <a href="https://www.sciencedirect.com/science/article/abs/pii/S003132031930069X" target="_blank">PR’19</a> | 56.2 | 67.7 | -
TRecgNet Aug | <a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Du_Translate-to-Recognize_Networks_for_RGB-D_Scene_Recognition_CVPR_2019_paper.html" target="_blank">CVPR’19</a> | 56.7 | 69.2 | -
G-L-SOOR | <a href="https://ieeexplore.ieee.org/abstract/document/8796408" target="_blank">TIP’20</a> | 55.5 | 67.4 | -
MSN | <a href="https://www.sciencedirect.com/science/article/abs/pii/S0925231219313347" target="_blank">Neurocomp.’20</a> | 56.2 | 68.1 | -
CBCL | <a href="https://www.bmvc2020-conference.com/conference/papers/paper_0063.html" target="_blank">BMVC’20</a> | 59.5 | 70.9 | -
ASK | <a href="https://ieeexplore.ieee.org/abstract/document/9337174" target="_blank">TIP’21</a> | 57.3 | 69.3 | -
2D-3D FusionNet | <a href="https://www.sciencedirect.com/science/article/pii/S1566253521001032" target="_blank">Inf. Fusion’21</a> | 58.6 | <b>75.1</b> | -
TRecgNet Aug | <a href="https://link.springer.com/article/10.1007/s11263-021-01475-7" target="_blank">IJCV’21</a> | 59.8 | 71.8 | - 
CNN-randRNN | <a href="https://authors.elsevier.com/a/1eXMb3qy-3WuW5" target="_blank">CVIU’22</a> | 60.7 | 69.1 | 78.3
<b> MMSNet </b> | <b>This work</b> | <b>62.0</b> | <b>72.2</b> | <b>81.7</b>

We also share our `LaTeX` comparison tables together with the `bibtext` file for SUN RGB-D and NYUV2 benchmarking (see `LaTeX` <a href="https://github.com/acaglayan/MMSNet/tree/main/latex">directory</a>). Feel free to use them.
## Citation
If you find this work useful in your research, please cite the following papers:
```
@article{Caglayan2022MMSNet,
    title={MMSNet: Multi-Modal Scene Recognition Using Multi-Scale Encoded Features},
    journal = {SSRN},
    author={Ali Caglayan and Nevrez Imamoglu and Ryosuke Nakamura},
    doi = {http://dx.doi.org/10.2139/ssrn.4032570 },
    year={2022}
}

@article{Caglayan2022CNNrandRNN,
    title={When CNNs meet random RNNs: Towards multi-level analysis for RGB-D object and scene recognition},
    journal = {Computer Vision and Image Understanding},
    author={Ali Caglayan and Nevrez Imamoglu and Ahmet Burak Can and Ryosuke Nakamura},
    volume = {217},
    pages = {103373},
    issn = {1077-3142},
    doi = {https://doi.org/10.1016/j.cviu.2022.103373},
    year={2022}
}
```

## License
This project is released under the MIT License (see the LICENSE file for details).

## Acknowledgment
This  paper  is  based  on  the  results  obtained  from  a  project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
