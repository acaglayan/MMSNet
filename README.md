# MMSNet: Multi-Modal scene recognition using multi-scale encoded features
This repository provides the implementation of the following paper:<br/>
<br/>
**MMSNet: Multi-Modal scene recognition using multi-scale encoded features**<br/>
<a href="https://github.com/acaglayan" target="_blank">Ali Caglayan *</a>, <a href="https://scholar.google.com/citations?hl=en&user=VJgx61MAAAAJ&view_op=list_works&sortby=pubdate" target="_blank">Nevrez Imamoglu *</a>, <a href="https://www.airc.aist.go.jp/en/gsrt/" target="_blank">Ryosuke Nakamura</a>  
[<a href="https://deliverypdf.ssrn.com/delivery.php?ID=580124069074127013026064019068093126084045031036095011024026100020003031060111038068121028114008004079028048042114056099030016111091093127049009044098084099064020122077092082092084037000111047054002044068049036094021097003112067071099018126099087064127006125020080111069087099107098113025103&EXT=pdf&INDEX=TRUE" target="_blank">Paper</a>]


<br/>

![Graphical abstract](https://github.com/acaglayan/MMSNet/blob/main/figures/graph_abs.png)

## Requirements
Before starting, it is needed to install following libraries. Note that the software package versions might need to be changed depending on the system:
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
Finally , source codes might need to be included to the PYTHONPATH (e.g. `export PYTHONPATH=$PYTHONPATH:/home/user/path_to_project/MMSNet/src/utils`).
## Data Preparation
### SUN RGB-D Scene
<a href="http://rgbd.cs.princeton.edu/" target="_blank">SUN RGB-D Scene</a> dataset is available <a href="http://rgbd.cs.princeton.edu/data/SUNRGBD.zip" target="_blank">here</a>. Keep the file structure as is after extracting the files. In addition, `allsplit.mat` and `SUNRGBDMeta.mat` files need to be downloaded from <a href="http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip" target="_blank">the SUN RGB-D toolbox</a>. `allsplit.mat` file is under `SUNRGBDtoolbox/traintestSUNRGBD` and  `SUNRGBDMeta.mat` is under `SUNRGBDtoolbox/Metada`. Both files need to be placed under the root folder of SUN RGB-D dataset. E.g. :
<pre>
sunrgbd
├── SUNRGBD
│   ├── kv1 ...
│   ├── kv2 ...
│   ├── realsense ...
│   ├── xtion ...
├── allsplit.mat
├── SUNRGBDMeta.mat
</pre>
The dataset is presented in a complex hierarchy. Therefore, it's adopted to the local system using the following commands: 

```
python utils/organize_sunrgb_scene.py --dataset-path </SUNRGBD ROOT PATH/> 
```
This creates train/eval splits, copies RGB and depth files together with camera calibration parameters files for depth data under the corresponding split structure. Then, depth colorization is applied, which takes a couple of hours.
```
python utils/depth_colorize.py --dataset "sunrgbd" --dataset-path </SUNRGBD ROOT PATH/> --models-root <ROOT PATH TO MODELS>
```

### NYUV2 RGB-D Scene
<a href="https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html" target="_blank">NYUV2 RGB-D Scene</a> dataset is available <a href="http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat" target="_blank">here</a>. In addition, `splits.mat` file needs to be downloaded from <a href="??" target="_blank">?? toolbox??</a> together with `sceneTypes.txt` from <a href="??" target="_blank">here</a>. The dataset structure should be something like below:
<pre>
nyuv2
├── nyu_depth_v2_labeled.mat
├── splits.mat
├── sceneTypes.txt
</pre>
Unlike other datasets, the dataset is provided as a Matlab .mat file in `nyu_depth_v2_labeled.mat`. We use the provided in-painted depth maps and RGB images. Depth colorization can be applied as follows in order to prepare depth data offline.
```
python utils/depth_colorize.py --dataset "nyuv2" --dataset-path <NYUV2 ROOT PATH> --models-root <ROOT PATH TO MODELS>
```
### Fukuoka RGB-D Scene
<a href="http://robotics.ait.kyushu-u.ac.jp/kyushu_datasets/indoor_rgbd.html" target="_blank">Fukuoka RGB-D Indoor Scene</a> dataset is used for the first time in the literature for benchmarking with this work. There are 6 categories: <a href="http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/corridors.tar.gz" target="_blank">Corridor</a>, <a href="http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/kitchens.tar.gz" target="_blank">kitchen</a>, <a href="http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/labs.tar.gz" target="_blank">lab</a>, <a href="http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/offices.tar.gz" target="_blank">office</a>, <a href="http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/studyrooms.tar.gz" target="_blank">study_room</a>, and <a href="http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/toilets.tar.gz" target="_blank">toilet</a>. The files should be extracted in a folder (e.g. `fukuoka`). The dataset structure should be something like below:
<pre>
fukuoka
├── fukuoka
│   ├── corridors ...
│   ├── kitchens ...
│   ├── labs ...
│   ├── offices ...
│   ├── studyrooms ...
│   ├── toilets ...
</pre> 
The dataset is organized using the following command, which creates `eval-set` under root `fukuoka` folder:
```
python utils/organize_fukuoka_scene.py --dataset-path </Fukuoka ROOT PATH/> 
```
Then, depth colorization is applied similar to other dataset usages.
```
python utils/depth_colorize.py --dataset "fukuoka" --dataset-path </Fukuoka ROOT PATH/> --models-root <ROOT PATH TO MODELS>
```
## Evaluation
### Trained Models
Trained models that give the results in the paper are provided as follows in the tree hierarchy. Download the models to run the evaluation code. Note that we share the used random weights here. However, it's possible to generate new random weights using the param `--reuse-randoms 0` (default 1). The results might change slightly (could be higher or lower). We discuss the effect of randomness in our previous paper <a href="https://authors.elsevier.com/a/1eXMb3qy-3WuW5" target="_blank">here</a>.
<pre>
models-features
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
python main.py --dataset "sunrgbd" --dataset-path </SUNRGBD ROOT PATH/> --models-root <ROOT PATH TO MODELS>
python main.py --dataset "nyuv2" --dataset-path <NYUV2 ROOT PATH> --models-root <ROOT PATH TO MODELS>
python main.py --dataset "fukuoka" --dataset-path </Fukuoka ROOT PATH/> --models-root <ROOT PATH TO MODELS>
```
