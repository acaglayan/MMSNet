# MMSNet: Multi-Modal scene recognition using multi-scale encoded features
This repository provides the implementation of the following paper:<br/>
<br/>
**MMSNet: Multi-Modal scene recognition using multi-scale encoded features**<br/>
<a href="https://github.com/acaglayan" target="_blank">Ali Caglayan</a>, <a href="https://scholar.google.com/citations?hl=en&user=VJgx61MAAAAJ&view_op=list_works&sortby=pubdate" target="_blank">Nevrez Imamoglu</a>, <a href="https://www.airc.aist.go.jp/en/gsrt/" target="_blank">Ryosuke Nakamura</a>
<br/>
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
