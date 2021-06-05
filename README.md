# CV-leaf-segm

A method for recognising plant leaves in rosette plants. Inspired by [[1]](#1). Given 16 plant images of avg size (w,h) 122x132*px* achieves HSV colour segmentation, k-means clustering segmentation, watershed instance segmentation and indexing. 

<p align="middle">
  <img src="/assets/Figure_11.png" width="500" />
</p>

## Starter
* install packages\
`$ pip install -r requirements.txt`

make sure input and labels are in ./data and run\
`python main.py`

#### Finding custom threshold values
* Create collage picture of all raw images.
* Use https://github.com/alkasm/colorfilters on collage to interactively find colour values. 
<p align="middle">
  <img src="/assets/merge_from_ofoct.jpg" width="500" />
  <img src="/assets/asset.png" width="400" /> 
</p>
Program outputs plots for all images (16 in this case) in 2 phases:

#### Phase 1
The pipeline's last stage instance segmentation with waterhsed algorithm (left) and leaf detection using hough circles on post-processed semantic segmentation output (right).
<p align="middle">
  <img src="/assets/Figure_12.png" width="500" />
</p>

#### Phase 2
Inputs (image, label) and output at each stage of the pipeline. DS is used for colour threshold accuracy and is calculated using the Sørensen–Dice coefficient. Blue denotes leaf count accuracy from watershed, purple from k-means clustering.

<p align="middle">
  <img src="/assets/Figure_2.png" width="700" />
</p>

## References
<a id="1">[1]</a> 
Kumar, J.P. and Domnic, S., 2019. Image based leaf segmentation and counting in rosette plants. Information Processing in Agriculture, 6(2), pp.233-246.
