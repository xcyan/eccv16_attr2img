# Attribute2Image

This is the code for ECCV 2016 paper [Attribute2Image: Conditional Image Generation from Visual Attributes](https://arxiv.org/abs/1512.00570) by Xinchen Yan, Jimei Yang, Kihyuk Sohn and Honglak Lee.

<img src="https://umich.box.com/shared/static/gm2kajfs7cf88hvcsbjviggt7u7tijyj.png" width="900px" height="360px"/>

Please follow the instructions to run the code.

## Requirements
Attribute2Image requires or works with
* Mac OS X or Linux
* NVIDIA GPU

## Installing Dependency
* Install [Torch](http://torch.ch)

## Data Preprocessing
* For LFW dataset, please run the script to download the pre-processed dataset
```
./prep_cropped_lfw.sh
```
* Disclaimer: Please cite the [LFW paper](https://link.springer.com/book/10.1007/978-3-319-25958-1) if you download this pre-processed version.

* For CelebA dataset, please download the [original dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and then run the script for pre-processing 
```
./prep_cropped_celeba.sh
```
* Alternatively, you can download the pre-processed .t7 files with the following script:
```
./download_preprocessed_celeba.sh
```
* Disclaimer: Please cite the [CelebA paper](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) if you download the pre-processed .t7 files.

* For CUB dataset, please run the script to download the pre-processed dataset
```
./prep_cropped_cub.sh
```
## Training (vanilla CVAE)
* If you want to train the LFW image generator, please run the script (less than 3 hours on a single Titan X GPU)
```
./demo_lfw_trainCVAE.sh
```
* If you want to train the CelebA image generator, please run the script (around 24 hours on a single Titan X GPU)
```
./demo_celeba_trainCVAE.sh
```
## Training (disentangling CVAE)
* If you want to train the LFW layered image generator, please run the script (less than 5 hours on a single Titan X GPU)
```
./demo_lfw_trainDisCVAE.sh
```
* If you want to train the CUB layered image generator, please run the script (less than 3 hours on a single Titan X GPU)
```
./demo_cub_trainDisCVAE.sh
```
## Visualization using Pre-trained Models
TBD

## Citation

If you find this useful, please cite our work as follows:
```
@article{yan2015attribute2image,
  title={Attribute2Image: Conditional Image Generation from Visual Attributes},
  author={Yan, Xinchen and Yang, Jimei and Sohn, Kihyuk and Lee, Honglak},
  journal={arXiv preprint arXiv:1512.00570},
  year={2015}
}
```

Please contact "skywalkeryxc@gmail.com" if any questions. 
