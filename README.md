# SparseCT

This repo is a tool to develop sparse view CT reconstruction projects and compare different methods easily. The following papers are developed using this code repository.

## Papers

### Self-Supervised Training For Low Dose CT Reconstruction

- [Click here to access the paper](https://ieeexplore.ieee.org/abstract/document/9433944)
- [Click here to reach the experiments of the paper](https://github.com/mozanunal/SparseCT/tree/SelfSuperLDCT-Paper)

```
@INPROCEEDINGS{9433944,
  author={Unal, Mehmet Ozan and Ertas, Metin and Yildirim, Isa},
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)}, 
  title={Self-Supervised Training For Low-Dose Ct Reconstruction}, 
  year={2021},
  volume={},
  number={},
  pages={69-72},
  doi={10.1109/ISBI48211.2021.9433944}}
```

### An Unsupervised Reconstruction Method For Low-Dose CT Using Deep Generative Regularization Prior

- [Click here to access the paper](https://www.sciencedirect.com/science/article/pii/S1746809422001203)
- [Click here to reach the experiments of the paper](https://github.com/mozanunal/SparseCT/tree/DGR-Paper)

```
@article{unal2020unsupervised,
  title={An Unsupervised Reconstruction Method For Low-Dose CT Using Deep Generative Regularization Prior},
  author={Unal, Mehmet Ozan and Ertas, Metin and Yildirim, Isa},
  journal={Biomedical Signal Processing and Control},
  volume={75},
  number={1746-8094},
  pages={103598},
  year={2020},
  publisher={Elsevier}
}
```

## Demo

Example resuls from from [this paper](https://ieeexplore.ieee.org/abstract/document/9433944).

![](https://raw.githubusercontent.com/mozanunal/SparseCT/master/docs/images/result2.png)

From left to right: ground truth, FBP, SART, SART+TV, SART+BM3D, the proposed method (learned self-supervised).


## Install

The installation tested on Ubuntu 18.04. The following linux packages are required.

```
sudo apt install python3-dev python3-pip \
        libopenblas-dev

```

The python libraries which are defined in requirements.txt should also be installed.

```
pip install -r requirements.txt
``` 

## Development

## Contributing

Please implement your constructor according to Reconstructor abstract class. A contribution guide will be added 

## Acknowledgements

In this code repository the packages in requirement.txt are used.
There are some code parts from following code repositories are used directly to port the methods for CT reconstruction.

- Noise2Self [[code]](https://github.com/czbiohub/noise2self) [[paper]](https://arxiv.org/abs/1901.11365)
- Deep Image prior [[code]](https://github.com/DmitryUlyanov/deep-image-prior) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.pdf)



## Licence

Please see LICENSE file
