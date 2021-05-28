# SparseCT

This repo is a tool to develop sparse view CT reconstruction projects and compare different methods easily. The following papers are developed using this code repository.

## Papers

### Self-Supervised Training For Low Dose CT Reconstruction

- [Click here to access the paper](https://arxiv.org/abs/2010.13232)
- [Click here to reach the experiments of the paper](https://github.com/mozanunal/SparseCT/tree/master/papers/self_super_ct_reconstuction)

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

- [Click here to access the paper](https://arxiv.org/abs/2012.06448)
- [Click here to reach the experiments of the paper](https://github.com/mozanunal/SparseCT/tree/master/papers/dgr)

```
@misc{unal2020unsupervised,
      title={An Unsupervised Reconstruction Method For Low-Dose CT Using Deep Generative Regularization Prior}, 
      author={Mehmet Ozan Unal and Metin Ertas and Isa Yildirim},
      year={2020},
      eprint={2012.06448},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

## Demo

Example resuls from from [this paper](https://arxiv.org/abs/2010.13232).

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
