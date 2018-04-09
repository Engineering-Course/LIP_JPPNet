## Joint Body Parsing & Pose Estimation Network (JPPNet)
Xiaodan Liang, Ke Gong, Xiaohui Shen, and Liang Lin, "Look into Person: Joint Body Parsing & Pose Estimation Network and A New Benchmark", T-PAMI 2018.

### Introduction

JPPNet is a state-of-art deep learning methord for human parsing and pose estimation built on top of [Tensorflow](http://www.tensorflow.org).

This novel joint human parsing and pose estimation network incorporates the multiscale feature connections and iterative location refinement in an end-to-end framework to investigate efficient context modeling and then enable parsing and pose tasks that are mutually beneficial to each other. This unified framework achieves state-of-the-art performance for both human parsing and pose estimation tasks. 


This distribution provides a publicly available implementation for the key model ingredients reported in our latest [paper](https://arxiv.org/pdf/1804.01984.pdf) which is accepted by T-PAMI 2018.

We simplify the network to solve human parsing by exploring a novel self-supervised structure-sensitive learning approach, which imposes human pose structures into the parsing results without resorting to extra supervision. There is also a public implementation of this self-supervised structure-sensitive JPPNet ([SS-JPPNet](https://github.com/Engineering-Course/LIP_SSL)).

Please consult and consider citing the following papers:

    @article{liang2018look,
      title={Look into Person: Joint Body Parsing \& Pose Estimation Network and a New Benchmark},
      author={Liang, Xiaodan and Gong, Ke and Shen, Xiaohui and Lin, Liang},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2018},
      publisher={IEEE}
    }

    @InProceedings{Gong_2017_CVPR,
      author = {Gong, Ke and Liang, Xiaodan and Zhang, Dongyu and Shen, Xiaohui and Lin, Liang},
      title = {Look Into Person: Self-Supervised Structure-Sensitive Learning and a New Benchmark for Human Parsing},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {July},
      year = {2017}
    }


### Look into People (LIP) Dataset

The SSL is trained and evaluated on our [LIP dataset](http://www.sysu-hcp.net/lip) for human parsing.  Please check it for more model details. The dataset is also available at [google drive](https://drive.google.com/drive/folders/0BzvH3bSnp3E9ZW9paE9kdkJtM3M?usp=sharing) and [baidu drive](http://pan.baidu.com/s/1nvqmZBN).


### Pre-trained models

Coming soon!


### Train and test

Coming soon!