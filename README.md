# One Framework to Register Them All

**[[Paper]](https://arxiv.org/abs/1912.05766)**

Source Code Author: Vinit Sarode, Xueqian Li and Animesh Dhagat

<p align="center">
	<img src="https://github.com/vinits5/pointnet-registration-framework/blob/master/images/flowchart.png" height="500">
</p>


### Requirements:
1. Cuda 10
2. tensorflow==1.14
3. transforms3d==0.3.1
4. h5py==2.9.0
5. pytorch==1.3.0

#### Dataset:
Path for dataset: [Link](https://drive.google.com/drive/folders/19X68JeiXdeZgFp3cuCVpac4aLLw4StHZ?usp=sharing)
1. Download 'train_data' folder from above link.
2. Download 'car_data' folder from above link.

### Point Cloud Registration Network:
#### Train Iterative-PCRNet:
1. cd pcrnet
2. chmod +x train_itrPCRNet.sh
3. ./train_itrPCRNet.sh

### PointNetLK:
#### Train PointNetLK:
1. cd pnlk
2. cd experiments
3. python train_pointlk.py

### Citation

```
@misc{sarode2019framework,
    title={One Framework to Register Them All: PointNet Encoding for Point Cloud Alignment},
    author={Vinit Sarode and Xueqian Li and Hunter Goforth and Yasuhiro Aoki and Animesh Dhagat and Rangaprasad Arun Srivatsan and Simon Lucey and Howie Choset},
    year={2019},
    eprint={1912.05766},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```