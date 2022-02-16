# amodal_road
1. amodal road segmentation using amodal datasets
2. code modified based on DANet

## Usage

1. Install pytorch

   - The code is tested on python3.6 and torch 1.4.0.
   - The code is modified from [DANet](https://github.com/junfu1115/DANet.git).

2. Clone the resposity

   ```shell
   git clone https://github.com/coolgrasshopper/amodal_road_segmentation.git 
   cd amodal_road_segmentation
   ```
3. Key files related to training and model construction:

  - `experiments/segmentation/encoding/nn/loss.py`: define the cross entropy loss that is used for training
  - `experiments/segmentation/encoding/model/sseg/danet.py`: define the proposed model architecture
  - `experiments/segmentation/encoding/nn/da_att.py`: the Positional Attention module (PAM) and the Channel Attention Module
  - `experiments/segmentation/train.py`: training code
  - `experiments/segmentation/test.py`: testing code
  - `overlay.py`: evaluation code of far and close IOU

4. training:

```shell (example)
   cd experiments/segmentation
   CUDA_VISIBLE_DEVICES=0,1 python3 train.py --model danet --backbone resnet50 --checkname danet50 --base-size 1024 --crop-size 768 --epochs 240 --batch-size 8 --lr 0.003 --workers 16  
```
5. testing:

```shell (example)
   cd experiments/segmentation
   CUDA_VISIBLE_DEVICES=0 python3 test.py --model danet --backbone resnet50 --base-size 1024 --crop-size 768 --batch-size 8 --workers 16  
```
## TODO:
1. update evaluation code to the repo

## acknowlegement
[semantic-foreground-inpainting](https://github.com/Chenyang-Lu/semantic-foreground-inpainting.git)

[DANet](https://github.com/junfu1115/DANet.git)
