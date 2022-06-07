# Best Paper Finalist: [Learning to Recover 3D Scene Shape from a Single Image](https://arxiv.org/abs/2012.09365)

This repository contains the source code of the paper:
Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, Chunhua Shen [Learning to Recover 3D Scene Shape from a Single Image](https://arxiv.org/abs/2012.09365). Published in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR) 2021.

[NEW] Training codes have been released!!

## Some Results
<table>
  <tr>
    <td><img src="../examples/1.gif" width=400 height=300></td>
    <td><img src="../examples/2.gif" width=400 height=300></td>
    <td><img src="../examples/3.gif" width=400 height=300></td>
  </tr>
 </table>

You may want to check [this video](http://www.youtube.com/watch?v=UuT5_GK_TWk) which provides a very brief introduction to the work:

## Prerequisite

```bash
conda create -n LeReS python=3.7
conda activate LeReS
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

If you only want to test the monocular depth estimation from a single image, you can directly go to 'Quick Start' and follow Step 3. 
If you want to reconstruct 3D shape from a single image, please install [torchsparse](https://github.com/mit-han-lab/torchsparse) packages as follows. If you have any issues with torchsparse, please refer to [torchsparse](https://github.com/mit-han-lab/torchsparse).

```bash
#torchsparse currently only supports PyTorch 1.6.0 + CUDA 10.2 + CUDNN 7.6.2.
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.2.0
```



## Quick Start (Inference)

1. Download the model weights
   * [ResNet50 backbone](https://cloudstor.aarnet.edu.au/plus/s/VVQayrMKPlpVkw9)
   * [ResNeXt101 backbone](https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31)

2. Prepare data. 
   * Move the downloaded weights to  `LeReS/Minist_Test/` 
   * Put the testing RGB images to `LeReS/Minist_Test/test_images/`. Predicted depths and reconstructed point cloud are saved under `LeReS/Minist_Test/test_images/outputs`

3. Test monocular depth prediction. Note that the predicted depths are affine-invariant. 

    ```bash
    export PYTHONPATH="<PATH to Minist_Test>"
    # run the ResNet-50
    python ./tools/test_depth.py --load_ckpt res50.pth --backbone resnet50
    # run the ResNeXt-101
    python ./tools/test_depth.py --load_ckpt res101.pth --backbone resnext101
    ```

4. Test 3D reconstruction from a single image.

    ```bash
    export PYTHONPATH="<PATH to Minist_Test>"
    # run the ResNet-50
    python ./tools/test_shape.py --load_ckpt res50.pth --backbone resnet50
    # run the ResNeXt-101
    python ./tools/test_shape.py --load_ckpt res101.pth --backbone resnext101
    ```

## Training

1. (Optional) Run a demo training to verify the python environment.
   ```
   cd Train/scripts
   sh train_demo.sh
   ```

2. Download the training data. Please run 'download_data.sh' to achieve datasets of taskonomy, DiverseDepth, HWRSI and Holopix50k. All data are organized under the `Train/datasets`. The structure of all data are as follows.

    ```
    |--Train
    |   |--data
    |   |--lib
    |   |--scripts
    |   |--tools
    |   |--datasets
    |   |   |--DiverseDepth
    |   |   |   |--annotations
    |   |   |   |--depths
    |   |   |   |--rgbs
    |   |   |--taskonomy
    |   |   |   |--annotations
    |   |   |   |--depths
    |   |   |   |--rgbs
    |   |   |   |--ins_planes
    |   |   |--HRWSI
    |   |   |--Holopix50k
    ```

3. Train the network. The default setting used 4 gpus. If you want to use more gpus, please set `$CUDA_VISIBLE_DEVICES`, such as `export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. The `--batchsize` is the number of samples on a single gpu. 
   ```
   cd Train/scripts
   sh train.sh
   ```

4. Test the network on a benchmark. We provide a sample code for testing on NYU. Please download the NYU testing data [test.mat](https://cloudstor.aarnet.edu.au/plus/s/G2ckXCJX3pvrzRU) for evaluation and move it to `Train/datasets/test.mat`. If you want to test on other benchmarks, you can follow the sample code.
   ```
   cd Train/scripts
   sh test.sh
   ```


## BibTeX

```BibTeX
@inproceedings{Wei2021CVPR,
  title     =  {Learning to Recover 3D Scene Shape from a Single Image},
  author    =  {Wei Yin and Jianming Zhang and Oliver Wang and Simon Niklaus and Long Mai and Simon Chen and Chunhua Shen},
  booktitle =  {Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (CVPR)},
  year      =  {2021}
}
```

## License

This project is under a non-commercial license from Adobe Research. See the [LICENSE file](LICENSE) for details.
