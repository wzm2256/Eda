# Equivariant Flow Matching for Point Cloud Assembly

This is the official implementation of the **Eda** model (equivariant diffusion assembly) described in [Equivariant Flow Matching for Point Cloud Assembly](https://arxiv.org/abs/2505.21539).

Eda assembles multiple 3D point clouds pieces into a complete shape following the geometric symmetric of the task. It is a probabilistic multi-piece extension of the [BiTr](https://github.com/wzm2256/BiTr) model.



Examples of the assembly process:

<img src="images\Eda_299_.gif" height="256"/> <img src="images\Eda_200_.gif" height="256"/> <img src="images\Eda_332_.gif" height="256"/> <img src="images\Eda_345_.gif" height="256"/> <img src="images\Eda_16_.gif" height="256"/><img src="images\Eda_89_.gif" height="128"/>



#### Envvironment

1) Install pytorch
2) Other requirement:
```
pip install einops scipy matplotlib tensorboard torch_ema torch_geometric pyyaml
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

#### Dataset and checkpoints

Download and unzip the zip files (BB_preload, match0.05, kitti) from https://drive.google.com/drive/folders/17iW6nqpUnLkcNLC-EEDGTyfqk9ATf_9K?usp=sharing to the `Data` folder.  The structure should be like:
```
Data/match0.05/
Data/BB_preload/
Data/kitti/
```

Download and unzip the `LOG.zip` file from the same link to the project folder. The structure should be like:
```
LOG/BB/
LOG/kitti4/
LOG/kitti3/
LOG/Match2L/
LOG/Match2Z/
```





#### Training and test
The script for training and testing 3DMatch/3DLMatch/3DZMatch, BB, and Kitti is in `a.txt`.



With the checkpoints and the test script, the following results should be reproduced.


| Task    | Rotate error | Translation error |
| -------- | ------- | ------- |
| 3DM  |  2.6   | 0.17  |
| 3DL  | 8.7     |0.4 |
| 3DZ  |  78.3   |2.7 |
| BB   | 85.4    | 0.18|
| Kitt-3-3  |   15.6  | 1.3 |
| Kitt-4-3  |  13.9   | 1.2 |

Here Kitti-m-n means Eda is trained on 2~m pieces and is tested on the n-piece task. See the paper for more details.



## Reference

    @misc{wang2025eq,
        title={Equivariant Flow Matching for Point Cloud Assembly}, 
        author={Ziming Wang and Nan Xue and Rebecka Jörnsten},
        year={2025},
        eprint={2505.21539},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2505.21539}, 
    }
    

For any question, please contact me (wzm2256@gmail.com).