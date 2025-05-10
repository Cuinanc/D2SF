# D2SF
3D人体姿态估计
****
## Environment
The project is developed under the following environment:
* python 3.9.19
* PyTorch 2.0.0
* CUDA 11.8

For installation of the project dependencies, please run:
```
pip install -r requirements.txt
```
## Dataset
### Human3.6M
#### Preprocessing
1.Download the fine-tuned Stacked Hourglass detections of MotionBERT's preprocessed H3.6M data [here](https://onedrive.live.com/?authkey=%21AMG5RlzJp%2D7yTNw&id=A5438CD242871DF0%21206&cid=A5438CD242871DF0&parId=root&parQt=sharedby&o=OneUp) and unzip it to 'data/motion3d'.
2.Slice the motion clips by running the following python code in `data/preprocess` directory:
#### For Our-S:
```
python h36m.py --n-frames 81
```
#### For Our-B and Our-L:
```
python h36m.py --n-frames 243
```
#### Visualization
Run the following command in the `data/preprocess` directory (it expects 243 frames):
```
python visualize.py --dataset h36m --sequence-number <AN ARBITRARY NUMBER>
```
This should create a gif file named `h36m_pose<SEQ_NUMBER>.gif` within `data` directory.
### MPI-INF-3DHP
#### Preprocessing
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.
#### Visualization
Run it same as the visualization for Human3.6M, but `--dataset` should be set to `mpi`.
## Training
After dataset preparation, you can train the model as follows:
### Human3.6M
You can train Human3.6M with the following command:
```
python train.py --config <PATH-TO-CONFIG>
```
where config files are located at `configs/h36m`. 
### MPI-INF-3DHP
You can train MPI-INF-3DHP with the following command:
```
python train_3dhp.py --config <PATH-TO-CONFIG>
```
where config files are located at `configs/mpi`.
## Training
Evaluation
|Method  | #frames| H3.6M weights  | MPI-INF-3DHP weights|
|--------- | --------|--------- | --------|
|Our-S  | 27 | [download](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fcheckpoint) |[download](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fcheckpoint) |
|Our-B  | 243 |[download](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fcheckpoint) |[download](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fcheckpoint) |
|Our-L  | 243 |[download](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fcheckpoint) |[download](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fcheckpoint) |
After downloading the weight from table above, you can evaluate Human3.6M models by:
```
python train.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```
Similarly, MPI-INF-3DHP can be evaluated as follows:
```
python train_3dhp.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```
## Demo
Our demo is a modified version of the one provided by [MHFormer](https://github.com/Vegetebird/MHFormer) repository. First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA) and put it in the './demo/lib/checkpoint' directory. Next, download our base model checkpoint from [here](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view) and put it in the './checkpoint' directory. Then, you need to put your in-the-wild videos in the './demo/video' directory.
Run the command below:
```
python demo/vis.py --video sample_video.mp4
```
Sample demo output:

![image](https://github.com/Cuinanc/D2SF/blob/main/tinywow_sample_video_80164781.gif)





























