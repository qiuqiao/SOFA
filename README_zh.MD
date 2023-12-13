# SOFA: Singing-Oriented Forced Aligner

[English](README.MD) | 简体中文

![example](example.png)

# 介绍

SOFA(Singing-Oriented Forced Aligner)是一个专为歌声设计的强制对齐器，同时也兼容非歌声的对齐。

包含以下优点：

* 容易安装
* 
**注意：目前SOFA仍旧是beta版本，可能会有很多bug且不保证效果。如果发现任何问题或者改进建议，欢迎提issue。**

# 使用方法

## 环境配置

1. 使用`git clone`​​下载本仓库的代码
2. 安装conda
3. 创建conda环境，python版本的要求为`3.8`
    ```bash
    conda create -n SOFA python=3.8 -y
    conda activate SOFA
    ```
4. 去[pytorch官网](https://pytorch.org/get-started/locally/)安装torch
5. （可选，用于提高wav读取速度）去[pytorch官网](https://pytorch.org/get-started/locally/)安装torchaudio
6. 安装其他python库
    ```bash
    pip install -r requirements.txt
    ```

## 推理

1. 下载模型文件。你可以在[本仓库的release](github.com/qiuqiao/SOFA/releases)中找到训练好的模型，也可以从[discussion板块](github.com/qiuqiao/SOFA/discussions/categories/pretrained-model-sharing)中找到用户分享的模型下载链接，文件后缀为`.ckpt`。
2. 把模型文件移动到`/ckpt`​​文件夹下。
3. 把字典文件放入`/dictionary`​​文件夹中。默认字典为`opencpop-extension.txt`​​
4. 准备需要强制对齐的数据，放入一个文件夹中（默认放在`/segments`​​文件夹），格式如下
    ```text
    - segments
        - singer1
            - segment1.lab
            - segment1.wav
            - segment2.lab
            - segment2.wav
            - ...
        - singer2
            - segment1.lab
            - segment1.wav
            - ...
    ```
    保证`.wav`文件和对应的`.lab`在同一个文件夹即可。
5. 命令行推理

    使用`python infer.py`进行推理。

    需要指定的参数：
    - `--ckpt`：（必须指定）模型权重路径；
    - `--folder`：存放待对齐数据的文件夹​（默认为`segments`）；
    - `--dictionary`：字典文件​（默认为`dictionary/opencpop-extension.txt`​）；

    ```bash
    python infer.py -c checkpoint_name -s segments_path -d dictionary_path
    ```
### 高级功能

   - 使用自定义的g2p，而不是使用词典
     - 参见[g2p模块使用说明](modules/g2p/readme_g2p_zh.md)
   - matching模式，推理时指定`-m`即可开启，会在给定的音素序列中找到一个使得概率最大的连续序列片段，而非必须用上所有音素。
## 训练

1. 参照上文进行环境配置。建议安装torchaudio以获得更快的binarize速度；
2. 把训练数据按以下格式放入`data`文件夹：

    ```
    - data
        - full_label
            - singer1
                - wavs
                    - audio1.wav
                    - audio2.wav
                    - ...
                - transcriptions.csv
            - singer2
                - wavs
                    - ...
                - transcriptions.csv
        - weak_label
            - singer3
                - wavs
                    - ...
                - transcriptions.csv
            - singer4
                - wavs
                    - ...
                - transcriptions.csv
        - no_label
            - audio1.wav
            - audio2.wav
            - ...
    ```
    其中：

    `transcriptions.csv`只需要和`wavs`文件夹的相对位置正确即可；

    `weak_label`中的`transcriptions.csv`无需拥有`ph_dur`这一个`column`；
3. 按需修改`binarize_config.yaml`，然后执行`python binarize.py`；
4. 在release中下载你需要的预训练模型，并按需修改`train_config.yaml`，然后执行`python train.py -p path_to_your_pretrained_model`；
5. 训练可视化：`tensorboard --logdir=ckpt/`。