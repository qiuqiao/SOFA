# readme

**说明：目前SOFA仍旧是beta版本，可能会有很多bug且不保证效果。如果发现任何问题或者改进建议，欢迎提issue。**

# 介绍

SOFA: Singing-Oriented Forced Aligner是一个专为歌声设计的强制对齐器，同时也兼容非歌声的对齐。

包含以下优点：

* 容易安装

# 使用方法

## 推理

1. 使用`git clone`​​下载本仓库的代码
2. 安装所需的python环境
   1. 创建conda环境，python版本的要求为`3.10`（并不代表其他版本无法工作，你可以自己试试）
   `conda create -n SOFA python=3.10 -y`
   `conda activate SOFA`
   2. 去[pytorch官网](https://pytorch.org/get-started/locally/)安装torch（仅在2.0.1版本进行测试，其他版本有可能出现错误）
   3. 安装其他python库`pip -r requirements.txt`
   如果你无法访问外网，请使用
   ```
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
3. 下载模型文件压缩包。你可以在本仓库的release中找到预训练模型
4. 把模型文件压缩包解压到`/ckpt`​​文件夹下。解压后，文件结构应当如下

    ```bash
    - ckpt
        - <model_name>
            - *.pth
            - config.yaml
            - vocab.yaml
    ```
5. （可选）把字典放入`/dictionary`​​文件夹中。默认为`opencpop-extension.txt`​​
6. 准备需要强制对齐的数据，放入一个文件夹中（默认放在`/segments`​​文件夹），格式如下

    ```bash
    - segments
            - 1.lab
            - 1.wav
            - 2.lab
            - 2.wav
            - ...
    ```

    如果有多个歌手，文件夹格式如下

    ```bash
    - segments
        - singer1
            - 1.lab
            - 1.wav
            - 2.lab
            - 2.wav
            - ...
        - singer2
            - 1.lab
            - 1.wav
            - 2.lab
            - 2.wav
            - ...
    ```

    **注意，singer文件夹内不可有其他文件夹。**
7. 命令行推理  
    需要指定模型所在的文件夹名称`model_name`​，待标注的数据集文件夹`segments_path`​（默认为`/segments`​），字典文件`dictionary_path`​（默认为`/dictionary/opencpop-extension.txt`​）
    如果.lab文件里是音素序列而不是单词序列，无需使用字典，请指定`-p`​或`--phoneme_mode`​。  
    除此之外，还支持指定推理时的一些参数，可以使用python infer.py -h查看。
 
    ```bash
    python infer.py model_name -s SEGMENTS_PATH [-d DICTIONARY_PATH/-p]
    ```

## 训练

由于目前的版本并不完善，计划在重构后再发布训练与微调说明

# todos

* 完善训练功能的README
* 完善config.yaml的说明
* 更改网络结构为conformer
* 增加微调功能
* 推理时输出每一条数据的置信度
* 增加textless alignment功能
