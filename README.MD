# SOFA: Singing-Oriented Forced Aligner

English | [简体中文](README_zh.MD)

![example](example.png)

# Introduction

SOFA (Singing-Oriented Forced Aligner) is a forced alignment tool designed specifically for singing voice.

It has the following advantages:

* Easy to install

**Note: SOFA is still in beta and may contain many bugs, and effectiveness is not guaranteed. If any issues are encountered or improvements are suggested, please feel free to raise an issue.**

# How to Use

## Environment Setup

1. Use `git clone` to download the code from this repository
2. Install conda
3. Create a conda environment, requiring Python version `3.8`
    ```bash
    conda create -n SOFA python=3.8 -y
    conda activate SOFA
    ```
4. Go to the [pytorch official website](https://pytorch.org/get-started/locally/) to install torch
5. (Optional, to improve wav file reading speed) Go to the [pytorch official website](https://pytorch.org/get-started/locally/) to install torchaudio
6. Install other Python libraries
    ```bash
    pip install -r requirements.txt
    ```

## Inference

1. Download the model files. You can find the trained models in the [release section of this repository](github.com/qiuqiao/SOFA/releases), or you can find community-shared model download links in the [discussion section](github.com/qiuqiao/SOFA/discussions/categories/pretrained-model-sharing), with the files having a `.ckpt` extension.

2. Move the model file to the `/ckpt` folder.
3. Place the dictionary file in the `/dictionary` folder. The default dictionary is `opencpop-extension.txt`
4. Prepare the data for forced alignment and place it in a folder (by default in the `/segments` folder), with the following format
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
    Ensure that the `.wav` files and their corresponding `.lab` files are in the same folder.
5. Command-line inference

    Use `python infer.py` to perform inference.

    Parameters that need to be specified:
    - `--ckpt`: (must be specified) The path to the model weights;
    - `--folder`: The folder where the data to be aligned is stored (default is `segments`);
    - `--dictionary`: The dictionary file (default is `dictionary/opencpop-extension.txt`);

    ```bash
    python infer.py --ckpt checkpoint_name --folder segments_path --dictionary dictionary_path
    ```
### Advanced Features

   - Using a custom g2p instead of a dictionary
     - See [g2p module instructions](modules/g2p/readme_g2p.md)
   - In the matching mode, you can activate it by specifying `-m` during inference. It finds the most probable contiguous sequence segment within the given phoneme sequence, rather than having to use all the phonemes.

## Training

1. Follow the steps above for setting up the environment. It is recommended to install torchaudio for faster binarization speed;
2. Place the training data in the `data` folder in the following format:

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
    Where:

    `transcriptions.csv` only needs to have the correct relative path to the `wavs` folder;

    The `transcriptions.csv` in `weak_label` does not need to have a `ph_dur` column;
3. Modify `binarize_config.yaml` as needed, then execute `python binarize.py`;
4. Download the pre-trained model you need from releases, modify `train_config.yaml` as needed, then execute `python train.py -p path_to_your_pretrained_model`;
5. For training visualization: `tensorboard --logdir=ckpt/`.