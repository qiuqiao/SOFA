# readme

[中文](README_zh.MD)

**Note: SOFA is currently in beta version and may have many bugs and unpredictable results. If you encounter any issues or have suggestions for improvement, please feel free to open an issue.**

# Introduction

SOFA: Singing-Oriented Forced Aligner is a forced aligner designed specifically for vocals, but it also supports alignment for non-vocal content.

It offers the following advantages:

* Easy installation

# Usage

## Inference

1. Clone the repository using `git clone`
2. Install the required Python environment
   1. Create a conda environment with Python version 3.10 (other versions might also work, but you can try them out yourself):
   ```
   conda create -n SOFA python=3.10 -y
   conda activate SOFA
   ```
   2. Install torch by visiting the [PyTorch website](https://pytorch.org/get-started/locally/) (tested with version 2.0.1, other versions might result in errors)
   3. Install other Python libraries by running `pip -r requirements.txt`
      If you are unable to access the internet, please use the following command:
      ```
      pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
      ```
3. Download the model file archive. You can find pre-trained models in the "release" section of this repository.
4. Extract the model file archive into the `/ckpt` folder. After extraction, the file structure should be as follows:

    ```bash
    - ckpt
        - <model_name>
            - *.pth
            - config.yaml
            - vocab.yaml
    ```
5. (Optional) Place the dictionary file in the `/dictionary` folder. The default file is `opencpop-extension.txt`.
6. Prepare the data for forced alignment and place it in a folder (by default, in the `/segments` folder) with the following format:

    ```bash
    - segments
        - 1.lab
        - 1.wav
        - 2.lab
        - 2.wav
        - ...
    ```

    If there are multiple singers, the folder format should be as follows:

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

    **Note that there should be no subfolders within the singer folder.**
7. Command-line inference  
   You need to specify the folder name where the model is located `model_name`, the dataset folder to be annotated `segments_path` (default is `/segments`), and the dictionary file `dictionary_path` (default is `/dictionary/opencpop-extension.txt`).
   If the .lab file contains phoneme sequences instead of word sequences, there is no need to use a dictionary. In that case, specify `-p` or `--phoneme_mode`.
   In addition, there are other optional parameters that can be specified during inference, which can be viewed by running `python infer.py -h`.

   ```bash
   python infer.py model_name -s SEGMENTS_PATH [-d DICTIONARY_PATH/-p]
   ```

## Training

Due to the current version's imperfections, I plan to release the training and fine-tuning instructions after refactoring.

# Todos

* Complete the README for training functionality
* Enhance the explanation of `config.yaml`
* Change the network architecture to conformer
* Add fine-tuning functionality
* Output confidence scores for each piece of data during inference
* Add textless alignment functionality
