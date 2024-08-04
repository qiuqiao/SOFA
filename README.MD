# SOFA: Singing-Oriented Forced Aligner

English | [简体中文](README_zh.MD)

![example](example.png)

# Introduction

SOFA (Singing-Oriented Forced Aligner) is a forced alignment tool designed specifically for singing voice.

On singing data, SOFA has the following advantages over MFA (Montreal Forced Aligner):

* Easier installation
* Better performance
* Faster inference speed

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

1. Download the model file. You can find the trained models in the [pretrained model sharing category of the discussion section](https://github.com/qiuqiao/SOFA/discussions/categories/pretrained-model-sharing), with the file extension `.ckpt`.
2. Place the dictionary file in the `/dictionary` folder. The default dictionary is `opencpop-extension.txt`
3. Prepare the data for forced alignment and place it in a folder (by default in the `/segments` folder), with the following format
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

    The `.lab` file is the transcription for the `.wav` file with the same name. The file extension for the transcription can be changed using the `--in_format` parameter.

    After the transcription is converted into a phoneme sequence by the `g2p` module, it is fed into the model for alignment.

    For example, when using the `DictionaryG2P` module and the `opencpop-extension` dictionary by default, if the content of the transcription is: `gan shou ting zai wo fa duan de zhi jian`, the `g2p` module will convert it based on the dictionary into the phoneme sequence `g an sh ou t ing z ai w o f a duan d e zh ir j ian`. For how to use other `g2p` modules, see [g2p module usage instructions](modules/g2p/readme_g2p.md).

4. Command-line inference

    Use `python infer.py` to perform inference.

    Parameters that need to be specified:
    - `--ckpt`: (must be specified) The path to the model weights;
    - `--folder`: The folder where the data to be aligned is stored (default is `segments`);
    - `--in_format`: The file extension of the transcription (default is `lab`);
    - `--out_formats`: The annotation format of the inferred files, multiple formats can be specified, separated by commas (default is `TextGrid,htk,trans`).
    - `--save_confidence`: Output confidence scores.
    - `--dictionary`: The dictionary file (default is `dictionary/opencpop-extension.txt`);

    ```bash
    python infer.py --ckpt checkpoint_path --folder segments_path --dictionary dictionary_path --out_formats output_format1,output_format2...
    ```
5. Retrieve the Final Annotation

   The final annotation is saved in a folder, the name of which is the annotation format you have chosen. This folder is located in the same directory as the wav files used for inference.

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
    Regarding the format of `transcriptions.csv`, see: https://github.com/qiuqiao/SOFA/discussions/5
    
    Where:

    `transcriptions.csv` only needs to have the correct relative path to the `wavs` folder;

    The `transcriptions.csv` in `weak_label` does not need to have a `ph_dur` column;
3. Modify `binarize_config.yaml` as needed, then execute `python binarize.py`;
4. Download the pre-trained model you need from releases, modify `train_config.yaml` as needed, then execute `python train.py -p path_to_your_pretrained_model`;
5. For training visualization: `tensorboard --logdir=ckpt/`.

## Evaluation (for model developers)

To measure the performance of a model, it is useful to calculate some objective evaluation metrics between the predictions (force-aligned labels) and the targets (manual labels), especially in a k-fold cross-validation.

Some useful metrics are:

- Boundary Edit Distance: the total moving distance from the predicted boundaries to the target boundaries.
- Boundary Edit Ratio: the boundary edit distance divided by the total duration of target intervals.
- Boundary Error Rate: the proportion of misplaced boundaries to all target boundaries under a given tolerance of distance.

To evaluate your model on a specific dataset, please first run the inference to get all predictions. You should put your predictions and targets in different folders, with same filenames and relative paths, containing the same phone sequences except for spaces. The script only supports TextGrid format currently.

Run the following command:

```bash
python evaluate.py <PRED_DIR> <TARGET_DIR> -r -s
```

where `PRED_DIR` is a directory containing all predictions and `TARGET_DIR` is a directory containing all targets.

Options:
- `-r`, `--recursive`: compare the files in subdirectories recursively
- `-s`, `--strict`: use strict mode (raise errors instead of skipping if the phones are not identical)
- `--ignore`: ignore some phone marks (default: `AP,SP,<AP>,<SP>,,pau,cl`)

The script will calculate:

- The boundary edit ratio
- The boundary error rate, under 10ms, 20ms and 50ms tolerance
