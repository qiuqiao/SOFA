# g2p Module Usage Instructions

## 1. Introduction

The g2p module is used for inferring from .lab files during run-time. It reads text from .lab files and then converts
the text into a phoneme sequence.

## 2. How to Use

When using infer.py for command-line inference, add the --g2p argument to specify the name of the g2p module (the
default is Dictionary, and it is not necessary to include the last "G2P" in the name).

Some g2p modules require additional parameters. For example, Dictionary G2P requires the extra parameter
--dictionary_path, which is used to specify the path of the dictionary file.

Different modules may need different parameters; for specific parameters, please refer to the explanations of each
module.

Example: If you wish to use Dictionary G2P, you could run infer.py like this:

```shell
python3 infer.py --g2p Dictionary --dictionary_path /path/to/dictionary.txt
```

Or, if you want to use Phoneme G2P, you could run infer.py like this:

```shell
python3 infer.py --g2p PhonemeG2P
```

## 3. List of Modules

### 3.1 Dictionary G2P

#### 3.1.1 Introduction

Dictionary G2P is a dictionary-based g2p module. It reads a sequence of words from a .lab file and converts the word
sequence into a phoneme sequence.

A SP phoneme will be automatically inserted between words, but not within words.

#### 3.1.2 Input Format

The input format is a .lab file containing only one line, which includes a sequence of words separated by spaces.

Example:

```text
I am a student
```

#### 3.1.3 Parameters

| Parameter Name  | Type   | Default | Description                                                                                                                                                                                                                                 |
|-----------------|--------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dictionary_path | string | None    | Path to the dictionary file. The format of the dictionary file is: each line contains an entry with a word and one or more phonemes separated by `\t`, and individual phonemes are separated by spaces. Example: `student\ts t uw d ah n t` |

### 3.2 Phoneme G2P

#### 3.2.1 Introduction

Phoneme G2P is a phoneme-based g2p module. It directly reads the sequence of phonemes from the .lab file and inserts a
SP phoneme between each phoneme.

#### 3.2.2 Input Format

The input format is a .lab file containing only one line, which includes a sequence of phonemes separated by spaces.
It's best not to include SP phonemes.

Example:

```text
ay ae m ah s t uw d ah n t
```

#### 3.2.3 Parameters

None

### 3.3 None G2P

#### 3.3.1 Introduction

None G2P is an empty g2p module. It will directly read the phoneme sequence from the .lab file and will not perform any
processing.

It can be viewed as a Phoneme G2P that does not insert SP phonemes.

#### 3.3.2 Input Format

The input format is a .lab file containing only one line, which includes a sequence of phonemes separated by spaces.

Example:

```text
SP ay SP ae m SP ah SP s t uw d ah n t SP
```

#### 3.3.3 Parameters

None
