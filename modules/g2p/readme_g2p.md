# g2p Module Documentation

## 1. Introduction

The g2p module is used for inference. It reads text from `.lab` files and then converts the text into a phoneme
sequence.

## 2. Usage

When using `infer.py` for command-line inference, add the `--g2p` argument to specify the g2p module name (default is
Dictionary, and the G2P suffix in the name can be omitted).

Some g2p modules require additional parameters. For instance, the Dictionary G2P requires the `--dictionary_path`
parameter to specify the path to the dictionary file.

Different modules require different parameters; for specifics, please refer to the documentation for each module.

Example: if you want to use Dictionary G2P, you would run `infer.py` like this:

```shell
python infer.py --g2p Dictionary --dictionary_path /path/to/dictionary.txt
```

Or: if you want to use Phoneme G2P, you would run `infer.py` like this:

```shell
python infer.py --g2p PhonemeG2P
```

## 3. Module List

### 3.1 Dictionary G2P

#### 3.1.1 Introduction

Dictionary G2P is a dictionary-based g2p module. It reads a sequence of words from a `.lab` file and then converts this
sequence of words into a sequence of phonemes.

Between words, the SP phoneme is automatically inserted, but it is not inserted within words.

#### 3.1.2 Input Format

The input `.lab` file only has one line, which contains a sequence of words separated by spaces.

Example:

```text
I am a student
```

#### 3.1.3 Parameters

| Parameter Name  | Type   | Default | Description                                                                                                                                                                                                                                                     |
|-----------------|--------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dictionary_path | string | none    | Path to the dictionary file. The format of the dictionary file: each line contains a word entry with a word and one or more phonemes. Words and phonemes are separated by `\t`, and phonemes by spaces. See `dictionary/opencpop-extension.txt` for an example. |

### 3.2 Phoneme G2P

#### 3.2.1 Introduction

Phoneme G2P is a phoneme-based g2p module. It reads a phoneme sequence directly from the `.lab` file and inserts a `SP`
phoneme between each phoneme.

#### 3.2.2 Input Format

The input `.lab` file only has one line, which contains a phoneme sequence separated by spaces, preferably not including
the 'SP' phoneme.

Example:

```text
ay ae m ah s t uw d ah n t
```

#### 3.2.3 Parameters

None

### 3.3 None G2P

#### 3.3.1 Introduction

None G2P is a null g2p module. It reads a phoneme sequence directly from the `.lab` file and does not process it.

It can be regarded as Phoneme G2P without inserting `SP` phonemes.

#### 3.3.2 Input Format

The input `.lab` file only has one line, which contains a phoneme sequence separated by spaces.

Example:

```text
SP ay SP ae m SP ah SP s t uw d ah n t SP
```

#### 3.3.3 Parameters

None

## 4. Custom g2p Module

If you wish to create a custom g2p module, you need to inherit from the `BaseG2P` class in `base_g2p.py` and implement
the `__init__` and `_g2p` methods.

The `__init__` method is for initializing the module, and the `_g2p` method is for converting text into a phoneme
sequence, word sequence, and the mapping between the two sequences.

### 4.1 `__init__` Method

The `__init__` method takes `**kwargs` as its parameters, which are passed from the command-line arguments `**g2p_args`
in `infer.py`.

If you want to use additional parameters, you can add them in `infer.py` and then receive these parameters in
the `__init__` method.

Example: If you want to use an additional parameter `--my_param`, you would add it in infer.py like this:

```python
...


@click.option('--my_param', type=str, default=None, help='My parameter for my g2p module')
def main(ckpt, folder, g2p, match_mode, **kwargs):
    ...
```

Then receive this parameter in the `__init__` method:

```python
class MyG2P(BaseG2P):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.my_param = kwargs['my_param']
        ...
```
**Note: The g2p module and the AP_detector module share the kwargs parameters.**

### 4.2 `_g2p` Method

The `_g2p` method takes `text` as a parameter, which is a string representing the text read from a .lab file.

The return value of the `_g2p` method is a tuple containing three elements:

- `ph_seq`: A list of phonemes, with SP as the silence phoneme. The first and last phonemes should be `SP`, and there should not be more than two consecutive `SP`s at any position.
- `word_seq`: A list of words.
- `ph_idx_to_word_idx`: `ph_idx_to_word_idx[i] = j` means that the ith phoneme belongs to the jth word. If
  `ph_idx_to_word_idx[i] = -1`, then the ith phoneme is a silence phoneme.

Example:

```python
text = 'I am a student'
ph_seq = ['SP', 'ay', 'SP', 'ae', 'm', 'SP', 'ah', 'SP', 's', 't', 'uw', 'd', 'ah', 'n', 't', 'SP']
word_seq = ['I', 'am', 'a', 'student']
ph_idx_to_word_idx = [-1, 0, -1, 1, 1, -1, 2, -1, 3, 3, 3, 3, 3, 3, 3, -1]
```

### 4.3 Usage

When in use, you can run `infer.py` like this:

```shell
python infer.py --g2p My --my_param my_value
```
