# g2p模块使用说明

## 1. 简介

g2p模块是用于推理时，从`.lab`文件中读取文本，然后将文本转换为音素的模块。

## 2. 使用方法

在使用`infer.py`进行命令行推理时，添加`--g2p`参数，指定g2p模块名称（默认为Dictionary，可以省略名称的G2P后缀）

有些g2p模块需要额外的参数。例如，Dictionary G2P需要额外指定参数`--dictionary_path`，用于指定字典文件的路径。

不同模块需要的参数不同，具体参数请参考各个模块的说明。

例：如果你想使用Dictionary G2P，你可以这样运行`infer.py`：

```shell
python infer.py --g2p Dictionary --dictionary_path /path/to/dictionary.txt
```

或：如果你想使用Phoneme G2P，你可以这样运行`infer.py`：

```shell
python infer.py --g2p PhonemeG2P
```

## 3. 模块列表

### 3.1 Dictionary G2P

#### 3.1.1 简介

Dictionary G2P是一个基于字典的g2p模块。它会从`.lab`文件中读取词语序列，然后将词语序列转换为音素序列。

在词语和词语之间，会自动插入SP音素，在词语之内，不会插入SP音素。

#### 3.1.2 输入格式

输入的`.lab`文件仅有一行，内容为词语序列，词语之间用空格分隔。

例：

```text
I am a student
```

#### 3.1.3 参数

| 参数名             | 类型     | 默认值 | 说明                                                                                                                  |
|-----------------|--------|-----|---------------------------------------------------------------------------------------------------------------------|
| dictionary_path | string | 无   | 字典文件的路径。字典文件的格式为：每一行一个词条，每个词条包含一个单词和一个或多个音素，单词和音素之间用`\t`隔开，多个音素之间用空格隔开。有关示例，请参阅`dictionary/opencpop-extension.txt`。 |

### 3.2 Phoneme G2P

#### 3.2.1 简介

Phoneme G2P是一个基于音素的g2p模块。它会直接从`.lab`文件中读取音素序列，并在每个音素之间插入`SP`音素。

#### 3.2.2 输入格式

输入的`.lab`文件仅有一行，内容为音素序列，音素之间用空格分隔，最好不要包含`SP`音素。

例：

```text
ay ae m ah s t uw d ah n t
```

#### 3.2.3 参数

无

### 3.3 None G2P

#### 3.3.1 简介

None G2P是一个空的g2p模块。它会直接从`.lab`文件中读取音素序列，并且不会进行任何处理。

你可以把它视为不插入`SP`音素的Phoneme G2P。

#### 3.3.2 输入格式

输入的`.lab`文件仅有一行，内容为音素序列，音素之间用空格分隔。

例：

```text
SP ay SP ae m SP ah SP s t uw d ah n t SP
```

#### 3.3.3 参数

无

## 4. 自定义g2p模块

如果你想自定义g2p模块，你需要继承`base_g2p.py`中的`BaseG2P`类，并实现其中的`__init__`和`_g2p`方法。

`__init__`方法用于初始化模块，`_g2p`方法用于将文本转换为音素序列、词语序列以及两个序列之间的对应关系。

### 4.1 `__init__`方法

`__init__`方法的参数为`**kwargs`，由`infer.py`中的命令行参数`**g2p_args`传入。

如果你想使用额外的参数，你可以在`infer.py`中添加额外的命令行参数，然后在`__init__`方法中接收这些参数。

例：如果你想使用额外的参数`--my_param`，你可以在infer.py中添加这个参数：

```python
...


@click.option('--my_param', type=str, default=None, help='My parameter for my g2p module')
def main(ckpt, folder, g2p, match_mode, **kwargs):
    ...
```

然后在`__init__`方法中接收这个参数：

```python
class MyG2P(BaseG2P):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.my_param = kwargs['my_param']
        ...
```

**注意：g2p模块和AP_detector模块共用kwargs参数。**

### 4.2 `_g2p`方法

`_g2p`方法的参数为`text`，是一个字符串，表示从.lab文件中读取的文本。

`_g2p`方法的返回值为一个元组，包含三个元素：

- `ph_seq`：音素列表，SP是静音音素。第一个音素和最后一个音素应当为SP，并且任何位置都不能有连续两个以上的SP。
- `word_seq`：单词列表。
- `ph_idx_to_word_idx`：`ph_idx_to_word_idx[i] = j` 意味着第i个音素属于第j个单词。如果`ph_idx_to_word_idx[i] = -1`
  ，则第i个音素是一个静音音素。

示例：

```python
text = 'I am a student'
ph_seq = ['SP', 'ay', 'SP', 'ae', 'm', 'SP', 'ah', 'SP', 's', 't', 'uw', 'd', 'ah', 'n', 't', 'SP']
word_seq = ['I', 'am', 'a', 'student']
ph_idx_to_word_idx = [-1, 0, -1, 1, 1, -1, 2, -1, 3, 3, 3, 3, 3, 3, 3, -1]
```

### 4.3 使用

使用时，你可以这样运行`infer.py`：

```shell
python infer.py --g2p My --my_param my_value
```
