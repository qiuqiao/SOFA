# g2p模块使用说明

## 1. 简介

g2p模块是用于推理时，从.lab文件中读取文本，然后将文本转换为音素的模块。

## 2. 使用方法

在使用infer.py进行命令行推理时，添加--g2p参数，指定g2p模块名称（默认为Dictionary，可以省略名称最后的G2P）

有些g2p模块需要额外的参数。例如，Dictionary G2P需要额外指定参数--dictionary_path，用于指定字典文件的路径。

不同模块需要的参数不同，具体参数请参考各个模块的说明。

例：如果你想使用Dictionary G2P，你可以这样运行infer.py：

```shell
python3 infer.py --g2p Dictionary --dictionary_path /path/to/dictionary.txt
```

或：如果你想使用Phoneme G2P，你可以这样运行infer.py：

```shell
python3 infer.py --g2p PhonemeG2P
```

## 3. 模块列表

### 3.1 Dictionary G2P

#### 3.1.1 简介

Dictionary G2P是一个基于字典的g2p模块。它会从.lab文件中读取词语序列，然后将词语序列转换为音素序列。

在词语和词语之间，会自动插入SP音素，在词语之内，不会插入SP音素。

#### 3.1.2 输入格式

输入格式为.lab文件，仅有一行，内容为词语序列，词语之间用空格分隔。

例：

```text
I am a student
```

#### 3.1.3 参数

| 参数名             | 类型     | 默认值 | 说明                                                                                                 |
|-----------------|--------|-----|----------------------------------------------------------------------------------------------------|
| dictionary_path | string | 无   | 字典文件的路径。字典文件的格式为：每一行一个词条，每个词条包含一个单词和一个或多个音素，单词和音素之间用`\t`隔开，多个音素之间用空格隔开。例`student\ts t uw d ah n t` |

### 3.2 Phoneme G2P

#### 3.2.1 简介

Phoneme G2P是一个基于音素的g2p模块。它会直接从.lab文件中读取音素序列，并在每个音素之间插入SP音素。

#### 3.2.2 输入格式

输入格式为.lab文件，仅有一行，内容为音素序列，音素之间用空格分隔。最好不要有SP音素。

例：

```text
ay ae m ah s t uw d ah n t
```

#### 3.2.3 参数

无

### 3.3 None G2P

#### 3.3.1 简介

None G2P是一个空的g2p模块。它会直接从.lab文件中读取音素序列，并且不会进行任何处理。

你可以把它视为不插入SP音素的Phoneme G2P。

#### 3.3.2 输入格式

输入格式为.lab文件，仅有一行，内容为音素序列，音素之间用空格分隔。

例：

```text
SP ay SP ae m SP ah SP s t uw d ah n t SP
```

#### 3.3.3 参数

无
