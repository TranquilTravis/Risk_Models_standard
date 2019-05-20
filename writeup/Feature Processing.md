# Feature Processing

## Introduction

## Numerical Feature Processing

### 缺失值探索

### 单一值探索

### 单特征效果探索

### 分箱

- Best-KS
- ChiMerge
- Decision Tree

### Encoding





## Categorical Feature Encoding

### Label Encoding 

先来看一组特征：(优，良，一般，差)，对于这样的离散特征最简单有效的编码方式是什么样的呢？由于这一组特征所包含的是程度上由"好"到"坏"信息，所以可以使用如下编码方式:

```
encode_dict = {优：0，良：1，一般：2，差：3}
```

或者使用相似的编码字典。

对于类似上述，<u>有内在顺序的离散型特征，可以采用简单直观的label encoding (ordinal encoding)方式进行编码</u>。

Label encoding: 如果一个离散型特征共包含m个类别，将这m个类别分别映射到m个连续的自然数上(比如0~m-1)。

优点：简单，有效，可解释性强，不增加额外的特征纬度

缺点：a). 仅适用于有内在顺序的特征；b). 很难融入自动化的特征处理

### One-Hot Encoding



### Frequency Encoding



###Mean Target Encoding



### WOE Encoding



## Feature Engineering in Modeling



