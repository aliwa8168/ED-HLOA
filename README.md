# [ED-HLOA](https://github.com/aliwa8168/ED-HLOA)



In this study, we propose an **E**lite **D**ifferential mutation-based **H**orned **L**izard **O**ptimization **A**lgorithm (ED-HLOA) to optimize the learning rate and compression factor of the DenseNet-201 model, and we validated it on a self-constructed traditional Chinese medicine image dataset.

## [Data Summary](https://github.com/aliwa8168/ED-HLOA/tree/main/Chinese%20herbal%20medicine%20Datasets)



| Categories             | Image count |
| ---------------------- | ----------- |
| mint                   | 2020        |
| fritillaria cirrhosa   | 2020        |
| honeysuckle            | 2020        |
| ophiopogonis japonicum | 2020        |
| ginseng                | 2020        |

There are 5 classes in this dataset, namely mint, fritillaria cirrhosa, honeysuckle, ophiopogonis japonicum, and ginseng. Each category contains 2020 images with a total of 10100 images of size 224x224.

## [Sample dataset](https://github.com/aliwa8168/ED-HLOA/tree/main/Sample%20dataset)

It contains examples of the dataset

![](https://github.com/aliwa8168/ED-HLOA/blob/main/Sample%20dataset/chuanbeimu.jpg)

## Code Resources



Several key resources are provided for implementing and testing ED-HLOA:

### [train.py](https://github.com/aliwa8168/ED-HLOA/blob/main/train.py)



Model training files

### [Read_picture.py](https://github.com/aliwa8168/ED-HLOA/blob/main/Read_picture.py)



A file that reads images from the dataset and converts them to numpy arrays

### [ED-HLOA.py](https://github.com/aliwa8168/ED-HLOA/blob/main/ED-HLOA.py)



Code file for the ED-HLOA algorithm

### [model.py](https://github.com/aliwa8168/ED-HLOA/blob/main/model.py)



It contains the DenseNet-201 model architecture

## Environment configuration



Python 3.8 environment using Anaconda with the following configuration:

| library        | version  |
| -------------- | -------- |
| cuda           | 11.2     |
| cudnn          | 8.2      |
| tensorflow-gpu | 2.10     |
| keras          | 2.10     |
| scikit-learn   | 1.3.2    |
| opencv-python  | 4.9.0.80 |

The computational experiments were conducted on a system equipped with an Intel® Xeon® Gold 6142 CPU, operating at 2.60 GHz, and supplemented with 128.0 GB of internal storage. The system also featured an NVIDIA RTX A5000 GPU, which was utilized for accelerated processing tasks.
