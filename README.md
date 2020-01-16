# ANSMESC

Source code about our paper:
Yuanzhi Ke, Masafumi Hagiwara: “A Non-image-based Subcharacter-level Method to Encode the Shape of Chinese Characters,” 人工知能学会論文誌 35 (2), pp. 1-11, 2020, (Accepted).

## About Demo Data

We provide the preprocessed data of our experiments introduced in the paper as demo data.

You can download it at 

The example data is a subset of tokenized Rakuten Ichiba Data provided by Rakuten.Inc. (https://rit.rakuten.co.jp/data_release/)

Each line is review text, where each word is replaced with the index in the vocab. NO information about the shop and the reviewer is concluded.

We are sorry but the vocab and raw text is NOT provided due to the license of the dataset.

## Requirements

- numpy==1.17.3 (newer maybe ok, but not checked)
- keras==2.1.6
- tensorflow==1.9.0 (Cannot run under tensorflow 2.x)
- ipython(=jupyter notebook)==7.9.0 (newer maybe ok, but not checked)
- ipykernel==5.1.3 (Maybe need for jupyter notebook)

## Usage

Please check experimentExample.ipynb
