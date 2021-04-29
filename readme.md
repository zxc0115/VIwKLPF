# Introduction
This is the source code for the paper **Variational Inference with KL Preserving Flow** 
## Environment
The developed environment is listed in below
* OS : Ubuntu 16.04
* CUDA : 9.0
* Nvidia Driver : 384.13
* Python 3.6
* Pytorch 0.4.0

The related python packages are listed in `requirements.txt`.
## Train text generation task
* There are three datasets folders chosse which task you want to run
```
$ cd [task]
```
In every dataset folders, there are two model folders
* if you want to train KLPF-VAE
```
$ cd KLPF
$ python3 KLPF_main.py
```
* if you want to train KLPF-VAE w/ AIR, Skip
```
$ cd KLPFwas
$ python3 KLPFwas_main.py
```
## Dialogue generation task
```
$ cd DailyDial
```
### Train
- Use pre-trained Word2vec
  Download Glove word embeddings `glove.twitter.27B.200d.txt` from https://nlp.stanford.edu/projects/glove/ and save it to the `./data` folder. The default setting use 200 dimension word embedding trained on Twitter.

- Modify the arguments at the top of `train.py`

- Train model by
  ```
    python train.py 
  ```
The logs and temporary results will be printed to stdout and saved in the `./output` path.

### Evaluation
Modify the arguments at the bottom of `sample.py`
    
Run model testing by:
```
    python sample.py
```
The outputs will be printed to stdout and generated responses will be saved at `results.txt` in the `./output` path.