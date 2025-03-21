# Deep Compressed Sensing for Terahertz Ultra-Massive MIMO Channel Estimation
This repository contains code associated with the paper "Deep Compressed Sensing for Terahertz Ultra-Massive MIMO Channel Estimation" published in IEEE Open Journal of the Communications Society. 

## Environment
- MATLAB R2022a or higher: [install from official site](https://www.mathworks.com/products/matlab.html)
- Python: see `environment.yml` and `requirements.txt` or run `create-conda-env.sh` to generate a conda environment under `/env`

## Dataset Generation
To generate THz dataset from TeraMIMO, do
1. Add `src/TeraMIMO-main` to your MATLAB path
2. Run `src/genreate_thz_channel.m`. This will generate channel dataset in `.mat` format under `data/`.
3. Run `/data/mat2pt.py` to convert `.mat` files to `.pt` files for Pytorch usage. 
4. Repeat the above steps to generate separate files for training and testing sets. 

## Training and Testing
- Run `dcs-train.py`/`gan-train.py` to train the DCS/GAN model. Checkpoints will be saved under `/ckpt`. 
- Run `dcs-test.py`/`gan-test.py` to test the DCS/GAN model. Output will be NMSE for various SNR. 

## Citing the Paper

If you find the code in this repository useful for your work, we kindly request that you cite the following paper:


```plaintext
@ARTICLE{10899780,
  author={Lin, Ganghui and Erdem, Mikail and Alouini, Mohamed-Slim},
  journal={IEEE Open Journal of the Communications Society}, 
  title={Deep Compressed Sensing for Terahertz Ultra-Massive MIMO Channel Estimation}, 
  year={2025},
  volume={6},
  number={},
  pages={1747-1762},
  keywords={Training;Terahertz communications;Generative adversarial networks;Computational modeling;Channel estimation;Accuracy;Neural networks;Dictionaries;Matching pursuit algorithms;Transmission line matrix methods;Channel estimation;generative neural network;Terahertz;ultra-massive MIMO},
  doi={10.1109/OJCOMS.2025.3544871}}


```
