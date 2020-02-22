Active Bayesian Assessment for Black-Box Classifiers
===

This repo contains an implementation of the active Bayesian assessment models described in "Active Bayesian Assessment 
for Black-Box Classifiers", Disi Ji, Robert L. Logan IV, Padhraic Smyth, Mark Steyvers. [[arXiv]](https://arxiv.org/pdf/2002.06532.pdf).


Setup
---
You will need Python 3.5+. Dependencies can be installed by running:
```{bash}
pip install -r requirements.txt
```


Data
---
We assess performance characteristics of neural models on several standard image and text classification datasets. 
The image datasets we use are: CIFAR-100 (Krizhevsky and Hinton, 2009), SVHN (Netzer et al., 2011) and ImageNet (Russakovsky et al., 2015). 
The text datasets we use are: 20 Newsgroups (Lang, 1995) and DBpedia (Zhang et al., 2015). 

For image classification we use ResNet (He et al., 2016) architectures with either 110 layers (CIFAR-100) or 152 layers (SVHN and ImageNet). 
For ImageNet we use the pretrained model provided by PyTorch, and for CIFAR and SVHN we use the pretrained model checkpoints provided at: 
https://github.com/bearpaw/pytorch-classification. 
For text classification tasks we use fine-tuned BERTBASE (Devlin et al., 2019) models. These models were all
trained on standard training sets in the literature, independent from the datasets used for assessment.

The assessment datasets are based on standard test sets used for each dataset in the literature.
We build our assessment models on predictions that the classifiers made on the assessment datasets.
Predictions can be downloaded from [here](https://drive.google.com/drive/folders/1G7-9GGMxujtQ7W0eMnYZ2ZJNkJ8QQoSZ).


Run the Active Bayesian Assessor
---

After downloading the data, update `DATA_DIR`, `RESULTS_DIR` and `FIGURE_DIR` and `src/data_utils.py` accordingly, to specify the input directory to read data from 
and the output output directory to write results and figures to.

To reproduce all the experimental results and figures we reported in the paper, run commands in `script`. 

For example, to identify the extreme classes, navigate to `src` directory and run:
```{bash}
python active_learning_topk.py [dataset] 
    -metric [metric] 
    -mode [mode] 
    -topk [topk] 
    -pseudocount=[pseudocount]
```
where 
- `dataset`: name of the dataset. Path to the corresponding input data is hard-coded in `src/data_utils.py`.
- `metric`: 'accuracy' or 'calibration_error', the metric used to rank classes.
- `mode`: 'min' or 'max', to identify the most/least accurate/calibrated classes.
- `topk`: the number of extreme classes to identify.
-  `pseudocount`: the strength of the prior.

