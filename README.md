# Grounding Consistency: Distilling Spatial Common Sense for Precise Visual Relationship Detection
### [deeplab.ai](https://deeplab.ai)

### Markos Diomataris, Nikolaos Gkanatsios, Vassilis Pitsikalis, Petros Maragos
 Using our Grounding Consistency Distillation
Loss we are able to counter the bias of relationships' context by leveraging grounding constraints on unlabeled object pairs.
This semi-supervised method is model agnostic. This is the repository containing code for reproducing our ICCV 2021 paper.

![alt text](.readme_figs/teaser_new.png)

[comment]: <> (![alt text]&#40;.readme_figs/GCD.png&#41;)




## Environment Setup
After cloning the repository run (while sourcing it) `setup.sh`, this will create and activate a python 3.8 environment called **vrd** using conda 
and will install all required packages.

`. ./setup.sh`

## Dataset Setup
Download VRD and VG200 by running `main_prerequisites.py`. Pass as arguments `VRD` (default) and/or `VG200` to set up the
appropriate supplementary training/testing files.

`python main_prerequisites.py`


## Train
Training has three steps:
1. Train grounding network

`python main_research.py --model=parsing_net --dataset=<VRD|VG200>`

2. Train the teacher network

`python main_research.py --model=atr_net --net_name=atr_teacher_net --teacher=parsing_net --dataset=<VRD|VG200>
--bg_perc=0.2`

The `--bg_perc` option signifies the percentage of unlabeled samples used while training.

3. Train a model using the teacher to distill knowledge

`python main_research.py --model=<model_name> --teacher=atr_teacher_net --dataset=<VRD|VG200>
--bg_perc=0.2`

# Testing

After training, testing is automatically performed.
To produce metrics for precision (mP+ in paper) add the option `--test_on_negatives` on step 3. 