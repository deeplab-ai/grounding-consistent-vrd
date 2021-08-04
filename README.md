# Grounding Consistency: Distilling Spatial Common Sense for Precise Visual Relationship Detection
### [deeplab.ai](https://deeplab.ai)

### Markos Diomataris, Nikolaos Gkanatsios, Vassilis Pitsikalis, Petros Maragos
 Using our **Grounding Consistency Distillation**
Loss we are able to counter the bias of relationships' context by leveraging grounding constraints on 
 predicted relations of unlabeled object pairs.
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

After training, testing is automatically performed and micro/macro Recal@[20, 50, 100] is printed for both constrained
and unconstrained scenarios while also calculating zero-shot results.
To produce metrics for precision (mP+ in paper) add the option `--test_on_negatives` on step 3. 

Checkpointing is performed so re-running step 3 for an already trained model will simply perform testing.

# VRD Tasks
These are the different scenarios you can test the Recall of a Scene Graph Generator depending on what information
is given during training.

| Task        | Objects' boxes| Objects' Categories| Interacting Objects |
| ------------- |:-------------:|:-------------:|:-------------:|
| Predicate Detection (PredDet)      | Yes | Yes | Yes | 
| Predicate Classification (PredCls) | Yes | Yes | No |
| Scene Graph Classification (SGCls) | Yes | No | No |
| Scene Graph Generation (SGGen)     | No | No | No |
| Phrase Detection (PhrDet)*          | No | No | No |

* Difference with SGGen is that we calculate IoU based on phrase box (minimum box containing subject/object)
  instead of object boxes.
  
In this repository we provide the opion to train on PredDet(default) or PredCls. Use the option `--task=<preddet|predcls>`

Feel free to contact us for any issues :)