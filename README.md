# Drug-Target Indication Prediction by Integrating End-to-End Learning and Fingerprints
This repository contains the accompanying code base and other ancillary files of the aforementioned study.
The dependencies could be found in [setup.py](./setup.py). The [PADME](https://github.com/simonfqy/PADME) project, 
was refactored to have the module name *padme* in this work.

The bash files found [here](./proj/dti) are used for model training and evaluation of the baseline and the IVPGAN models.
The bash files with the *padme_* prefix train the baseline models reflected in their name. 
For instance, *padme_cold_drug_gconv_cv_kiba* trains our implementation of the GraphConv-PSC model using k-fold
Cross-Validation with a cold drug splitting scheme on the KIBA dataset. The IVPGAN models are trained using
the bash files with *integrated_* prefix. They also follow the same naming pattern as the *padme_* files.

The bash file with *\_eval\_* in their names are used for evaluating a trained model. We use a resource tree
structure to aggregate all training and evaluation statistics which are then saved
 as JSON files for later analysis. For more on the resource tree structure, you can examine 
 [sim_data.py](./ivpgan/utils/sim_data.py) and its usage in [singleview.py](./proj/dti/singleview.py) and
 [train_joint_gan.py](./proj/dti/train_joint_gan.py). The performance data saved in a JSON file of 
 each evaluated model is analysed using [worker.py](proj/dti/analysis/worker.py).
