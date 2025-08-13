# Multimodal-PopBias
This repository contains code and resources for the ACM WSDM 2026 submission "Content-Based Mitigation of Popularity Bias in Multimodal Recommender Systems".

### Data Splitting
To facilitate our analysis of popularity bias in multimodal recommender systems, we split the datasets into in-distribution (ID) and out-of-distribution (OOD) sets (see relevant [script](https://github.com/RecAnon/Multimodal-PopBias/blob/main/utils/ood_data_splitting.py)). We adapt our code structure for this process, and for the training of the item content encoder, from the [ColdRec repository](https://github.com/YuanchenBei/ColdRec). 

### Content Encoding
Once the data splitting has been performed, the contrastive content encoder can be trained with the command

`python ./item_encoding/run_encoding.py --dataset [DATASET-NAME]`

### Base Models
The [CF and multimodal base models](https://github.com/RecAnon/Multimodal-PopBias/tree/main/models) were trained using [MMRec](https://github.com/enoche/MMRec). We include the relevant config files used for the hyperparameter sweeps (based on the ranges in the original papers) in the [configs](https://github.com/RecAnon/Multimodal-PopBias/tree/main/configs) folder. We also provide our implementation of the in-process bias mitigation methods on TMLP. Full implementations for [SOIL](https://github.com/TL-UESTC/SOIL), [TMLP](https://github.com/jessicahuang0163/TMLP), and [GUME](https://github.com/NanGongNingYi/GUME) can be found in their respective repositories.

### Ensembling
The process to run the Raw-KNN/Encoded-KNN ensembling with the base models is contained in the [ensembling](https://github.com/RecAnon/Multimodal-PopBias/blob/main/utils/ensemble_calcs.py) script, as well as the scripts to run the PC and EDGE post-processing mitigation methods.
