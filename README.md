# TabPFN-TDC
TabPFNv2 predicting Absorption, Distribution, Metabolism, Excretion, and Toxicitys (ADMET) of Drugs in the Therapeutic Data Challenge (TDC).  

This work make use of the [TabPFNv2📝](https://www.nature.com/articles/s41586-024-08328-6) tabular foundation model using the 217 RDKit molecular descriptors as features.

For classification tasks, the fine-tuned version of TabPFNv2 ([📝](https://arxiv.org/abs/2507.03971)) is used, which was trained with real datasets from internet after the synthetic dataset pretraining.

See the [ADMET benchmark](https://tdcommons.ai/benchmark/admet_group/overview/) for more details about the challenge.



### Installation
```bash
conda create --prefix ./env python=3.12
conda activate ./env
pip install -r requirements.txt
```

### Usage
```bash
python tdc_submission.py
```

### Citation
TabPFNv2 paper: [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6)
TabPFNv2 Finetuned paper: [Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data](https://arxiv.org/abs/2507.03971)