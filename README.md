## Code and data for the paper "End-to-end Learning of Logical Rules for Document-level Relation Extraction"

### Prerequisites

 * Python 3.8
 * pytorch==2.0.0

More prerequisites can be found in the original repositories.

### Datasets
We use seven datasets in our experiments.

| Datasets           | Download Links (original)                |
|--------------------|------------------------------------------|
| DWIE               | https://github.com/klimzaporojets/DWIE   |
| DocRED             | https://github.com/thunlp/DocRED         |
| Re-DocRED          | https://github.com/bigai-nlco/DocGNRE    |
| DocGNRE            | https://github.com/bigai-nlco/DocGNRE    |

### Models
We use five models in our experiments.

| Models             | Code Download Links (original)                  |
|--------------------|-------------------------------------------------|
| LSTM               | https://github.com/thunlp/DocRED                |
| Bi-LSTM            | https://github.com/thunlp/DocRED                |
| GAIN               | https://github.com/DreamInvoker/GAIN            |
| ATLOP              | https://github.com/wzhouad/ATLOP                |
| DREEAM             | https://github.com/YoumiMa/dreeam               |

## Use examples

### JMRL-LSTM

Path for code: ``./JMRL-LSTM-BiLSTM``

The script for training on the DWIE dataset is:
```
python train.py --model_name LSTM --save_name checkpoint_LSTM --train_prefix dev_train --test_prefix dev_dev
```

The script for evaluation on the DWIE dataset is:
```
python3 test.py --model_name LSTM --save_name checkpoint_LSTM --train_prefix dev_train --test_prefix dev_dev --input_theta [theta]
```

### JMRL-BiLSTM

Path for code: ``./JMRL-LSTM-BiLSTM``

The script for training on the DWIE dataset is:
```
python train.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev
```

The script for evaluation on the DWIE dataset is:
```
python3 test.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev --input_theta [theta]
```

### JMRL-GAIN

Path for code: ``./JMRL-GAIN``

The script for training on the DWIE dataset is:
```
python -u train.py --train_set ../dataset_dwie/train_annotated.json --train_set_save ../dataset_dwie/prepro_data/train_BERT.pkl --dev_set ../dataset_dwie/dev.json --dev_set_save ../dataset_dwie/prepro_data/dev_BERT.pkl --test_set ../dataset_dwie/test.json --test_set_save ../dataset_dwie/prepro_data/test_BERT.pkl --use_model bert --model_name JMRL_GAIN_BERT_base_DWIE --lr 0.00002 --batch_size 4 --test_batch_size 4 --epoch 300 --test_epoch 1 --log_step 1 --save_model_freq 5 --negativa_alpha 4 --gcn_dim 808 --gcn_layers 2 --bert_hid_size 768 --bert_path ../PLM/bert-base-uncased --use_entity_type --use_entity_id --dropout 0.1 --activation relu --coslr
```

The script for evaluation on the DWIE dataset is:
```
python -u test.py --train_set ../dataset_dwie/train_annotated.json --train_set_save ../dataset_dwie/prepro_data/train_BERT.pkl --dev_set ../dataset_dwie/dev.json --dev_set_save ../dataset_dwie/prepro_data/dev_BERT.pkl --test_set ../dataset_dwie/test.json --test_set_save ../dataset_dwie/prepro_data/test_BERT.pkl --use_model bert --pretrain_model checkpoint/JMRL_GAIN_BERT_base_DWIE_best.pt --lr 0.00002 --batch_size 4 --test_batch_size 4 --epoch 300 --test_epoch 1 --log_step 1 --save_model_freq 5 --negativa_alpha 4 --gcn_dim 808 --gcn_layers 2 --bert_hid_size 768 --bert_path ../PLM/bert-base-uncased --use_entity_type --use_entity_id --dropout 0.1 --activation relu --coslr --input_theta [theta]
```

The script for training on the DOCRED dataset is:
```
python -u train.py --train_set ../dataset_docred/train_annotated.json --train_set_save ../dataset_docred/prepro_data/train_BERT.pkl --dev_set ../dataset_docred/dev.json --dev_set_save ../dataset_docred/prepro_data/dev_BERT.pkl --test_set ../dataset_docred/test.json --test_set_save ../dataset_docred/prepro_data/test_BERT.pkl --use_model bert --model_name JMRL_GAIN_BERT_base_DOCRED --lr 0.00002 --batch_size 4 --test_batch_size 4 --epoch 20 --test_epoch 1 --log_step 1 --save_model_freq 5 --negativa_alpha 4 --gcn_dim 808 --gcn_layers 2 --bert_hid_size 768 --bert_path ../PLM/bert-base-uncased --use_entity_type --use_entity_id --dropout 0.1 --activation relu --coslr
```

The script for evaluation on the DOCRED dataset is:
```
python -u test.py --train_set ../dataset_docred/train_annotated.json --train_set_save ../dataset_docred/prepro_data/train_BERT.pkl --dev_set ../dataset_docred/dev.json --dev_set_save ../dataset_docred/prepro_data/dev_BERT.pkl --test_set ../dataset_docred/test.json --test_set_save ../dataset_docred/prepro_data/test_BERT.pkl --use_model bert --pretrain_model checkpoint/JMRL_GAIN_BERT_base_DOCRED_best.pt --lr 0.00002 --batch_size 4 --test_batch_size 4 --epoch 20 --test_epoch 1 --log_step 1 --save_model_freq 5 --negativa_alpha 4 --gcn_dim 808 --gcn_layers 2 --bert_hid_size 768 --bert_path ../PLM/bert-base-uncased --use_entity_type --use_entity_id --dropout 0.1 --activation relu --coslr --input_theta [theta] 
```

### JMRL-ATLOP

Path for code: ``./JMRL-ATLOP``

The script for both training and evaluation on the DWIE dataset is:
```
python -u train.py --dataset dwie --transformer_type bert --model_name_or_path ../PLM/bert-base-uncased --train_file train_annotated.json --dev_file dev.json --test_file test.json --save_path ../trained_model/model_JMRL_ALTOP_DWIE.pth --num_train_epochs 300.0 --train_batch_size 4 --test_batch_size 4 --seed 66 --num_class 66 --tau 1.0 --lambda_al 1.0
```

The script for both training and evaluation on the DocRED dataset is:
```
python -u train.py --dataset docred --transformer_type bert --model_name_or_path ../PLM/bert-base-uncased --train_file train_annotated.json --dev_file dev.json --test_file test.json --save_path ../trained_model/model_JMRL_ALTOP_DOCRED.pth --num_train_epochs 20.0 --train_batch_size 4 --test_batch_size 4 --seed 66 --num_class 97 --tau 0.2 --lambda_al 1.0
```

### JMRL-DREEAM

Path for code: ``./JMRL-DREEAM``

1. The script for inferring on the distantly-supervised data:
```
bash scripts/infer_distant_roberta.sh ${name} ${load_dir} # for RoBERTa
```
where ``${name}`` is the logging name and ``${load_dir}`` is the directory that contains the checkpoint (Checkpoint for teacher model can be downloaded from: [Google drive](https://drive.google.com/file/d/1Frs8PZiBAoN2l2elZUgYVcejbxbo2dJz/view). The command will perform an inference run on ``train_distant.json`` and record token importance as train_distant.attns saved under ``${load_dir}``.
Note that you should alter model.py to model.py.bak for this step, as the teacher model do not involve rule reasoning.

2. The script for utilizing the recorded token importance as supervisory signals for the self-training of the student model:
```
bash scripts/run_self_train_roberta.sh ${name} ${teacher_signal_dir} ${lambda} ${seed}
```
where ``${name}`` is the logging name, ``${teacher_signal_dir}`` is the directory that stores the train_distant.attns file, ${lambda} is the scaler than controls the weight of evidence loss, and ``${seed}`` is the value of random seed.

3. The script for fine-tuning the model on human-annotated data:
```
bash scripts/run_finetune_roberta.sh ${name} ${student_model_dir} ${lambda} ${seed}
```
where ``${name}`` is the logging name and ``${student_model_dir}`` is the directory that stores the checkpoint of student model.

4. The script for testing on the dev set:
```
bash scripts/isf_roberta.sh ${name} ${model_dir} dev
```
where ``${name}`` is logging name and ``${model_dir}`` is the directory that contains the checkpoint we are going to evaluate. The commands have two functions:

a. Perform inference-stage fusion on the development data, return the scores and dump the predictions into ``${model_dir}/``;

b. Select the threshold and record it as ``${model_dir}/thresh``.

5. With ${model_dir}/thresh available, we can make predictions on test set:
```
bash scripts/isf_roberta.sh ${name} ${model_dir} test
```
where ``${model_dir}`` is the directory that contains the checkpoint we are going to evaluate. 

### Running script on DocGNRE
```
bash scripts/run_roberta_gpt.sh ${name} ${lambda} ${seed}
```
where ${name} is the logging name, ${lambda} is the scaler that controls the weight of evidence loss, and ${seed} is the value of random seed.

### Rule extraction

The script for rule extraction:
```
python rule_extraction.py [model_path] [beam_size]
```
where ``[model_path]`` is the path for the trained JMRL-enhanced model and ``[beam_size]`` is the beam size.

## Citation
Please consider citing the following paper if you find our codes helpful. Thank you!

```
@inproceedings{QiWDC22,
  author    = {Kunxun Qi and Jianfeng Du and Hai Wan },
  title     = {End-to-end Learning of Logical Rules for Document-level Relation Extraction},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational
               Linguistics},
  year      = {2024}
}
```