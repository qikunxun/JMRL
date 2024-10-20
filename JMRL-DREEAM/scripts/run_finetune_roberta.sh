TYPE=$1
LOAD_DIR=$2
LAMBDA=$3
SEED=$4
python=python
NAME=${TYPE}_lambda${LAMBDA}

$python -u run.py --do_train \
--data_dir dataset/docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--display_name  ${NAME} \
--train_file train_annotated.json \
--dev_file dev.json \
--save_path ${NAME} \
--load_path ${LOAD_DIR} \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--lr_transformer 1e-6 \
--lr_added 3e-6 \
--evi_thresh 0.2 \
--evi_lambda ${LAMBDA} \
--warmup_ratio 0.1 \
--num_train_epochs 10.0 \
--seed ${SEED} \
--num_class 97