TYPE=$1
LAMBDA=$2
SEED=$3
python=~/miniconda3/bin/python3
NAME=${TYPE}_lambda${LAMBDA}

CUDA_VISIBLE_DEVICES=4 $python -u run.py --do_train \
--data_dir resource/ \
--transformer_type roberta \
--model_name_or_path roberta-large \
--display_name  ${NAME} \
--train_file ../resource/enhancement_data/re_docred_train_data_enhancement_more.json \
--dev_file ../resource/Re-DocRED/dev_revised.json \
--test_file ../resource/enhancement_data/re_docred_test_data_enhancement_human.json \
--save_path output/${NAME} \
--train_batch_size 8 \
--test_batch_size 16 \
--gradient_accumulation_steps 4 \
--num_labels 4 \
--lr_transformer 3e-5 \
--max_grad_norm 1.0 \
--evi_thresh 0.2 \
--evi_lambda ${LAMBDA} \
--warmup_ratio 0.06 \
--num_train_epochs 20.0 \
--seed ${SEED} \
--num_class 97
