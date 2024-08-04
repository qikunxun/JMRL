NAME=$1
MODEL_DIR=$2
SPLIT=$3
python=~/miniconda3/bin/python3
CUDA_VISIBLE_DEVICES=3 $python -u run.py --data_dir dataset/docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--display_name  ${NAME} \
--load_path ${MODEL_DIR} \
--eval_mode single \
--test_file ${SPLIT}.json \
--test_batch_size 8 \
--num_labels 4 \
--evi_thresh 0.2 \
--num_class 97 \
--data_dir resource/
