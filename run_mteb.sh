#!/bin/bash

#SBATCH --job-name=cmteb # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=20G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 # number of gpus per node
#SBATCH -p pot # number of gpus per node
##SBATCH -w ccnl07
#SBATCH -o ./log/%x-%j.log # output and error log file names (%x for job id)

# pot-preempted

MODEL_NAME=colbert_110M
MODEL_PATH=/cognitive_comp/pankunhao/pretrained/chinese_colbert_60000
INDEX_PATH=./indexes
TASK_NAME=VideoRetrieval

python build_index.py \
    --model_name ${MODEL_NAME} \
    --model_path ${MODEL_PATH} \
    --index_path ${INDEX_PATH} \
    --task_name ${TASK_NAME}

python run_colbert_mteb_chinese.py \
    --model_name ${MODEL_NAME} \
    --model_path ${MODEL_PATH} \
    --index_path ${INDEX_PATH} \
    --task_name ${TASK_NAME}
