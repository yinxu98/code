model=$1
dataset=$2
gpu=$3

root=$PWD
run=${root}/scripts/pretrain_multiple.py
config=${root}/configs/${model}/
taskname=${model}_${dataset}_pretrain_scale
ls_scale=('1')

for scale in "${ls_scale[@]}"
do
    CUDA_VISIBLE_DEVICES=${gpu} python ${run} ${config}/${taskname}${scale}.py
done