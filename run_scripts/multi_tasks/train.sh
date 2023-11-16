#!/usr/bin/env

log_dir=./logs
save_dir=./checkpoints
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../omnifm_dr_module

restore_file=../../checkpoints/ofa_huge.pt

### report and classification
data_dir=../../datasets
data=${data_dir}/report_classification/train.tsv
negative_data=${data_dir}/report_classification/train_negative.tsv

### localization
vg_data=../../datasets/localization/trian.tsv

### segmentation
seg_data=../../datasets/segmentation/trian.tsv

### attribute classification
attri_data=${data_dir}/vqa_attribute/train.tsv
attri_data_negative=${data_dir}/vqa_attribute/train_negative.tsv

selected_cols=0,4,2,3,5,6
vg_selected_cols=1,3,2,0,4
seg_selected_cols=0,3,1,2
attri_selected_cols=0,4,7

task=dr_joint_task
arch=ofa_huge
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.0
lr=1e-5
max_epoch=50
warmup_ratio=0.01
batch_size=2
update_freq=10
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=256
max_tgt_length=128
num_bins=1000
patch_image_size=512
sample_patch_num=196
max_image_size=512

save_path=./checkpoints

log_tag=multi_tasks
log_file=${log_dir}/${log_tag}_${max_epoch}"_"${warmup_ratio}_${lr}".log"
save_path=${save_dir}/${log_tag}_${max_epoch}"_"${warmup_ratio}_${lr}
mkdir -p $save_path
cp ${0} ${save_path}/

srun -p ${PARTITION} --mpi=pmi2 --gres=gpu:${GPU_NUM} -n${NODES} --ntasks-per-node=1 --cpus-per-task=${CPU_PER_TASKS} \
        --nodelist=${NODE119},${NODE47} --job-name=${task_name} --kill-on-bad-exit=1 \
        python3 ../../train.py \
        $data \
        --tensorboard-logdir=${save_path} \
        --report-negative-data=${negative_data} \
        --vg-data=${vg_data} \
        --seg-data=${seg_data} \
        --vqa-attritube-dataset-positive=${attri_data} \
        --vqa-attritube-dataset-negative=${attri_data_negative} \
        --selected-cols=${selected_cols} \
        --vg-selected-cols=${vg_selected_cols} \
        --seg-selected-cols=${seg_selected_cols} \
        --vqa-attritube-selected-cols=${attri_selected_cols} \
        --bpe-dir=${bpe_dir} \
        --user-dir=${user_dir} \
        --restore-file=${restore_file} \
        --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir=${save_path} \
        --neg-sample-dir=${neg_sample_dir} \
        --task=${task} \
        --arch=${arch} \
        --criterion=${criterion} \
        --label-smoothing=${label_smoothing} \
        --batch-size=${batch_size} \
        --update-freq=${update_freq} \
        --encoder-normalize-before \
        --decoder-normalize-before \
        --share-decoder-input-output-embed \
        --share-all-embeddings \
        --layernorm-embedding \
        --patch-layernorm-embedding \
        --code-layernorm-embedding \
        --resnet-drop-path-rate=${resnet_drop_path_rate} \
        --encoder-drop-path-rate=${encoder_drop_path_rate} \
        --decoder-drop-path-rate=${decoder_drop_path_rate} \
        --dropout=${dropout} \
        --attention-dropout=${attention_dropout} \
        --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=5.0 \
        --lr-scheduler=polynomial_decay --lr=${lr} \
        --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
        --log-format=simple --log-interval=10 \
        --fixed-validation-seed=7 \
        --keep-last-epochs=15 \
        --save-interval=1 \
        --save-interval-updates=1000 \
        --disable-validation \
        --max-src-length=${max_src_length} \
        --max-tgt-length=${max_tgt_length} \
        --add-type-embedding \
        --scale-attn \
        --scale-fc \
        --scale-heads \
        --disable-entangle \
        --num-bins=${num_bins} \
        --patch-image-size=${patch_image_size} \
        --sample-patch-num=${sample_patch_num} \
        --max-image-size=${max_image_size} \
        --fp16 \
        --fp16-scale-window=128 \
        --num-workers=0 \
        --ddp-backend=no_c10d \
        --distributed-port 17819 > ${log_file} 2>&1
