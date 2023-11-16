#!/usr/bin/env

log_dir=./logs
save_dir=./checkpoints
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../omnifm_dr_module
data_dir=../../datasets/
data=${data_dir}/report_generation/train.tsv,${data_dir}/report_generation/val.tsv
restore_file=../../checkpoints/ofa_huge.pt
selected_cols=0,4,3,3,3

task=caption
arch=ofa_huge
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=1e-5
max_epoch=10
warmup_ratio=0.06
batch_size=2
update_freq=4
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=100
max_tgt_length=100
num_bins=1000
patch_image_size=512
eval_cider_cached=${data_dir}/val-words.p
drop_worst_ratio=0.2
log_tag=report_generation

for max_epoch in {${max_epoch},}; do
  echo "max_epoch "${max_epoch}
  for warmup_ratio in {0.06,}; do
    echo "warmup_ratio "${warmup_ratio}
    for drop_worst_after in {60000,}; do
      echo "drop_worst_after "${drop_worst_after}

      log_file=${log_dir}/${log_tag}_${max_epoch}"_"${warmup_ratio}"_"${drop_worst_after}".log"
      save_path=${save_dir}/${log_tag}_${max_epoch}"_"${warmup_ratio}"_"${drop_worst_after}
      mkdir -p $save_path
      cp ${0} ${save_path}/
	
	python ../../train.py \
          $data \
          --tensorboard-logdir=${save_path} \
          --data-negative=${data_negative} \
          --selected-cols=${selected_cols} \
          --bpe-dir=${bpe_dir} \
          --user-dir=${user_dir} \
          --restore-file=${restore_file} \
          --reset-optimizer --reset-dataloader --reset-meters \
          --save-dir=${save_path} \
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
          --weight-decay=0.01 \
          --optimizer=adam \
          --adam-betas="(0.9,0.999)" \
          --adam-eps=1e-08 \
          --clip-norm=1.0 \
          --lr-scheduler=polynomial_decay \
          --lr=${lr} \
          --max-epoch=${max_epoch} \
          --warmup-ratio=${warmup_ratio} \
          --log-format=simple --log-interval=2 \
          --fixed-validation-seed=7 \
          --no-epoch-checkpoints \
          --keep-best-checkpoints=1 \
          --save-interval=1 \
          --save-interval-updates=1000 \
          --max-src-length=${max_src_length} \
          --max-tgt-length=${max_tgt_length} \
          --find-unused-parameters \
          --add-type-embedding \
          --scale-attn \
          --scale-fc \
          --scale-heads \
          --disable-entangle \
          --num-bins=${num_bins} \
          --patch-image-size=${patch_image_size} \
          --drop-worst-ratio=${drop_worst_ratio} \
          --drop-worst-after=${drop_worst_after} \
          --fp16 \
          --fp16-scale-window=512 \
          --num-workers=0 \
          --distributed-port 17813 > ${log_file} 2>&1
    done
  done
done

