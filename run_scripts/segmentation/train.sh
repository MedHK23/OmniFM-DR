#!/usr/bin/env`

log_dir=./logs
save_dir=./checkpoints
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../omnifm_dr_module
data_dir=../../datasets/
data=${data_dir}/segmentation/train.tsv,${data_dir}/segmentation/val.tsv
restore_file=../../checkpoints/ofa_huge.pt

selected_cols=0,4,2,3
log_tag=segmentation

task=seg
arch=ofa_huge
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=1e-5
max_epoch=5
warmup_ratio=0.06  #0.06
batch_size=8
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=80
num_bins=1000
patch_image_size=512

for max_epoch in {10,}; do
  echo "max_epoch "${max_epoch}
  for lr in {1e-5,}; do
    for patch_image_size in {256,}; do
      echo "patch_image_size "${patch_image_size}

      log_file=${log_dir}/${log_tag}_${max_epoch}"_"${lr}"_"${patch_image_size}".log"
      save_path=${save_dir}/${log_tag}_${max_epoch}"_"${lr}"_"${patch_image_size}
      mkdir -p $save_path
    
      python ../../train.py \
          $data \
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
          --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
          --lr-scheduler=polynomial_decay --lr=${lr} \
          --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
          --log-format=simple --log-interval=10 \
          --fixed-validation-seed=7 \
          --no-epoch-checkpoints --keep-best-checkpoints=1 \
          --save-interval=1 --validate-interval=1 \
          --save-interval-updates=500 --validate-interval-updates=500 \
          --eval-acc \
          --eval-args='{"beam":5,"min_len":52,"max_len_a":0,"max_len_b":52}' \
          --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
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
          --fp16 \
          --fp16-scale-window=256 \
          --freeze-resnet \
          --num-workers=0 > ${log_file} 2>&1
    done
  done
done
