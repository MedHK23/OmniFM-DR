# pip install g2p-en
# apt install libsndfile1
# pip install opencv-python-headless


import os
import cv2
import numpy
import sys
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from tasks.mm_tasks.refcoco import RefcocoTask
from tasks.mm_tasks.caption import CaptionTask
from tasks.mm_tasks.seg import SegTask
from tasks.mm_tasks.vqa_gen import VqaGenTask
from tasks.pretrain_tasks.joint_task import DRJointTask

from models.ofa import OFAModel
from PIL import Image
from torchvision import transforms
from PIL import Image, ImageOps
import gensim
from gensim import corpora

import base64
from io import BytesIO

CUDA_VISIBLE_DEVICES=6

PRIVATE_LABEL={
    'Atelectasis': '肺不张',
    'Cardiomegaly': '心影增大',
    'Edema': '肺水肿',
    'Pneumonia': '肺炎',
    'Pneumothorax': '气胸',
    'Nodule': '结节',
    'RibFracture': '肋骨骨折',
    'Effusion': '胸腔积液',
    'Impurity': '暂无',
    'Emphysema': '肺气肿',
    'FiberFoci': '纤维灶',
    'PleuralThickening': '胸膜增厚',
    'Tuberculosis': '肺结核',
    'HilarEnlargement': '肺门增大',
    'AbnormalAorticKnob': '纵隔病变',
}
generator = None
bos_item = None
eos_item = None
pad_idx = None
task = None
models = None
# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

# Image transform
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
patch_resize_transform = None

def init_task():
    global generator
    global bos_item
    global eos_item
    global pad_idx
    global task
    global models
    global patch_resize_transform
    
    target_task = 'refcoco'
    tasks.register_task(target_task, RefcocoTask)

    # specify some options for evaluation
    parser = options.get_generation_parser()
    input_args = ["", "--task=seg", \
        "--beam=5", "--path=checkpoints/ofa_large_384.pt", \
            "--bpe-dir=utils/BPE", "--no-repeat-ngram-size=3", "--patch-image-size=384"]

    checkpoint_base = '/mnt/lustre/niziyu/projects/OFA/run_scripts/caption/stage1_checkpoints'
    input_args[1] = f'--task={target_task}'
    input_args[3] = '--path=/data/niziyu/projects/DR/third_party/OFA_bak/run_scripts/pretraining/checkpoints/checkpoint_report_vg-0.3_vqa.pt'

    args = options.parse_args_and_arch(parser, input_args)
    cfg = convert_namespace_to_omegaconf(args)

    # Load pretrained ckpt & config
    task = tasks.setup_task(cfg.task)

    models, cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        task=task
    )

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)
    
    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((task.cfg.patch_image_size, task.cfg.patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    token_result = []
    bin_result = []
    img_result = []
    for token in x.strip().split():
      if token.startswith('<bin_'):
        bin_result.append(token)
      elif token.startswith('<code_'):
        img_result.append(token)
      else:
        if bpe is not None:
          token = bpe.decode('{}'.format(token))
        if tokenizer is not None:
          token = tokenizer.decode(token)
        if token.startswith(' ') or len(token_result) == 0:
          token_result.append(token.strip())
        else:
          token_result[-1] += token

    return ' '.join(token_result), ' '.join(bin_result), ' '.join(img_result)


def coord2bin(coords, w_resize_ratio, h_resize_ratio):
    coord_list = [float(coord) for coord in coords.strip().split()]
    bin_list = []
    bin_list += ["<bin_{}>".format(int(round(coord_list[0] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[1] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[2] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[3] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    return ' '.join(bin_list)


def bin2coord(bins, w_resize_ratio, h_resize_ratio):
    bin_list = [int(bin[5:-1]) for bin in bins.strip().split()]
    if len(bin_list) == 0 or len(bin_list) % 4 != 0:
        return None, False
    coord_list = []
    coord_list += [bin_list[0] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[1] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    coord_list += [bin_list[2] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[3] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    return coord_list, True


def bin2coord_seg(bins, w_resize_ratio, h_resize_ratio):
    bin_list = [int(bin[5:-1]) for bin in bins.strip().split()]
    # print(f'BINS: {len(bin_list)}')
    # print(f"Bin list: {bin_list}")
    coord_list = []
    for i in range(len(bin_list)):
        if i % 2 == 0:
            coord_list += [bin_list[i] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
        else:
            coord_list += [bin_list[i] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    return coord_list


def encode_text(text, length=None, append_bos=False, append_eos=False):
    line = [
      task.bpe.encode(' {}'.format(word.strip())) 
      if not word.startswith('<code_') and not word.startswith('<bin_') else word
      for word in text.strip().split()
    ]
    line = ' '.join(line)
    s = task.tgt_dict.encode_line(
        line=line,
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


def construct_sample(image: Image, instruction: str):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])

    instruction = encode_text(' {}'.format(instruction.lower().strip()), append_bos=True, append_eos=True).unsqueeze(0)
    instruction_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in instruction])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": instruction,
            "src_lengths": instruction_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        }
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def task_assgin(inst1, insts):
    best_sim = -1
    match_id = 0
    for i, inst2 in enumerate(insts):
        tokens_1 = gensim.utils.simple_preprocess(inst1)
        tokens_2 = gensim.utils.simple_preprocess(inst2)

        dictionary = corpora.Dictionary([tokens_1, tokens_2])

        corpus = [dictionary.doc2bow(tokens) for tokens in [tokens_1, tokens_2]]

        similarity = gensim.similarities.MatrixSimilarity(corpus)
        cosine_sim = similarity[corpus[0]][1]
        print(f'************** Consin sim: {cosine_sim}')
        if cosine_sim > best_sim:
            best_sim = cosine_sim
            match_id = i

    if best_sim <= 0.05:
        return -1, match_id
    # print('余弦相似度为:', cosine_sim)
    return best_sim, match_id


def resize_img(img, w, h, target=768):
    s = max(w, h)
    ratio = float(target) / s
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    img = cv2.resize(img, (new_w, new_h))
    return img

    
def ask_answer(image, instruction):
    
    detect_flag = False
    w, h = image.size
    w_resize_ratio = task.cfg.patch_image_size / w
    h_resize_ratio = task.cfg.patch_image_size / h

    report_inst = ' what can we get from this chest medical image? '
    detect_inst = ' Which region does the text describe? '
    vqa_inst = 'what disease does the image show?'
    inst_pool = [report_inst, detect_inst, vqa_inst]
    
    tasks = ['report', 'detect', 'vqa']
    task_sim, task_id = task_assgin(instruction, inst_pool)
    cur_instruction = inst_pool[task_id]
    cur_task = tasks[task_id]
    if cur_task == 'detect' or 'where' in instruction.lower() or 'locate' in instruction.lower() or 'find' in instruction.lower():
        detect_flag = True
        cur_instruction = " Which region does the text {} describe?"
        key_cls = list(PRIVATE_LABEL.keys())
        target = key_cls[0]
        exits = False
        for k in key_cls:
            if k.lower() in instruction.lower():
                target = k
                exits = True
                break
        if exits:
            cur_instruction = cur_instruction.format(target)
        else:
            cur_instruction = instruction
    else:
        if task_sim >= 0.8:
            cur_instruction = instruction
        
    if task_sim == -1 and not detect_flag:
        return 'Sorry, we can not answer your question. Please try other questions.'
    print(f"Current Inst: {cur_instruction, detect_flag}")
    # Construct input sample & preprocess for GPU if cuda available
    sample = construct_sample(image, cur_instruction)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    # Generate result
    with torch.no_grad():
        hypos = task.inference_step(generator, models, sample)
        tokens, bins, imgs = decode_fn(hypos[0][0]["tokens"], task.tgt_dict, task.bpe, generator)
        score = round(hypos[0][0]["score"].exp().item(), 2)
        # print("Score: ", score)
        # print(f"Tokens: {tokens}")
        # print(f"Bins:")
        # print(bins)
    
    tokens = tokens.replace('support devices', '')
    if len(tokens.split('.')) == 0 and not detect_flag:
        return 'Sorry, we can not answer your question. Please try other questions.'
    report = tokens.split('.')
    report = [r.strip().capitalize() for r in report]
    report = ". ".join(report)
    
    if detect_flag:
        img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        for b, color in zip([bins], [(0, 229, 238)]):
            coord_list, bin_flag = bin2coord(b, w_resize_ratio, h_resize_ratio)
            if not bin_flag:
                return 'Sorry, we can not answer your question. Please try other questions.'
            cv2.rectangle(
                img,
                (int(coord_list[0]), int(coord_list[1])),
                (int(coord_list[2]), int(coord_list[3])),
                color,
                8
            )
            # cv2.putText(img, f'result', (int(coord_list[0]), int(coord_list[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0),
            #                                 thickness=2)
        out_name = './results/tmp/detect.jpg'
        img = resize_img(img, w, h)
        cv2.imwrite(out_name, img)
        return out_name, report
    return report