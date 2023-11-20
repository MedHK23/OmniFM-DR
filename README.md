# OmniFM-DR

## News
We now provide a pretrained OmniFM-DR on June 9, 2023! 

## Online Demo
Click the image to have a try with OmniFM-DR around the chest DR images

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://479d81e5a2e14b9538.gradio.live"><img width="600px" height="auto" src="https://github.com/MedHK23/OmniFM-DR/blob/main/resources/demo.gif"></a>
</div>


## Key Features

This repository provides the official implementation of OmniFM-DR

key feature bulletin points here
- First multi-modality model for multi-task analysis of chest DR image
- The largest full labeled chest DR dataset
- Supoort 4 tpyes of downstream tasks
    - Classification
    - Disease Localization
    - Segmentation
    - Report Generation

## Links

- [Paper](https://arxiv.org/abs/2311.01092)
- [Model](https://huggingface.co/MedHK23/OmniFM-DR)
- [Dataset](https://huggingface.co/datasets/MedHK23/OmniFM-Dr)


<!-- give a introduction of your project -->
## Details

 We have built a multimodal multitask model for DR data, aiming to solve all tasks in this field with one model, such as report generation, disease detection, disease question answering, and even segmentation. Without any fine-tuning, our model has achieved satisfactory results in report generation, disease detection and question answering.

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://847656a535c7a29317.gradio.live/"><img width="1000px" height="auto" src="https://github.com/MedHK23/OmniFM-DR/blob/main/resources/diagram.png"></a>
</div>


## Dataset Links

We utilize 10 public and 6 private datasets for pre-training and provide the download via the following links:

Public dataset: 

- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/)
- [VinDR](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data)
- [ChestX-Det-Dataset](https://github.com/Deepwise-AILab/ChestX-Det-Dataset)
- [ChestX-ray14]( https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [CheXpert]( https://stanfordmlgroup.github.io/competitions/chexpert/)
- [TBX11K](https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified)
- [object-CXR]( https://github.com/hlk-1135/object-CXR)
- [JSRT Database]( http://db.jsrt.or.jp/eng.php)
- [Shenzhen chest X-ray Set]( https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)
- [Montgomery County chest X-ray Set]( https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)

## Get Started

**Main Requirements**  

- python 3.7.4
- pytorch 1.8.1
- torchvision 0.9.1
- gradio 3.34.0


**Installation**
```bash
git clone https://github.com/MedHK23/OmniFM-DR.git
pip install -r requirements.txt
```


**Training**
```bash
### before training, please download the pretrained models and datasets and place them in their respective folders.
bash ./run_scripts/multi_tasks/train.sh
```


**Testing**
```bash
from demo_base import init_task, ask_answer
from PIL import Image

print('Initializing Chat')
init_task()
print('Initialization Finished')

instruction = 'describe this image'
image = Image.open('test.png').convert('RGB')
report = ask_answer(image, instruction)
```


## 🛡️ License

This project is under the Apache License. See [LICENSE](LICENSE.txt) for details.

## 🙏 Acknowledgement

A lot of code is modified from [OFA](https://github.com/OFA-Sys/OFA).

## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
{xu2023learning,
      title={Learning A Multi-Task Transformer Via Unified And Customized Instruction Tuning For Chest Radiograph Interpretation}, 
      author={Lijian Xu and Ziyu Ni and Xinglong Liu and Xiaosong Wang and Hongsheng Li and Shaoting Zhang},
      year={2023},
      eprint={2311.01092},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

