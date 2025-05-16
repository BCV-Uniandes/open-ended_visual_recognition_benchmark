# Benchmarking Open-Ended Visual Recognition

## Installation

### Requirements

1. Clone this repository and navigate to OPAL folder
```
git clone https://github.com/CircleRadon/Osprey.git
cd OPAL
```
2. Install packages
```
conda create -n opal python=3.10 -y
conda activate opal
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Dataset Preparation

### Training Dataset

# Dataset Preparation

Osprey-724K 🤗 [download](https://huggingface.co/datasets/AntGroup-MI/Osprey-724K)

| Data | Size |
| --- | ---: |
| osprey_short_form.json | 57 MB |
| osprey_conversation.json |  106 MB |
| osprey_detail_description.json | 63.4 MB |
| osprey_part_level.json | 153 MB |
| osprey_lvis_positive_negative.json | 140 MB |


- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip), `imgs` should contain all the images including training set and validation set.
- pascal_part: [train.json](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/pascalpart_train.json?download=true), [VOCdevkit](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar).
- partImagenet: [train_format.json](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/partImagenet_train_format.json?download=true),
[PartImageNet_OOD](https://drive.google.com/file/d/19kA8-pAxssQI0GD5H8y8KESGaALwChtx/view?usp=sharing).
- refcocos: [refcoco](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/finetune_refcoco_train_with_mask.json?download=true), [refcoco+](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/finetune_refcoco%2B_train_with_mask.json?download=true).
- vg: [vg_train_with_mask.json](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/vg_train_with_mask.json?download=true) (mask is generated from [HQ-SAM](https://github.com/SysCV/sam-hq)), images can be downloaded from [OpendataLab](https://opendatalab.com/OpenDataLab/Visual_Genome_Dataset_V1_dot_2), `image` should contain all the vg images(VG_100K and VG_100K_2).
- vcr: [vcr](https://visualcommonsense.com/download/).

After downloading all of them, organize the data as follows in `./data`,


```
├── coco
│   ├── annotations
│   │   └── instances_train2017.json
│   └── imgs
├── part data
│   ├── pascal_part
│   │   ├── train.json
│   │   └── VOCdevkit
│   └── partImagenet
│       ├── train_format.json
│       └── train
├── refcocos
│   ├── finetune_refcoco_train_with_mask.json
│   └── finetune_refcoco+_train_with_mask.json
├── Osprey-724K
│   ├── osprey_short_form.json
│   ├── osprey_conversation.json
│   ├── osprey_detail_description.json
│   ├── osprey_part_level.json
│   └── osprey_lvis_positive_negative.json
├── vg
│   ├── vg_train_with_mask.json
│   └── image
└── vcr
    ├── train.jsonl
    └── vcr1images
```

### Evaluation Datasets
A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  ADEChallengeData2016/
  cityscapes/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

## Expected dataset structure for [cityscapes](https://www.cityscapes-dataset.com/downloads/):
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # below are generated Cityscapes panoptic annotation
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Note: to create labelTrainIds.png, first prepare the above structure, then run cityscapesescript with:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```
These files are not needed for instance segmentation.

Note: to generate Cityscapes panoptic dataset, run cityscapesescript with:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createPanopticImgs.py
```
These files are not needed for semantic and instance segmentation.


## Expected dataset structure for [ADE20k](http://sceneparsing.csail.mit.edu/):
```
ADEChallengeData2016/
  images/
  annotations/
  objectInfo150.txt
  # download instance annotation
  annotations_instance/
  # generated by prepare_ade20k_sem_seg.py
  annotations_detectron2/
  # below are generated by prepare_ade20k_pan_seg.py
  ade20k_panoptic_{train,val}.json
  ade20k_panoptic_{train,val}/
  # below are generated by prepare_ade20k_ins_seg.py
  ade20k_instance_{train,val}.json
```

The directory `annotations_detectron2` is generated by running `python datasets/prepare_ade20k_sem_seg.py`.

Install panopticapi by:
```bash
pip install git+https://github.com/cocodataset/panopticapi.git
```

Download the instance annotation from http://sceneparsing.csail.mit.edu/:
```bash
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
```

Then, run `python datasets/prepare_ade20k_pan_seg.py`, to combine semantic and instance annotations for panoptic annotations.

And run `python datasets/prepare_ade20k_ins_seg.py`, to extract instance annotations in COCO format.

### Train setup:

- **Stage1: Image-Text Alignment Pre-training**
  - The pretrained projector weights for Convnext-large-CLIP can be found in [projector weights](https://huggingface.co/sunshine-lwt/osprey-v1.0-mlp2x-512px-convnext-pretrain-vicuna-7b-v1.5/tree/main).

- **Stage2: Mask-Text Alignment Pre-training**
  - Download [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main).
  - Download projector weights trained in stage1: [projector weights](https://huggingface.co/sunshine-lwt/osprey-v1.0-mlp2x-512px-convnext-pretrain-vicuna-7b-v1.5/tree/main).
  - Set `model_name_or_path` in `stage2.sh` to the path of `vicuna-7b-v1.5`.
  - Set `pretrain_mm_mlp_adapter` in `stage2.sh` to the path of `mm_projector`.
  - Set `vision_tower` in `stage2.sh` to the path of [Convnext-large-CLIP-model](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/blob/main/open_clip_pytorch_model.bin).
  - Run `sh scripts/stage2.sh`.

- **Stage3: End-to-End Fine-tuning**

  - Set `model_name_or_path` in `stage2.sh` to the path of `stage2 checkpoint`.
  - Set `vision_tower` in `stage2.sh` to the path of [Convnext-large-CLIP-model](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/blob/main/open_clip_pytorch_model.bin).
  - Run `sh scripts/stage3.sh`.

### Test setup:

Para correr la evaluación de LACE, debe correr la evaluación con el comando 

```CUDA_VISIBLE_DEVICES=0 python -m ALA.eval_open_vocab_seg_full_metric_set \
    --dataset {dataset} \
    --model {model_name} \
    --model_outputs_path {descriptions_json_path} \
    --semantic_relationship_file_path {semantic_relationship_json_file_path} \
    --num-gpus 1 \
```
From the root directory of this repository, you can find an example to run the evaluation of OPAL on Cityscapes in [ALA/eval_ala.sh](https://github.com/BCV-Uniandes/open-ended_visual_recognition_benchmark/blob/main/ALA/eval_ala.sh). The output descriptions from the models are available [here](https://github.com/BCV-Uniandes/open-ended_visual_recognition_benchmark/tree/main/outputs), and the semantic relationships for ADE20K and Cityscapes can be found [here](https://github.com/BCV-Uniandes/open-ended_visual_recognition_benchmark/tree/main/ALA/semantic_relationships).

If you want to see the results of our user study, the json file can be found in [this path](https://github.com/BCV-Uniandes/open-ended_visual_recognition_benchmark/tree/main/user_study_results).

## Pretrained model

To reproduce all our results as reported bellow, you can use our [pretrained model](https://drive.google.com/drive/folders/14_X53LXUkznjrdGtPZgKtE-k1BtmbrN8?usp=share_link) and our source code.
