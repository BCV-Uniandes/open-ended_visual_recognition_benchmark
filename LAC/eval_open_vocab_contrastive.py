"""
Reference: https://github.com/bytedance/fc-clip/blob/main/fcclip/fcclip.py
"""

import argparse
import json
import os
from functools import partial

import cv2
import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import default_argument_parser, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    print_csv_format,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.memory import retry_if_cuda_oom
from osprey.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from osprey.conversation import SeparatorStyle, conv_templates
from osprey.mm_utils import tokenizer_image_token
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM
from osprey.train.train import DataArguments, preprocess_multimodal
from panopticapi.utils import rgb2id
from PIL import Image
from pycocotools import mask as maskUtils
from sentence_transformers import SentenceTransformer, util
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, GenerationConfig
from utils.accuracy_evaluation import AccuracyEvaluator
from utils.instance_evaluation import InstanceSegEvaluator
from utils.register_ade20k_instance import register_all_ade20k_instance
from utils.register_ade20k_panoptic import register_all_ade20k_panoptic
from utils.register_cityscapes_panoptic import register_all_cityscapes_panoptic

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]

ADE150_CAT_TEMPL = [
    "a {}.",
    "a {} in the scene.",
    "this is a {}.",
    "there is a {} in the scene.",
    "a photo of a {}.",
    "this is a photo of a {}.",
    "a photo of a {} in the scene.",
]


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    evaluator_list.append(AccuracyEvaluator(dataset_name, output_folder))
    # semantic segmentation
    if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    # panoptic segmentation
    if evaluator_type in [
        "coco_panoptic_seg",
        "ade20k_panoptic_seg",
        "cityscapes_panoptic_seg",
    ]:
        # if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    # Cityscapes
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "cityscapes_panoptic_seg":
        # if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        # if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
    # ADE20K
    if evaluator_type == "ade20k_panoptic_seg" or evaluator_type == "coco":
        evaluator_list.append(
            InstanceSegEvaluator(dataset_name, output_dir=output_folder)
        )
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class EvalOsprey(nn.Module):
    def __init__(self, model_path, bert_model, dataset, cfg):
        super().__init__()
        self.model_name = model_path
        self.dataset = dataset

        model_path = os.path.expanduser(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=True,
        )
        self.model = OspreyLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).cuda()

        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.image_processor = CLIPImageProcessor(
            do_resize=True,
            size={"shortest_edge": 512},
            resample=3,
            do_center_crop=True,
            crop_size={"height": 512, "width": 512},
            do_rescale=True,
            rescale_factor=0.00392156862745098,
            do_normalize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            do_convert_rgb=True,
        )

        for m in self.model.modules():
            m.tokenizer = self.tokenizer

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.float16, device="cuda")

        train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN)
        test_metadata = MetadataCatalog.get(cfg.DATASETS.TEST)
        if "instance" in dataset:
            with open(test_metadata.json_file, "r") as f:
                data = json.load(f)
            self.gt_labels = {image["id"]: [] for image in data["images"]}
            annotations = data["annotations"]
            for ann in annotations:
                self.gt_labels[ann["image_id"]].append(ann)
            # filter out images without annotations
            self.gt_labels = {k: v for k, v in self.gt_labels.items() if len(v) > 0}

        if dataset == "ade":
            self.len_data = 150
        elif dataset == "ade_instance":
            self.len_data = 100
        elif dataset == "coco_instance":
            self.len_data = 80
        elif dataset == "coco":
            self.len_data = 133
        else:
            self.len_data = 19
        self.test_metadata = test_metadata

        _, self.train_num_templates, self.train_class_names = (
            self.prepare_class_names_from_metadata(
                train_metadata, train_metadata, VILD_PROMPT
            )
        )
        (
            self.category_overlapping_mask,
            self.test_num_templates,
            self.test_class_names,
        ) = self.prepare_class_names_from_metadata(
            test_metadata, train_metadata, VILD_PROMPT
        )

        _, self.region_test_num_templates, self.region_test_class_names = (
            self.prepare_class_names_from_metadata(
                test_metadata, train_metadata, ADE150_CAT_TEMPL
            )
        )

        self.num_queries = 300

        region_class_sentences = self.region_test_class_names
        if "instance" in dataset:
            region_class_sentences = [
                x.split(",")[0] for x in test_metadata.thing_classes
            ]
        else:
            region_class_sentences = [
                x.split(",")[0] for x in test_metadata.stuff_classes
            ]
        # region_class_sentences = ["{:s}\n\n".format(x) for x in region_class_sentences]
        input_ids = self.tokenizer(
            region_class_sentences,
            padding=False,
            truncation=True,
        )["input_ids"]
        input_ids = [torch.tensor(i)[1:] for i in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=IGNORE_INDEX
        ).cuda()
        attention_mask = input_ids.ne(IGNORE_INDEX)
        attention_mask, inputs_embeds = (
            self.model.prepare_inputs_labels_for_contrastive_learning(
                torch.ones([len(region_class_sentences), 1]).cuda(),
                input_ids,
            )
        )
        attention_mask = attention_mask.bool()
        with torch.inference_mode():
            # with torch.amp.autocast(device_type="cuda"):
            outputs = self.model.model(
                input_ids=None,
                attention_mask=attention_mask,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden_states = outputs[0]
            # select index of the last valid token in each sequence
            last_token_index = (torch.sum(attention_mask, dim=1) - 1).long()
            self.class_sentence_embeddings = hidden_states[
                torch.arange(hidden_states.size(0), device=hidden_states.device),
                last_token_index,
            ].bfloat16()

    def prepare_class_names_from_metadata(self, metadata, train_metadata, PROMPT):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(", ", ",")
                # there can be multiple synonyms for single class
                x_ = x_.split(",")
                res.append(x_)
            return res

        # get text classifier
        try:
            # it includes both thing and stuff
            class_names = split_labels(metadata.stuff_classes)
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(
                set(test_class_names)
            )
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long
        )

        def fill_all_templates_ensemble(x_="", templates=None):
            res = []
            for x in x_:
                for template in templates:
                    res.append(template.format(x))
            return res, len(res) // len(templates)

        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(
                x, templates=PROMPT
            )
            templated_class_names += templated_classes
            # how many templates for current classes
            num_templates.append(templated_classes_num)
        class_names = templated_class_names
        # print("text for classification:", class_names)
        return category_overlapping_mask, num_templates, class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        (
            self.category_overlapping_mask,
            self.test_num_templates,
            self.test_class_names,
        ) = self.prepare_class_names_from_metadata(
            metadata, self.train_metadata, VILD_PROMPT
        )
        self.test_text_classifier = None
        return

    def get_text_classifier(self):
        if self.training:
            if self.train_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(
                        self.backbone.get_text_classifier(
                            self.train_class_names[idx : idx + bs], "cuda"
                        ).detach()
                    )
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(
                    text_classifier.shape[0] // len(VILD_PROMPT),
                    len(VILD_PROMPT),
                    text_classifier.shape[-1],
                ).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    text_classifier.append(
                        self.backbone.get_text_classifier(
                            self.test_class_names[idx : idx + bs], "cuda"
                        ).detach()
                    )
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(
                    text_classifier.shape[0] // len(VILD_PROMPT),
                    len(VILD_PROMPT),
                    text_classifier.shape[-1],
                ).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
            return self.test_text_classifier, self.test_num_templates

    @torch.no_grad()
    def forward(self, inputs):
        images = [x["image"].cuda() for x in inputs]
        images = ImageList.from_tensors(images)

        batch_cosine_scores = []
        processed_results = [{}]
        processed_results[-1]["descriptions"] = []

        if "instance" in self.dataset and inputs[0]["image_id"] not in self.gt_labels:
            return processed_results
        mask_for_pooling_raw, is_void_prob = self.get_gt_label(inputs)

        num = int(min(self.num_queries, self.num_queries - sum(is_void_prob[0]).item()))
        # num = min(self.num_queries, len(inputs[0]["segments_info"]))
        masks = [mask_for_pooling_raw[0][:num, ...].cuda().bfloat16()]

        last_source = dict()
        round_ids = 0

        img_path = inputs[0]["file_name"]

        init_inputs = get_init_inputs(
            img_path,
            self.image_processor,
            round_ids=round_ids,
            last_round_source=last_source,
        )

        last_source = init_inputs
        image = init_inputs["image"].unsqueeze(0).cuda().bfloat16()

        self.model.model.tokenizer = self.tokenizer
        self.model = self.model.bfloat16()

        with torch.inference_mode():
            # with torch.amp.autocast(device_type="cuda"):
            _, image_features_dict = self.model.encode_images(image)
            outputs_embeddings, _ = self.model.mask_extractor(
                image_features_dict, masks
            )
            outputs_embeddings = outputs_embeddings[0].bfloat16()

        outputs_embeddings = F.normalize(outputs_embeddings, dim=-1)
        self.class_sentence_embeddings = F.normalize(
            self.class_sentence_embeddings, dim=-1
        )
        region_cosine_scores = torch.matmul(
            outputs_embeddings, self.class_sentence_embeddings.T
        ) * (1 / self.model.temperature)
        # cosine_scores = util.cos_sim(outputs_embeddings, self.class_sentence_embeddings) * (1 / self.model.temperature)

        # final_cosine_scores = []
        # cur_idx = 0
        # for num_t in self.region_test_num_templates:
        #     final_cosine_scores.append(
        #         cosine_scores[:, cur_idx : cur_idx + num_t].max(-1).values
        #     )
        #     cur_idx += num_t

        # final_pred_logits = torch.stack(final_cosine_scores, dim=-1)
        # batch_cosine_scores.append(final_pred_logits)

        # region_cosine_scores = torch.concat(batch_cosine_scores, dim=0)
        out_vocab_cls_results = torch.zeros(
            [self.num_queries, self.len_data], dtype=torch.float
        ).to(region_cosine_scores.device)

        out_vocab_cls_results[:num, ...] = region_cosine_scores

        is_void_prob = is_void_prob.cuda()
        mask_pred_results = mask_for_pooling_raw.cuda()
        cls_results = out_vocab_cls_results.unsqueeze(dim=0)

        # This is used to filtering void predictions.
        mask_cls_probs = torch.cat(
            [cls_results.softmax(-1) * (1.0 - is_void_prob), is_void_prob], dim=-1
        )
        mask_cls_results = torch.log(mask_cls_probs + 1e-8)

        for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, inputs, images.image_sizes
        ):

            # semantic segmentation inference
            mask_cls_result = mask_cls_result.to(mask_pred_result)
            processed_results[-1]["pred_labels"] = mask_cls_result.max(1).indices

            if "instance" not in self.dataset:
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )
                processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                    mask_cls_result, mask_pred_result
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # instance segmentation inference
            instance_r = retry_if_cuda_oom(self.instance_inference)(
                mask_cls_result, mask_pred_result
            )
            processed_results[-1]["instances"] = instance_r

        return processed_results

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        num_classes = len(self.test_metadata.stuff_classes)
        keep = labels.ne(num_classes) & (scores > 0)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = (
                    pred_class
                    in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
                )
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < 0.8:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # if this is panoptic segmentation
        if "instance" not in self.dataset:
            num_classes = len(self.test_metadata.stuff_classes)
        else:
            num_classes = len(self.test_metadata.thing_classes)
        labels = (
            torch.arange(num_classes, device="cuda")
            .unsqueeze(0)
            .repeat(self.num_queries, 1)
            .flatten(0, 1)
        )
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(100, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        # if self.panoptic_on:
        keep = torch.zeros_like(scores_per_image).bool()
        for i, lab in enumerate(labels_per_image):
            keep[i] = (
                lab in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
            )

        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (
            mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)
        ).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def get_gt_label(self, x):
        if "instance" in self.dataset:
            output_mask = torch.zeros(
                1, self.num_queries, x[0]["height"], x[0]["width"]
            )
            is_void = torch.ones(1, self.num_queries, 1)
            annotations = self.gt_labels[x[0]["image_id"]]
            for i, ann in enumerate(annotations):
                segmentation = ann["segmentation"]
                if isinstance(segmentation, list):  # Polygon format
                    mask = np.zeros((x[0]["height"], x[0]["width"]), dtype=np.uint8)
                    for polygon in segmentation:
                        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], 1)
                elif isinstance(segmentation, dict):  # RLE format
                    if isinstance(segmentation["counts"], list):
                        segmentation = maskUtils.frPyObjects(
                            segmentation,
                            segmentation["size"][0],
                            segmentation["size"][1],
                        )
                    mask = maskUtils.decode(segmentation)

                output_mask[0, i] = torch.from_numpy(mask).float()
                is_void[0, i, 0] = 0
            return output_mask, is_void
        else:
            gt_label = np.array(Image.open(x[0]["pan_seg_file_name"]))
            gt_label = rgb2id(gt_label)
            gt_ids = np.unique(gt_label)
            gt_label = torch.from_numpy(gt_label)
            output_mask = torch.zeros(1, self.num_queries, *gt_label.shape)
            is_void = torch.ones(1, self.num_queries, 1)

            output_mask[0, 0] = (gt_label == 0).float()
            cnt = 0

            for i, id in enumerate(gt_ids):
                if i >= self.num_queries:
                    break
                if id == 0:
                    continue
                output_mask[0, cnt] = (gt_label == id).float()
                is_void[0, cnt, 0] = 0
                cnt += 1

            return output_mask, is_void


def get_init_inputs(
    img_path,
    processor,
    round_ids=0,
    last_round_source=None,
):
    if round_ids == 0:

        image = Image.open(img_path).convert("RGB")

        image = processor.preprocess(image, do_center_crop=False, return_tensors="pt")[
            "pixel_values"
        ][0]

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False
        ).squeeze(0)

    else:
        image = last_round_source["image"]

    data_dict = {}
    data_dict["image"] = image

    return data_dict


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    cfg = get_cfg()

    cfg.OUTPUT_DIR = "/media/SSD4/cigonzalez/output/stage3+contrastive_lr1e-5_bs16_wu10_cg1_all_data_contrastive"
    cfg.DATASETS.PROPOSAL_FILES_TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4

    if args.dataset == "ade":  # TE AMO CRIS ATT: NICO
        cfg.DATASETS.TRAIN = "openvocab_ade20k_panoptic_train"
        cfg.DATASETS.TEST = "openvocab_ade20k_panoptic_val"
        dataset_name = "openvocab_ade20k_panoptic_val"
    elif args.dataset == "cityscapes":
        cfg.DATASETS.TRAIN = "openvocab_cityscapes_fine_panoptic_train"
        cfg.DATASETS.TEST = "openvocab_cityscapes_fine_panoptic_val"
        dataset_name = "openvocab_cityscapes_fine_panoptic_val"
    elif args.dataset == "ade_instance":
        cfg.DATASETS.TRAIN = "ade20k_instance_train"
        cfg.DATASETS.TEST = "ade20k_instance_val"
        dataset_name = "ade20k_instance_val"
    elif args.dataset == "coco_instance":
        cfg.DATASETS.TRAIN = "coco_2017_train"
        cfg.DATASETS.TEST = "coco_2017_val"
        dataset_name = "coco_2017_val"
    elif args.dataset == "coco":
        cfg.DATASETS.TRAIN = "coco_2017_train_panoptic"
        cfg.DATASETS.TEST = "coco_2017_val_panoptic"
        dataset_name = "coco_2017_val_panoptic"
    else:
        raise NotImplementedError

    logger = setup_logger()

    data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = build_evaluator(cfg, dataset_name)
    evaluator.reset()

    # Set seed
    set_seed(42)

    osprey = EvalOsprey(args.model, args.bert, args.dataset, cfg)
    osprey.eval()

    logger.info("Start inference on {} batches".format(len(data_loader)))

    for idx, inputs in tqdm(enumerate(data_loader), total=len(data_loader)):
        outputs = osprey(inputs)
        if "pred_labels" in outputs[0]:
            inputs[0]["gt_labels"] = outputs[0]["pred_labels"]
        evaluator.process(inputs, outputs)

    results = evaluator.evaluate()
    print_csv_format(results)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--dataset", help="eval dataset type ade/cityscapes", default="ade"
    )
    parser.add_argument(
        "--model", help="path to osprey model", default="/path/to/osprey-7b"
    )
    parser.add_argument(
        "--bert", help="path to bert model", default="/path/to/all-MiniLM-L6-v2"
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
