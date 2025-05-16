"""
Reference: https://github.com/bytedance/fc-clip/blob/main/fcclip/fcclip.py

Usage:
--data_path: path of refcoco annotation. 
--image_path:  path of refcoco images. 
--answers-file: path of output result.

Example:
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m gpt4roi.eval_panoptic \
    --dataset cityscapes \
    --num-gpus 4
"""

import argparse
import copy
import json
import logging
import os
from collections import OrderedDict
from functools import partial

import detectron2.utils.comm as comm
import inflect
import nltk
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk import pos_tag, word_tokenize

# Download necessary data
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
import numpy as np
import spacy
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.memory import retry_if_cuda_oom
from fvcore.common.config import CfgNode
from nltk.tokenize import sent_tokenize, word_tokenize
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import SeparatorStyle, conv_templates
from osprey.mm_utils import tokenizer_image_token
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM
from osprey.train.train import DataArguments, preprocess_multimodal
from panopticapi.utils import rgb2id
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

from .utils.accuracy_evaluation import AccuracyEvaluator
from .utils.evaluator import DatasetEvaluatorDict
from .utils.instance_evaluation import InstanceSegEvaluator
from .utils.openseg_classes import ADE20K_150_CATEGORIES, CITYSCAPES_CATEGORIES
from .utils.register_ade20k_panoptic import register_all_ade20k_panoptic
from .utils.register_cityscapes_panoptic import register_all_cityscapes_panoptic
from .utils.visualizer import Visualizer

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

ADE150_CAT_TEMPL = ["There is a {} in the scene."]


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
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
        evaluator_list.append(AccuracyEvaluator(dataset_name))
        # semantic segmentation
        if evaluator_type in [
            "sem_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
        ]:
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
            evaluator_list.append(
                COCOPanopticEvaluator(dataset_name, output_folder)
            )

        # ADE20K
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
        ]:
            evaluator_list.append(
                InstanceSegEvaluator(dataset_name, output_dir=output_folder)
            )

        model_name_stats = cfg.OUTPUT_DIR.split("/")[-1]
        evaluator = DatasetEvaluatorDict(
            {
                s: DatasetEvaluators(copy.deepcopy(evaluator_list))
                for s in cfg.TEST.VALID_SEMANTIC_RELATIONSHIPS
            },
            cfg
        )

        return evaluator

    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #     """
    #     Create evaluator(s) for a given dataset.
    #     This uses the special metadata "evaluator_type" associated with each
    #     builtin dataset. For your own dataset, you can simply create an
    #     evaluator manually in your script and do not have to worry about the
    #     hacky if-else logic here.
    #     """

    #     # cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    #     # cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True


    #     if output_folder is None:
    #         output_folder = os.path.join(cfg.OUTPUT_DIR, dataset_name)
    #     evaluator_list = []
    #     evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    #     # Accuracy
    #     evaluator_list.append(
    #         AccuracyEvaluator(
    #             dataset_name,
    #             distributed=True,
    #             ),
    #         )

    #     # semantic segmentation
    #     if evaluator_type in [
    #         "sem_seg",
    #         "ade20k_panoptic_seg",
    #         "entityseg_panoptic_seg",
    #         "mapillary_vistas_panoptic_seg",
    #         # "cityscapes_panoptic_seg",
    #     ]:

    #         evaluator_list.append(
    #             SemSegEvaluator(
    #                 dataset_name,
    #                 distributed=True,
    #                 output_dir=output_folder,
    #             )
    #             )
    #     # panoptic segmentation
    #     if evaluator_type in [
    #         # "coco_panoptic_seg",
    #         "ade20k_panoptic_seg",
    #         "cityscapes_panoptic_seg",
    #         "mapillary_vistas_panoptic_seg",
    #         "entityseg_panoptic_seg",
    #     ]:

    #         evaluator_list.append(
    #             COCOPanopticEvaluator(
    #                 dataset_name, output_folder, open_metric=False
    #             )
    #         )

    #     # instance segmentation
    #     if evaluator_type in [
    #         "coco",
    #         "ade20k_panoptic_seg",
    #         "entityseg_panoptic_seg",
    #         "mapillary_vistas_panoptic_seg",
    #         # "cityscapes_panoptic_seg",
    #     ]:
    #         evaluator_list.append(
    #             COCOEvaluator(
    #                 dataset_name,
    #                 output_dir=output_folder,
    #             )
    #         )
    #     # COCO
    #     if (
    #         evaluator_type == "coco_panoptic_seg"
    #     ):
    #         evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

    #     # Cityscapes
    #     if evaluator_type == "cityscapes_instance":
    #         assert (
    #             torch.cuda.device_count() > comm.get_rank()
    #         ), "CityscapesEvaluator currently do not work with multiple machines."
    #         return CityscapesInstanceEvaluator(dataset_name)
    #     if evaluator_type == "cityscapes_sem_seg":
    #         assert (
    #             torch.cuda.device_count() > comm.get_rank()
    #         ), "CityscapesEvaluator currently do not work with multiple machines."
    #         return CityscapesSemSegEvaluator(dataset_name)
    #     if evaluator_type == "cityscapes_panoptic_seg":
    #         if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
    #             assert (
    #                 torch.cuda.device_count() > comm.get_rank()
    #             ), "CityscapesEvaluator currently do not work with multiple machines."
    #             evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
    #         if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
    #             assert (
    #                 torch.cuda.device_count() > comm.get_rank()
    #             ), "CityscapesEvaluator currently do not work with multiple machines."
    #             evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))

    #     # LVIS
    #     if evaluator_type == "lvis":
    #         evaluator_list.append(LVISEvaluator(dataset_name, output_dir=output_folder))
    #     if len(evaluator_list) == 0:
    #         raise NotImplementedError(
    #             "no Evaluator for the dataset {} with the type {}".format(
    #                 dataset_name, evaluator_type
    #             )
    #         )
    #     elif len(evaluator_list) == 1:
    #         return evaluator_list[0]


    #     evaluator = DatasetEvaluatorDict(
    #         cfg, 
    #         {
    #             s: DatasetEvaluators(copy.deepcopy(evaluator_list))
    #             for s in cfg.TEST.VALID_SEMANTIC_RELATIONSHIPS
    #         }
    #     )


    #     return evaluator

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):

            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)

            if comm.is_main_process():
                # save results in json format
                with open(
                    os.path.join(cfg.OUTPUT_DIR, "inference", "results.json"), "w"
                ) as f:
                    json.dump(results_i, f)

            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(dataset_name)
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


class EvalOsprey(nn.Module):
    def __init__(self, model_path, dataset, sentences, cfg):
        super().__init__()
        self.model_name = model_path
        self.dataset = dataset
        self.sentences = sentences

        train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN)
        test_metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.test_metadata = test_metadata

        if dataset == "ade":
            self.len_data = 150
        else:
            self.len_data = 19
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
        with open(cfg.TEST.SEMANTIC_RELATIONSHIPS_FILE, "r") as f:
            self.semantic_relationships = json.load(f)
        if dataset == "ade":
            self.dataset_categories = [c["name"] for c in ADE20K_150_CATEGORIES]
        else:
            self.dataset_categories = [c["name"] for c in CITYSCAPES_CATEGORIES]
        self.possible_semantic_relationships = cfg.TEST.SEMANTIC_RELATIONSHIPS
        self.valid_semantic_relationships = cfg.TEST.VALID_SEMANTIC_RELATIONSHIPS
        self.output = {}

        for method, file in cfg.TEST.OUTPUT_FILES:
            with open(file, "r") as f:
                output = json.load(f)
            self.output[method] = {x["image_id"]: x["descriptions"] for x in output}
        
        self.tagger = SequenceTagger.load("pos")
        self.human_labels = json.load(open(cfg.TEST.HUMAN_LABELS_FILE, "r"))

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
        image_list = ImageList.from_tensors(images)
        mask_for_pooling_raw, is_void_prob = self.get_gt_label(inputs)
        is_void_prob = (1 - is_void_prob).bool().squeeze()
        mask_pred_results = mask_for_pooling_raw.cuda()[:, is_void_prob, :, :]

        outputs = self.get_output(inputs)
        mask_cls_results = [{k: {} for k in self.output.keys()}]
        processed_results = {
            m: {k: [] for k in self.valid_semantic_relationships}
            for m in self.output.keys()
        }

        for k in self.output.keys():
            segments_info = inputs[0]["segments_info"]
            # sort segments_info by id
            segments_info = sorted(segments_info, key=lambda x: x["id"])
            targets = [s["category_id"] for s in segments_info]
            mapping, nouns = self.map_to_dataset_categories(
                outputs[k], targets, sentences=self.sentences
            )
            processed_results[k]["descriptions"] = outputs[k]
            processed_results[k]["nouns"] = nouns
            mask_cls_results[0][k].update(mapping)

        for (
            mask_cls_result,
            mask_pred_result,
            input_per_image,
            image_size,
        ) in zip(
            mask_cls_results,
            mask_pred_results,
            inputs,
            image_list.image_sizes,
        ):
            for m in mask_cls_result.keys():
                for k in self.valid_semantic_relationships:
                    processed_results[m][k].append({})
                    mask_cls_result[m][k] = mask_cls_result[m][k].to(mask_pred_result)

                    # accuracy inference
                    processed_results[m][k][-1]["pred_labels"] = (
                        mask_cls_result[m][k].max(1).indices
                    )

                    try:
                        # semantic segmentation inference
                        r = retry_if_cuda_oom(self.semantic_inference)(
                            mask_cls_result[m][k], mask_pred_result
                        )
                        processed_results[m][k][-1]["sem_seg"] = r
                    except:
                        breakpoint()


                    # panoptic segmentation inference
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result[m][k], mask_pred_result
                    )
                    processed_results[m][k][-1]["panoptic_seg"] = panoptic_r

                    # instance segmentation inference
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result[m][k], mask_pred_result
                    )
                    processed_results[m][k][-1]["instances"] = instance_r

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
        # if self.panoptic_on:
        num_classes = len(self.test_metadata.stuff_classes)
        # else:
        #     num_classes = len(self.test_metadata.thing_classes)
        labels = (
            torch.arange(num_classes, device="cuda")
            .unsqueeze(0)
            .repeat(self.num_queries, 1)
            .flatten(0, 1)
        )

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            min(100, scores.flatten(0, 1).shape[0]), sorted=False
        )
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

    def get_output(self, x):
        return {k: self.output[k][x[0]["image_id"]] for k in self.output.keys()}

    def singularize(self, noun):
        if noun == "":
            return noun
        p = inflect.engine()
        is_plural = [t[1] in ["NNS", "NNPS"] for t in pos_tag(word_tokenize(noun))]
        singular = [
            p.singular_noun(n) if pl and isinstance(p.singular_noun(n), str) else n
            for n, pl in zip(noun.split(" "), is_plural)
        ]
        singular = " ".join(singular)
        # Check if the singular form is a string
        if not isinstance(singular, str):
            singular = noun
        return singular

    def map_to_dataset_categories(self, output, targets, sentences):
        def _match_category(self, c, semantic_relationships):
            if c in semantic_relationships:
                return (semantic_relationships[c]["id"], "exact")
            else:
                for r in self.possible_semantic_relationships[1:]:
                    for d in semantic_relationships.keys():
                        if c in semantic_relationships[d][r]:
                            return (semantic_relationships[d]["id"], r)
            return (-1, "NIV")

        def _get_match(self, noun, category, semantic_relationships):
            for r in self.possible_semantic_relationships:
                if noun.lower().strip() in semantic_relationships[category][r]:
                    return r
            return "NIV"

        nlp = spacy.load("en_core_web_sm")
        if not sentences:
            nouns = [self.singularize(o) for o in output]
            hyponyms = ["" for _ in output]
            meronyms = [[] for _ in output]
        if sentences:
            nouns = []
            hyponyms, meronyms = [], []
            for o in output:
                tags = Sentence(o)
                self.tagger.predict(tags)
                # tags = pos_tag(word_tokenize(o))
                o = nlp(o)

                hyponym, meronym_list = "", []
                tokens = [t for t in o]
                text_tokens = [t.text for t in tokens]
                if "of" in text_tokens:
                    idx = text_tokens.index("of")
                    # find pattern a <word> of <word> or <word> of <word>
                    if (
                        tokens[idx - 1].dep_ != "pobj"
                        and tokens[idx - 1].text != "full"
                        and (
                            idx - 1 >= 0
                            and tokens[0].dep_ == "det"
                            and tokens[0].text == "a"
                        )
                    ):
                        hyponym = tokens[idx + 1 :]
                        hyponym = [
                            t
                            for t in hyponym
                            if t.dep_ == "pobj" or t.dep_ == "compound"
                        ]
                        # remove non-consecutive tokens
                        for i, t in enumerate(hyponym):
                            if i > 0:
                                if t.i != hyponym[i - 1].i + 1:
                                    hyponym = hyponym[:i]
                                    break
                        hyponym = " ".join([t.text for t in hyponym])
                        hyponym = self.singularize(hyponym)
                    # find pattern <word> of a <word>
                    if (
                        tokens[idx - 1].dep_ != "pobj"
                        and tokens[idx - 1].text != "made"
                        and tokens[idx - 1].text != "full"
                        # and tokens[0].dep_ == "det"
                    ):
                        meronym = tokens[idx + 1 :]
                        meronym = [
                            t
                            for t in meronym
                            if t.dep_ == "pobj"
                            or t.dep_ == "compound"
                            or t.dep_ == "pcomp"
                        ]
                        # remove non-consecutive tokens
                        for i, t in enumerate(meronym):
                            if i > 0:
                                if t.i != meronym[i - 1].i + 1:
                                    meronym = meronym[:i]
                                    break
                        if len(meronym) > 1:
                            meronym = [
                                t
                                for i, t in enumerate(meronym)
                                if tags[i].tag not in ["JJ", "JJR", "JJS"]
                            ]
                        meronym = " ".join([t.text for t in meronym])
                        meronym = self.singularize(meronym)
                        meronym_list.append(meronym)
                if "on" in text_tokens:
                    idx = text_tokens.index("on")
                    # find pattern <word> on the <word> or <word> on <word>
                    if (
                        len(tokens)
                        > idx + 1
                    ):
                        meronym = tokens[idx + 1 :]
                        meronym = [
                            t
                            for t in meronym
                            if t.dep_ == "pobj"
                            or t.dep_ == "compound"
                            or t.dep_ == "pcomp"
                        ]
                        # remove non-consecutive tokens
                        for i, t in enumerate(meronym):
                            if i > 0:
                                if t.i != meronym[i - 1].i + 1:
                                    meronym = meronym[:i]
                                    break
                        # remove adjectives
                        if len(meronym) > 1:
                            meronym = [
                                t
                                for i, t in enumerate(meronym)
                                if tags[i].tag not in ["JJ", "JJR", "JJS"]
                            ]
                        meronym = " ".join([t.text for t in meronym])
                        meronym = self.singularize(meronym)
                        meronym_list.append(meronym)
                noun = [
                    t
                    for i, t in enumerate(o)
                    if (
                        t.dep_ == "nsubj"
                        or t.dep_ == "nsubjpass"
                        or t.dep_ == "compound"
                    )
                ]
                if (
                    len(noun) == 0
                    or len([t for t in noun if t.dep_ != "compound"]) == 0
                ):
                    noun = [
                        t
                        for i, t in enumerate(o)
                        if (t.dep_ == "ROOT" or t.dep_ == "compound")
                    ]

                root = [
                    t
                    for t in noun
                    if t.dep_ == "ROOT" or t.dep_ == "nsubj" or t.dep_ == "nsubjpass"
                ]
                # filter compound nouns that are not part of the root
                noun = [t for t in noun if t.i <= root[0].i]

                if (
                    len(
                        [
                            t
                            for i, t in enumerate(noun)
                            if tags[i].tag in ["NN", "NNS", "NNP", "NNPS"]
                        ]
                    )
                    == 0
                    and len([t for t in noun if t.tag_ in ["NN", "NNS", "NNP", "NNPS"]])
                    == 0
                    and len(
                        [
                            t
                            for i, t in enumerate(o)
                            if (t.dep_ == "attr" or t.dep_ == "pobj")
                        ]
                    )
                    == 1
                ) or (
                    len(
                        [
                            t
                            for i, t in enumerate(o)
                            if (t.dep_ == "attr" or t.dep_ == "pobj")
                        ]
                    )
                    == 1
                    and noun[0].text == "answer"
                ):
                    noun = [
                        t
                        for i, t in enumerate(o)
                        if t.dep_ == "attr" or t.dep_ == "pobj"
                    ]

                # remove adjectives
                if len(noun) > 1:
                    noun = [
                        t
                        for i, t in enumerate(noun)
                        if tags[i].tag not in ["JJ", "JJR", "JJS"]
                    ]
                    if len(noun) > 1:
                        meronym_list.extend([self.singularize(t.text) for t in noun])

                noun = " ".join([t.text for t in noun])
                noun = self.singularize(noun)

                nouns.append(noun)
                hyponyms.append(hyponym)
                meronyms.append(meronym_list)

        tmp = {
            r: torch.zeros([len(output), len(self.semantic_relationships) + 1])
            for r in self.valid_semantic_relationships
        }
        gt_relationships = [
            _get_match(
                self,
                n.lower().strip(),
                self.dataset_categories[targets[j]].lower().strip(),
                self.semantic_relationships,
            )
            for j, n in enumerate(nouns)
        ]
        # consider compound nouns
        gt_relationships = [
            (
                _get_match(
                    self,
                    self.singularize(nouns[j].split(" ")[-1]).lower().strip(),
                    self.dataset_categories[targets[j]].lower().strip(),
                    self.semantic_relationships,
                )
                if len(nouns[j].split(" ")) > 1 and r == "NIV"
                else r
            )
            for j, r in enumerate(gt_relationships)
        ]
        gt_relationships = [
            self.possible_semantic_relationships.index(r) if r != "NIV" else -1
            for r in gt_relationships
        ]

        h_gt_relationships = [
            _get_match(
                self,
                h.lower().strip(),
                self.dataset_categories[targets[j]].lower().strip(),
                self.semantic_relationships,
            )
            for j, h in enumerate(hyponyms)
        ]
        h_gt_relationships = [
            self.possible_semantic_relationships.index(r) if r != "NIV" else -1
            for r in h_gt_relationships
        ]

        m_gt_relationships = [
            [
                _get_match(
                    self,
                    m.lower().strip(),
                    self.dataset_categories[targets[j]].lower().strip(),
                    self.semantic_relationships,
                )
                for m in m_list
            ]
            for j, m_list in enumerate(meronyms)
        ]
        m_gt_relationships = [
            [
                self.possible_semantic_relationships.index(r) if r != "NIV" else -1
                for r in r_list
            ]
            for r_list in m_gt_relationships
        ]

        # select minimum value greater than -1
        tmp_r, tmp_m = [], []
        for r_list, m_list in zip(m_gt_relationships, meronyms):
            if len(r_list) > 0:
                m_list = [m for r, m in zip(r_list, m_list) if r >= 0]
                r_list = [r for r in r_list if r >= 0]
            tmp_r.append(min(r_list) if len(r_list) > 0 else -1)
            tmp_m.append(m_list[r_list.index(min(r_list))] if len(r_list) > 0 else "")
        m_gt_relationships = tmp_r
        meronyms = tmp_m
        for i, (n, h, m, gt_r, h_gt_r, m_gt_r) in enumerate(
            zip(
                nouns,
                hyponyms,
                meronyms,
                gt_relationships,
                h_gt_relationships,
                m_gt_relationships,
            )
        ):
            # consider compund nouns a single noun or multiple nouns
            match, r = _match_category(self, n, self.semantic_relationships)

            h_match, h_r = _match_category(self, h, self.semantic_relationships)
            m_match, m_r = _match_category(self, m, self.semantic_relationships)
            h_r = self.possible_semantic_relationships.index(h_r) if h_r != "NIV" else 0
            m_r = self.possible_semantic_relationships.index(m_r) if m_r != "NIV" else 0
            r = self.possible_semantic_relationships.index(r) if r != "NIV" else 0
            for j in range(len(self.possible_semantic_relationships)):
                if (
                    self.possible_semantic_relationships[j]
                    in self.valid_semantic_relationships
                ):
                    if j >= 3 and h != "":
                        gt_r = (
                            h_gt_r
                            if gt_r < 0 or (h_gt_r < gt_r and h_gt_r >= 0)
                            else gt_r
                        )
                        if r < 0 or (h_r < r and h_r >= 0):
                            r = h_r
                            match = h_match
                    if j >= 4 and m != "":

                        gt_r = (
                            m_gt_r
                            if gt_r < 0 or (m_gt_r < gt_r and m_gt_r >= 0)
                            else gt_r
                        )
                        if r < 0 or (m_r < r and m_r >= 0):
                            r = m_r
                            match = m_match
                    if j >= gt_r and gt_r >= 0:
                        tmp[self.possible_semantic_relationships[j]][i, targets[i]] = 1
                    else:
                        if j >= r:
                            tmp[self.possible_semantic_relationships[j]][i, match] = 1
                        else:
                            tmp[self.possible_semantic_relationships[j]][i, -1] = 1
        tmp = {k: torch.log(v + 1e-8) for k, v in tmp.items()}

        return tmp, nouns

def main(args):
    cfg = get_cfg()

    cfg.OUTPUT_DIR = "./ALA/{}/{}/{}/".format(
        args.output_dir, args.model, args.dataset
    )

    cfg.DATASETS.PROPOSAL_FILES_TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4

    if args.dataset == "ade":
        cfg.DATASETS.TRAIN = "openvocab_ade20k_panoptic_train"
        cfg.DATASETS.TEST = ["openvocab_ade20k_panoptic_val"]
    elif args.dataset == "cityscapes":
        cfg.DATASETS.TRAIN = "openvocab_cityscapes_fine_panoptic_train"
        cfg.DATASETS.TEST = ["openvocab_cityscapes_fine_panoptic_val"]
        
    else:
        raise NotImplementedError
    cfg.TEST.HUMAN_LABELS_FILE = "./user_study_results/user_study_results.json"
    cfg.TEST.SEMANTIC_RELATIONSHIPS_FILE = (args.semantic_relationship_file_path)

    cfg.TEST.SEMANTIC_RELATIONSHIPS = [
        "exact",
        "synonyms",
        "hyponyms",
        "meronyms",
    ]

    cfg.TEST.VALID_SEMANTIC_RELATIONSHIPS = [
        "exact",
        "synonyms",
        "hyponyms",
        "meronyms",
    ]

    cfg.TEST.OUTPUT_FILES = [
        (
            args.model,
            args.model_outputs_path
        ),
    ]

    if comm.is_main_process():
        setup_logger()

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank())
    model = EvalOsprey(args.model, args.dataset, args.sentence_descriptions ,cfg)
    res = Trainer.test(cfg, model)
    if comm.is_main_process():
        verify_results(cfg, res)
    return res


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--dataset", help="eval dataset type ade/cityscapes", default="ade"
    )

    parser.add_argument(
        "--semantic_relationship_file_path",
        help="path to semantic relationships file",
        default="./ALA//semantic_relationships/output_semantics_gpt4_ade20k.json",
    )

    parser.add_argument(
        "--model",
        help="path to model outputs/descriptions",
        default="shikra",
    )

    parser.add_argument(
        "--model_outputs_path",
        help="path to model outputs/descriptions",
        default="./outputs/shikra/ade20k/descriptions.json"
    )

    parser.add_argument(
        "--output_dir",
        help="path to save logs",
        default="./logs"
    )

    parser.add_argument(
        "--sentence_descriptions",
        help="If set, the script will evaluate sentence descriptions. If not, the script will evaluate only nouns.",
        action="store_true"
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
