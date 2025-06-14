import json
from dataclasses import dataclass

import torch
import transformers
from osprey.constants import IGNORE_INDEX
from torch.utils.data import ConcatDataset

from .osprey_724k import (
    OspreyConversations,
    OspreyDetailedDescription,
    OspreyLVISPosNeg,
    OspreyPartLevel,
    OspreyShortForm,
)
from .stage2_data import COCODataset, PartImagenet, PascalPart, RefCOCO, RefCOCOP
from .vcr import VCRDataset
from .vg import VGDATA


@dataclass
class DataCollatorForDetDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # breakpoint()
        input_ids, labels, img_metas, masks, gt_labels = tuple(
            [instance.get(key, None) for instance in instances]
            for key in ("input_ids", "labels", "img_metas", "masks", "gt_labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        # gt_labels = None
        if not any([l is None for l in gt_labels]):
            gt_labels = [l for sublist in gt_labels for l in sublist]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            img_metas=img_metas,
            masks=masks,
            gt_labels=gt_labels,
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


def make_multitask_data_module(tokenizer, data_args):
    """Make dataset and collator for supervised fine-tuning."""

    if data_args.dataset_config is not None:
        dataset_config = json.load(open(data_args.dataset_config))

    train_dataset = build_osprey_dataset(
        dataset_config, tokenizer=tokenizer, data_args=data_args
    )

    data_collator = DataCollatorForDetDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def build_osprey_dataset(dataset_config, tokenizer=None, data_args=None, **kwargs):
    if isinstance(dataset_config, list):
        datasets = []
        for cfg in dataset_config:
            temp_dataset = build_osprey_dataset(
                cfg, tokenizer=tokenizer, data_args=data_args, **kwargs
            )
            datasets.append(temp_dataset)

        for dataset in datasets:
            print(type(dataset), f"len = {len(dataset)}")

        return ConcatDataset(datasets)

    dataset_type = dataset_config.pop("type")

    if dataset_type == "coco_data":
        dataset = COCODataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )

    elif dataset_type == "vcr":
        dataset = VCRDataset(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == "VGDATA":
        dataset = VGDATA(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == "RefCOCO":
        dataset = RefCOCO(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == "RefCOCOP":
        dataset = RefCOCOP(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == "PascalPart":
        dataset = PascalPart(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == "PartImagenet":
        dataset = PartImagenet(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == "OspreyDetailedDescription":
        dataset = OspreyDetailedDescription(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == "OspreyConversations":
        dataset = OspreyConversations(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == "OspreyShortForm":
        dataset = OspreyShortForm(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == "OspreyPartLevel":
        dataset = OspreyPartLevel(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type == "OspreyLVISPosNeg":
        dataset = OspreyLVISPosNeg(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )

    else:
        raise NotImplementedError

    return dataset


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def collater(self, samples):
        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
