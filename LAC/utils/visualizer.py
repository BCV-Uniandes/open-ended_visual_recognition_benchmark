import numpy as np
import torch
import wandb
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from panopticapi.utils import rgb2id
from PIL import Image


class Visualizer(DatasetEvaluator):

    def __init__(self, dataset_name, models, cfg, distributed=True):
        self._dataset_name = dataset_name
        self._models = models
        self._valid_semantic_relationships = cfg.TEST.VALID_SEMANTIC_RELATIONSHIPS
        self._distributed = distributed
        meta = MetadataCatalog.get(dataset_name)

        if meta.get("stuff_classes", None) is not None:
            self._class_names = meta.stuff_classes + ["background"]
            self._num_classes = len(meta.stuff_classes)
        else:
            self._class_names = meta.thing_classes + ["background"]
            self._num_classes = len(meta.thing_classes)

        stuff_classes = meta.stuff_classes  # Stuff includes things
        things_classes = meta.thing_classes

        self.stuff_ids = [
            stuff_classes.index(i) for i in stuff_classes if i not in things_classes
        ] + [
            self._num_classes
        ]  # stuff + bg
        self.things_ids = [
            stuff_classes.index(i) for i in stuff_classes if i in things_classes
        ] + [
            self._num_classes
        ]  # things + bg

        self.columns = [
            "Image",
            "Mask",
            "GT",
        ]
        for model in self._models:
            self.columns.append(model + "'s Prediction")
            self.columns.append("Noun for {:s}".format(model))
            self.columns.append(model + "'s Mapping")
        self.table = wandb.Table(columns=self.columns)
        self.data = []

    def reset(self):
        return

    def get_gt_label(self, x):
        gt_label = np.array(Image.open(x[0]["pan_seg_file_name"]))
        gt_label = rgb2id(gt_label)
        gt_ids = np.unique(gt_label)
        gt_label = torch.from_numpy(gt_label)

        output_mask = []
        cnt = 0
        for i, id in enumerate(gt_ids):
            if id == 0:
                continue
            output_mask.append((gt_label == id).float())
            cnt += 1
        output_mask = torch.stack(output_mask, dim=0)
        output_mask = output_mask

        return output_mask

    def process(self, inputs, outputs):
        for i, input in enumerate(inputs):
            segments_info = input["segments_info"]
            # Sort segments by id
            segments_info = sorted(segments_info, key=lambda x: x["id"])
            input["gt_labels"] = [s["category_id"] for s in segments_info]
            image = None
            for m in range(len(input["gt_labels"])):
                if (
                    outputs["Osprey"]["sentence_bert"][i]["pred_labels"][m]
                    == input["gt_labels"][m]
                    and outputs["Osprey"]["exact"][i]["pred_labels"][m]
                    == input["gt_labels"][m]
                ) and (
                    outputs["Ours"]["sentence_bert"][i]["pred_labels"][m]
                    != input["gt_labels"][m]
                    and outputs["Ours"]["exact"][i]["pred_labels"][m]
                    == input["gt_labels"][m]
                ):
                    if image is None:
                        image = input["image"]
                        mask_pred_result = self.get_gt_label([input])
                    data = [
                        wandb.Image(image[[2, 1, 0], :, :]),
                        wandb.Image(
                            ((mask_pred_result[m]) * 255).float(),
                        ),
                        self._class_names[input["gt_labels"][m]],
                    ]
                    for method in outputs.keys():
                        mapping = {
                            k: (
                                self._class_names[
                                    outputs[method][k][i]["pred_labels"][m]
                                ]
                                if outputs[method][k][i]["pred_labels"][m]
                                < len(self._class_names)
                                else "background"
                            )
                            for k in self._valid_semantic_relationships
                        }
                        mapping = "\n".join(
                            ["{:s}: {:s}".format(k, v) for k, v in mapping.items()]
                        )
                        data.append(outputs[method]["descriptions"][m])
                        data.append(outputs[method]["nouns"][m])
                        data.append(mapping)
                    self.data.append(data)

    def evaluate(self):
        if self._distributed:
            synchronize()
            data_list = all_gather(self.data)
            if not is_main_process():
                return

            self.data = []
            for data in data_list:
                for d in data:
                    self.table.add_data(*d)

        test_samples = wandb.Artifact("test_samples_cityscapes", type="predictions")
        test_samples.add(self.table, "predictions")
        wandb.run.log_artifact(test_samples)
