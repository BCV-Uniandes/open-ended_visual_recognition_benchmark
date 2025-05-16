import json
import os
from collections import OrderedDict

import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.logger import logging


class AccuracyEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_folder=None, distributed=True):
        self._dataset_name = dataset_name
        self._distributed = distributed
        meta = MetadataCatalog.get(dataset_name)
        if meta.get("stuff_classes", None) is not None:
            self._class_names = meta.stuff_classes + ["background"]
            self._num_classes = len(meta.stuff_classes)
        else:
            self._class_names = meta.thing_classes + ["background"]
            self._num_classes = len(meta.thing_classes)
        self._logger = logging.getLogger("detectron2.trainer")
        self.output_folder = output_folder

    def reset(self):
        self._conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            if "descriptions" in output:
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "descriptions": output["descriptions"],
                    }
                )
            segments_info = input["segments_info"]
            # sort segments_info by id
            segments_info = sorted(segments_info, key=lambda x: x["id"])
            input["gt_labels"] = [s["category_id"] for s in segments_info]
            for gt, pred in zip(input["gt_labels"], output["pred_labels"]):
                self._conf_matrix[gt, pred] += 1
        if len(self._predictions) > 0:
            with open(os.path.join(self.output_folder, "descriptions.json"), "w") as f:
                json.dump(self._predictions, f)

    def evaluate(self):
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            predictions_list = all_gather(self._predictions)
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix
            self._predictions = []
            for predictions in predictions_list:
                self._predictions.extend(predictions)
            if len(self._predictions) > 0:
                print("Writing all text descriptions to file")
                with open(
                    os.path.join(self.output_folder, "descriptions.json"), "w"
                ) as f:
                    json.dump(self._predictions, f)

        acc = np.full(self._num_classes + 1, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal().astype(float)
        pos_gt = np.sum(self._conf_matrix, axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        res["total"] = np.sum(pos_gt)
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]

        results = OrderedDict({"accuracy": res})
        self._logger.info(results)
        return results
