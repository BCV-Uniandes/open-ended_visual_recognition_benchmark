from collections import OrderedDict

from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.comm import is_main_process


class DatasetEvaluatorDict(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(
        self,
        evaluators,
        cfg,
    ):
        """
        Args:
            evaluators (dict): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators
        self._model_name = "debug"#cfg.TEST.OUTPUT_FILES[0][0] 

    def reset(self):
        for k in self._evaluators:
            self._evaluators[k].reset()

    def process(self, inputs, outputs):

        for k in self._evaluators:
            self._evaluators[k].process(inputs, outputs[self._model_name][k])

    def evaluate(self):
        results = {k: OrderedDict() for k in self._evaluators}

        for k in self._evaluators:
            result = self._evaluators[k].evaluate()
            if is_main_process() and result is not None:
                for _k, v in result.items():
                    assert (
                        _k not in results
                    ), "Different evaluators produce results with the same key {}".format(
                        _k
                    )
                    results[k][_k] = v
        return results
