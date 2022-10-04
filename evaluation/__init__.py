import evaluate
import torch
import numpy as np
import pandas as pd
import os
import logging
import time

logger = logging.getLogger(__name__)

from contextlib import contextmanager
from timeit import default_timer
import gc

from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


class Evaluation:
    def __init__(self, tokenizer, ignore_pad_token_for_loss, eval_dataset):
        self.trainer = None
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.split_name = "validation"
        self.eval_dataset = eval_dataset

        # Metric
        self.bleu = evaluate.load("sacrebleu")
        self.chrf = evaluate.load("chrf")
        # self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        self.ter = evaluate.load("ter")

    def postprocess_text(self, preds, _labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in _labels]
        flat_labels = [label.strip() for label in _labels]

        return preds, labels, flat_labels

    def _compute_metrics(self, eval_preds):
        preds, labels = eval_preds

        # bertscore = evaluate.load("bertscore")
        # comet = evaluate.load("comet", config_name="eamt22-cometinho-da")
        # bleurt = evaluate.load("bleurt", config_name="BLEURT-20")

        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        sys, refs, flat_refs = self.postprocess_text(decoded_preds, decoded_labels)

        result = {}

        sources = list(self.eval_dataset["input"])
        # print(da)
        # assert len(sources) == len(sys)
        # e2e_result = run_e2e_metrics(sys, refs)
        # for metric_name, score in e2e_result:
        #     result[f"E2E_{metric_name}"] = score
        # ser = run_slot_error_rate_metric(sys, da)
        # result["SER"] = ser

        # n-gram based
        with elapsed_timer() as elapsed:
            result["BLEU"] = self.bleu.compute(predictions=sys, references=refs)[
                "score"
            ]
            result["BLEU_lowercase"] = self.bleu.compute(
                predictions=sys, references=refs, lowercase=True
            )["score"]
            result["chrF++"] = self.chrf.compute(
                predictions=sys,
                references=refs,
                word_order=2,
            )["score"]
            result["TER"] = self.ter.compute(predictions=sys, references=refs)["score"]

            # result["rougeL"] = self.rouge.compute(
            #     predictions=sys, references=flat_refs
            # )["rougeL"]
            meteor = self.meteor.compute(predictions=sys, references=flat_refs)[
                "meteor"
            ]
            result["meteor"] = meteor
        print(f"word overlap based elapsed {elapsed()}")

        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)

        result = {k: round(v, 4) for k, v in result.items()}

        assert self.trainer is not None
        step = self.trainer.state.global_step
        output_dir = self.trainer.args.output_dir
        df = pd.DataFrame(
            {
                "sources": sources,
                "prediction": sys,
                "reference": flat_refs,
                # "bertscore_precision": bertscore_output["precision"],
                # "bertscore_recall": bertscore_output["recall"],
                # "bertscore_f1": bertscore_output["f1"],
                # "comet": comet_output["scores"],
                # "bleurt": bleurt_output["scores"],
            }
        )
        csv_path = os.path.join(output_dir, f"outputs_{self.split_name}-{step}.csv")
        df.to_csv(csv_path)
        logger.info(f"Evaluation set outputs saved in {csv_path}")

        # print the evaluation predictions
        print(df)

        df = pd.DataFrame([result])
        csv_path = os.path.join(output_dir, f"results_{self.split_name}-{step}.csv")
        df.to_csv(csv_path)
        logger.info(f"Evaluation set results saved in {csv_path}")

        return result

    def compute_metrics(self, eval_preds):
        print("before collect")
        print_gpu_utilization()
        gc.collect()
        torch.cuda.empty_cache()
        print("after collect")
        print_gpu_utilization()
        result = self._compute_metrics(eval_preds)
        print("after compute_metrics")
        print_gpu_utilization()
        gc.collect()
        torch.cuda.empty_cache()
        print("after compute_metrics and collect")
        print_gpu_utilization()
        return result
