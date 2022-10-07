import evaluate
import numpy as np
import pandas as pd
import os
import logging
import time
from glob import glob

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from contextlib import contextmanager
from timeit import default_timer
import gc

from pynvml import *


try:
    import wandb
except ImportError:
    pass

# Flattens references, predictions and sources into a single list
# while keeping track of the original indices
def flatten_references(references, predictions, sources):
    flat_references = []
    flat_predictions = []
    flat_sources = []
    original_indices = []
    for i, (refs, pred, src) in enumerate(zip(references, predictions, sources)):
        for ref in refs:
            flat_references.append(ref)
            flat_predictions.append(pred)
            flat_sources.append(src)
            original_indices.append(i)

    return flat_references, flat_predictions, flat_sources, original_indices


def test_flatten_references():
    references = [["a", "b"], ["c", "d"], ["e", "f"]]
    predictions = ["p1", "p2", "p3"]
    sources = ["s1", "s2", "s3"]
    (
        flat_references,
        flat_predictions,
        flat_sources,
        original_indices,
    ) = flatten_references(references, predictions, sources)
    assert flat_references == ["a", "b", "c", "d", "e", "f"]
    assert flat_predictions == ["p1", "p1", "p2", "p2", "p3", "p3"]
    assert flat_sources == ["s1", "s1", "s2", "s2", "s3", "s3"]


test_flatten_references()

# Selects the best score for each group
def select_max_per_group(scores, original_indices):
    max_scores = []
    for i, score in zip(original_indices, scores):
        if len(max_scores) <= i:
            max_scores.append(score)
        else:
            max_scores[i] = max(max_scores[i], score)
    return max_scores


def test_select_max_per_group():
    scores = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    original_indices = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert select_max_per_group(scores, original_indices) == [3, 5, 9]


test_select_max_per_group()


class Evaluation:
    def __init__(
        self,
        tokenizer,
        ignore_pad_token_for_loss,
        eval_dataset,
        output_dir,
        compute_light_metrics,
        compute_heavy_metrics,
    ):
        self.trainer = None
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.split_name = "validation"
        self.metric_key_prefix = "eval"
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir

        # Light metrics
        self.compute_light_metrics = compute_light_metrics
        if compute_light_metrics:
            self.bleu = evaluate.load("sacrebleu")
            self.chrf = evaluate.load("chrf")
            self.rouge = evaluate.load("rouge")
            self.meteor = evaluate.load("meteor")
            self.ter = evaluate.load("ter")

        # Heavy metrics
        self.compute_heavy_metrics = compute_heavy_metrics
        if compute_heavy_metrics:
            self.bleurt = evaluate.load("bleurt", config_name="BLEURT-20")
            self.bertscore = evaluate.load("bertscore")
            self.comet = evaluate.load("comet", config_name="wmt20-comet-da")

    def evaluate_predictions(self, predictions):
        # Prepare the references
        references = list(self.eval_dataset["target"])
        max_references = max([len(ref) for ref in references])

        # Sacrebleu needs the same number of references for each sample
        padded_references = []
        for ref in references:
            padded_references.append(ref + [""] * (max_references - len(ref)))

        # Prepare the sources
        sources = list(self.eval_dataset["input"])

        result = {}

        # n-gram based
        if self.compute_light_metrics:
            result["BLEU"] = self.bleu.compute(
                predictions=predictions, references=padded_references
            )["score"]
            result["BLEU_lower"] = self.bleu.compute(
                predictions=predictions, references=padded_references, lowercase=True
            )["score"]
            result["chrF++"] = self.chrf.compute(
                predictions=predictions,
                references=padded_references,
                word_order=2,
            )["score"]
            result["TER"] = self.ter.compute(
                predictions=predictions, references=padded_references
            )["score"]

            result["rougeL"] = self.rouge.compute(
                predictions=predictions, references=references
            )["rougeL"]
            result["meteor"] = self.meteor.compute(
                predictions=predictions, references=references
            )["meteor"]

        # result = {k: round(v, 4) for k, v in result.items()}
        per_example_result = {}

        if self.compute_heavy_metrics:
            # Flatten the references, predictions and sources
            (
                flat_references,
                flat_predictions,
                flat_sources,
                original_indices,
            ) = flatten_references(references, predictions, sources)

            # Compute the heavy metrics
            bertscore_output = self.bertscore.compute(
                predictions=predictions,
                references=references,
                model_type="microsoft/deberta-xlarge-mnli",
            )
            result["BERTscore_precision"] = np.mean(bertscore_output["precision"])
            result["BERTscore_recall"] = np.mean(bertscore_output["recall"])
            result["BERTscore_f1"] = np.mean(bertscore_output["f1"])
            per_example_result["BERTscore_precision"] = bertscore_output["precision"]
            per_example_result["BERTscore_recall"] = bertscore_output["recall"]
            per_example_result["BERTscore_f1"] = bertscore_output["f1"]

            comet_output = self.comet.compute(
                predictions=flat_predictions,
                references=flat_references,
                sources=flat_sources,
            )
            comet_flat_scores = comet_output["scores"]
            comet_scores = select_max_per_group(comet_flat_scores, original_indices)
            result["COMET"] = np.mean(comet_scores)
            per_example_result["COMET"] = comet_scores

            bleurt_output = self.bleurt.compute(
                predictions=flat_predictions, references=flat_references
            )
            bleurt_flat_scores = bleurt_output["scores"]
            bleurt_scores = select_max_per_group(bleurt_flat_scores, original_indices)
            result["BLEURT"] = np.mean(bleurt_scores)
            per_example_result["BLEURT"] = bleurt_scores

        examples = {
            "source": sources,
            "prediction": predictions,
            "reference": references,
        }
        per_example_result = {**examples, **per_example_result}

        return result, per_example_result

    def evaluate_csv_outputs(self, csv_dir, log_to_csv, log_to_wandb):
        csv_paths = glob(os.path.join(csv_dir, "outputs_*.csv"))
        steps = [int(csv_path.rsplit("_")[-1].rsplit(".")[0]) for csv_path in csv_paths]
        # Sort the csv paths by step
        for step, csv_path in sorted(zip(steps, csv_paths)):
            logger.info(f"Evaluating {csv_path}")
            # Read the csv
            df = pd.read_csv(csv_path)
            predictions = df["prediction"].tolist()
            result, per_example_result = self.evaluate_predictions(predictions)

            if log_to_csv:
                self.save_results_csv(per_example_result, "outputs", step=step)
                self.save_results_csv([result], "results", step=step)
            if log_to_wandb:
                per_example_result = pd.DataFrame(per_example_result)
                self.save_results_wandb(result, step=step)
                self.save_results_wandb({"outputs": per_example_result}, step=step)

    def save_results_csv(self, results, name, step=None):
        step = step if step is not None else self.trainer.state.global_step
        output_dir = self.output_dir

        csv_path = os.path.join(output_dir, f"{name}_{self.split_name}_{step}.csv")

        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        logger.info(f"Evaluation results saved in {csv_path}")

        # print the evaluation result
        logger.info(df)

    def save_results_wandb(self, results, step=None):
        assert wandb is not None, "You need to install wandb to use this feature"

        results_with_prefix = {
            f"{self.metric_key_prefix}/{result_name}": result
            for result_name, result in results.items()
        }

        step = step if step is not None else self.trainer.state.global_step
        wandb.log(
            {**results_with_prefix, "train/global_step": step},
            commit=True,
        )

    def compute_metrics(self, eval_preds):
        assert (
            self.trainer is not None
        ), "Trainer was not passed to the Evaluation class."

        # Prepare the model predictions
        preds, _ = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        predictions = [pred.strip() for pred in decoded_preds]

        # Compute the evaluation metrics using the decoded predictions
        result, per_example_result = self.evaluate_predictions(predictions)

        # Compute prediction lengths (in tokens)
        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)

        self.save_results_csv(per_example_result, "outputs")
        self.save_results_csv([result], "results")

        return result
