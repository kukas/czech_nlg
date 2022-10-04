import os
import logging

from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
import csv
import numpy as np

from datasets import load_dataset
import evaluate
from dataset_utils import DatasetPreprocessing
import pandas as pd


@dataclass
class EvaluationArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    split: str = field(
        metadata={
            "help": "Choose the dataset split to evaluate on",
            "choices": ["train", "validation", "test"],
        },
    )
    outputs_csv: Optional[str] = field(
        metadata={
            "help": "Path to the CSV file containing the system outputs",
            "nargs": "+",
            "default": None,
        }
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory containing multiple system outputs",
        },
    )
    small_models: Optional[bool] = field(
        default=False,
        metadata={"help": "Use small models for evaluation"},
    )


def compute_bertscore(predictions, references):
    bertscore = evaluate.load("bertscore")
    return bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="microsoft/deberta-xlarge-mnli",
    )


def compute_comet(predictions, references, sources):
    comet = evaluate.load("comet")
    # When a record comes with several references, we run BLEURT on each reference and report the highest value (Zhang et al., 2020).
    return comet.compute(
        predictions=predictions, references=references, sources=sources
    )


def compute_bleurt(predictions, references):
    bleurt = evaluate.load("bleurt", config_name="BLEURT-20-D3")
    return bleurt.compute(predictions=predictions, references=references)


def load_predictions_from_csv(csv_path):
    # Read the CSV file, save sources and predictions in a list
    sources = []
    predictions = []
    with open(csv_path, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            sources.append(row["sources"])
            predictions.append(row["prediction"])

    # Deduplicate the sources and system outputs while preserving the order
    # TODO: only needed for older training runs, remove later
    unique_sources = []
    unique_predictions = []
    for source, system_output in zip(sources, predictions):
        # Check only last 3 unique sources
        if source not in unique_sources[-3:]:
            unique_sources.append(source)
            unique_predictions.append(system_output)

    return unique_sources, unique_predictions


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


def main():
    # Parse command line arguments
    parser = HfArgumentParser(EvaluationArguments)
    (args,) = parser.parse_args_into_dataclasses()

    # Load the CSVs
    if args.output_dir:
        metric_key_prefix = "eval"
        csv_paths = glob.glob(
            os.path.join(args.output_dir, "outputs_{}*.csv".format(metric_key_prefix))
        )
    else:
        csv_paths = args.outputs_csv

    # Load the WebNLG dataset using Huggingface datasets
    raw_dataset = load_dataset(
        "datasets/web_nlg/web_nlg.py",
        "release_v3.0_ru_cs",
        data_dir="datasets/web_nlg/webnlg-dataset-cz",
        cache_dir=None,
        split=args.split,
    )

    dp = DatasetPreprocessing(16)

    preprocessed_dataset = dp.filter_web_nlg_lexicalizations(raw_dataset)
    preprocessed_dataset = dp.generate_inputs_web_nlg(preprocessed_dataset)
    # preprocessed_dataset = dp.flatten_lexicalizations_web_nlg(
    #     preprocessed_dataset, only_first_lexicalization=True
    # )

    inputs = preprocessed_dataset["input"]
    references = preprocessed_dataset["lex"]

    for csv_path in csv_paths:
        print(f"Computing metrics for {csv_path}")
        # Extract step number from the csv path
        step = int(csv_path.split("-")[-1].split(".")[0])
        print(step)

        sources, predictions = load_predictions_from_csv(csv_path)

        # Check if all unique sources match the inputs in the dataset
        assert sources == list(inputs), "Mismatch between sources and dataset inputs"

        # Flatten the references, predictions and sources
        (
            flat_references,
            flat_predictions,
            flat_sources,
            original_indices,
        ) = flatten_references(references, predictions, sources)

        print(flat_sources[:3], "\n", flat_predictions[:3], "\n", flat_references[:3])

        result = {}
        per_example_result = pd.DataFrame(
            {
                "source": sources,
                "prediction": predictions,
                "references": references,
            }
        )

        # Compute the heavy metrics
        bertscore_output = compute_bertscore(predictions, references)
        result["BERTscore_precision"] = np.mean(bertscore_output["precision"])
        result["BERTscore_recall"] = np.mean(bertscore_output["recall"])
        result["BERTscore_f1"] = np.mean(bertscore_output["f1"])
        per_example_result["BERTscore_precision"] = bertscore_output["precision"]
        per_example_result["BERTscore_recall"] = bertscore_output["recall"]
        per_example_result["BERTscore_f1"] = bertscore_output["f1"]

        # comet_output = compute_comet(flat_predictions, flat_references, flat_sources)
        # comet_flat_scores = comet_output["scores"]
        # comet_scores = select_max_per_group(comet_flat_scores, original_indices)
        # result["COMET"] = np.mean(comet_scores)
        # per_example_result["COMET"] = comet_scores

        # bleurt_output = compute_bleurt(flat_predictions, flat_references)
        # bleurt_flat_scores = bleurt_output["scores"]
        # bleurt_scores = select_max_per_group(bleurt_flat_scores, original_indices)
        # result["BLEURT"] = np.mean(bleurt_scores)
        # per_example_result["BLEURT"] = bleurt_scores

        print(bleurt_flat_scores, bleurt_scores)

        assert len(bleurt_scores) == len(references)

        print(result)
        # Save the overall results
        pd.DataFrame(result).to_csv(
            csv_path.replace(".csv", "_heavy_overall.csv"), index=False
        )

        # Save the per example scores to a new CSV file
        per_example_result.to_csv(
            csv_path.replace(".csv", "_heavy_scores.csv"), index=False
        )


if __name__ == "__main__":
    main()
