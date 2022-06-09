import os
import logging

from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    default_data_collator,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    HfArgumentParser,
)

# Multilingual tokenizers
from transformers import (
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    M2M100Tokenizer,
)
from transformers.trainer_utils import get_last_checkpoint

import numpy as np

import datasets
from datasets import load_dataset
import evaluate

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "b"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on "
            "a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on "
            "a jsonlines file."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
            "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
            "needs to be the target language token.(Usually it is the target language token)"
        },
    )


def parse_args():
    # Parse command line arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sanity checks

    # TODO: add t5 warning (from run_translation.py)
    # TODO: add checkpoint reload warning (from run_translation.py)

    # Check if we have a dataset

    if (
        data_args.dataset_name is None
        and data_args.train_file is None
        and data_args.validation_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation file.")

    check_dataset_extensions = ["train_file", "validation_file", "test_file"]
    for check_file in check_dataset_extensions:
        if getattr(data_args, check_file) is not None:
            extension = getattr(data_args, check_file).split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], f"`{check_file}` should be a csv or a json file."

    return model_args, data_args, training_args


def load_custom_dataset(train_file, validation_file, test_file, cache_dir=None):
    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
        extension = train_file.split(".")[-1]
    if validation_file is not None:
        data_files["validation"] = validation_file
        extension = validation_file.split(".")[-1]
    if test_file is not None:
        data_files["test"] = test_file
        extension = test_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )

    return raw_datasets


class DatasetPreprocessing:
    """Class that handles dataset preprocessing pipeline"""

    def __init__(self, preprocessing_num_workers):
        self.preprocessing_num_workers = preprocessing_num_workers

    def rename_columns_cs_restaurants(self, datasets):
        return datasets.rename_column("text", "target").rename_column("da", "input")

    def select_dataset_subset(
        self, datasets, max_train_samples, max_eval_samples, max_predict_samples
    ):
        limits = {
            "train": max_train_samples,
            "validation": max_eval_samples,
            "test": max_predict_samples,
        }
        for split_name, args_limit in limits.items():
            if args_limit is not None:
                limit = min(len(datasets[split_name]), args_limit)
                datasets[split_name] = datasets[split_name].select(range(limit))

        return datasets

    def tokenize(
        self,
        datasets,
        tokenizer,
        padding,
        max_source_length,
        max_target_length,
        ignore_pad_token_for_loss,
    ):
        # assert tokenizer is not None

        def _preprocess_function(examples):
            inputs = examples["input"]
            targets = examples["target"]
            # inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(
                inputs,
                max_length=max_source_length,
                padding=padding,
                truncation=True,
            )

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=max_target_length,
                    padding=padding,
                    truncation=True,
                )

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label]
                    for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        any_split = next(iter(datasets.keys()))
        column_names = datasets[any_split].column_names

        return datasets.map(
            _preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=column_names,
        )

    def call_method(self, func_name, datasets, **kwargs):
        assert hasattr(
            self, func_name
        ), f"Preprocessing method {func_name} not defined!"

        func = getattr(self, func_name)
        datasets = func(datasets, **kwargs)

        return datasets


class Evaluation:
    def __init__(self, tokenizer, ignore_pad_token_for_loss):
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss

        # Metric
        self.metric = evaluate.load("sacrebleu")

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(
            decoded_preds, decoded_labels
        )

        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


def handle_mbart(model, tokenizer, forced_bos_token):
    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).

    MULTI = (
        MBartTokenizer,
        MBartTokenizerFast,
        MBart50Tokenizer,
        MBart50TokenizerFast,
        M2M100Tokenizer,
    )
    if isinstance(tokenizer, MULTI):
        # assert (
        #     data_args.target_lang is not None and data_args.source_lang is not None
        # ), (
        #     f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
        #     "--target_lang arguments."
        # )

        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "cs_CZ"

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = None
        if forced_bos_token is not None:
            forced_bos_token_id = tokenizer.lang_code_to_id[forced_bos_token]

        model.config.forced_bos_token_id = forced_bos_token_id


def detect_last_checkpoint(output_dir, overwrite_output_dir, resume_from_checkpoint):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(output_dir) and not overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def main():
    model_args, data_args, training_args = parse_args()

    # logger.info(f"Training/evaluation parameters {training_args}")

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        transformers.set_seed(training_args.seed)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    last_checkpoint = None
    if training_args.do_train:
        last_checkpoint = detect_last_checkpoint(
            training_args.output_dir,
            training_args.overwrite_output_dir,
            training_args.resume_from_checkpoint,
        )

    # Load raw data
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        # Load data from local files
        raw_datasets = load_custom_dataset(
            data_args.train_file,
            data_args.validation_file,
            data_args.test_file,
            cache_dir=model_args.cache_dir,
        )

    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    # Prepare data
    # - Preprocess data
    # - remove markup in das
    # - Delexicalize
    # - Tokenize
    dp = DatasetPreprocessing(data_args.preprocessing_num_workers)

    preprocessed_datasets = raw_datasets

    preprocessed_datasets = dp.rename_columns_cs_restaurants(preprocessed_datasets)

    preprocessed_datasets = dp.select_dataset_subset(
        preprocessed_datasets,
        data_args.max_train_samples,
        data_args.max_eval_samples,
        data_args.max_predict_samples,
    )

    padding = "max_length" if data_args.pad_to_max_length else False
    preprocessed_datasets = dp.tokenize(
        preprocessed_datasets,
        tokenizer,
        padding,
        data_args.max_source_length,
        data_args.max_target_length,
        data_args.ignore_pad_token_for_loss,
    )

    # Prepare model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )

    # Prepare data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Custom code for mBART
    handle_mbart(model, tokenizer, data_args.forced_bos_token)

    # Evaluation class
    evaluation = Evaluation(tokenizer, data_args.ignore_pad_token_for_loss)

    train_data = preprocessed_datasets["train"] if training_args.do_train else None
    eval_data = preprocessed_datasets["validation"] if training_args.do_eval else None
    compute_metrics = (
        evaluation.compute_metrics if training_args.predict_with_generate else None
    )
    # Prepare trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    # - custom evaluation function
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        # max_train_samples = (
        #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        # )
        # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    # - custom decoding (constrained decoding)
    # - lexicalization
    max_length = training_args.generation_max_length
    num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_data)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_data))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
