from argparse import ArgumentParser, BooleanOptionalAction
import wandb
import datetime
import os
import numpy as np
from functools import partial
from evaluation import run_fast_metrics, run_e2e_metrics
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from torchinfo import summary

def preprocess_function(examples, tokenizer, task_prefix, max_input_length, max_target_length):
    inputs = [task_prefix + da for da in examples["da"]]
    targets = examples["text"]
    model_inputs = tokenizer(inputs, padding=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    # model_inputs["decoder_input_ids"] = labels["input_ids"]
    return model_inputs

def preprocess_for_metrics(eval_prediction, tokenizer):
    preds, labels = eval_prediction
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    return decoded_preds, decoded_labels


def compute_metrics(eval_preds, tokenizer):
    sys, ref = preprocess_for_metrics(eval_preds, tokenizer)

    text_table = wandb.Table(columns=["System output", "Reference"])
    for sys_line, ref_line in zip(sys[:50], ref):
        text_table.add_data(sys_line, ref_line)
    wandb.log({"validation_samples_subset" : text_table})

    return run_fast_metrics(sys, ref)


def compute_final_metrics(eval_preds, model, tokenizer):
    sys, ref = preprocess_for_metrics(eval_preds, tokenizer)

    bleu_metric = load_metric("sacrebleu")

    text_table = wandb.Table(columns=["System output", "Reference", "BLEU", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "Loss"])
    for sys_line, ref_line in zip(sys, ref):
        # pos_score, neg_score, neutral_score = model(headline)
        bleu = bleu_metric.compute(
            predictions=[sys_line], references=[[ref_line]], lowercase=True
        )
        #model_output = model(**inputs)
        text_table.add_data(sys_line, ref_line, bleu["score"], bleu["precisions"][0], bleu["precisions"][1], bleu["precisions"][2], bleu["precisions"][3], 0)

    wandb.log({"validation_samples" : text_table})

    return run_e2e_metrics(sys, ref)



def run_experiment(do_train, do_eval, do_predict, model_checkpoint, batch_size, validation_batch_size, epochs, learning_rate, weight_decay, train_size, validation_size, scratch_dir, logging_strategy, logging_steps):
    method_name = "text2text"

    # raw_datasets = load_dataset("wmt16", "cs-en")
    dataset_name = "cs_restaurants"
    raw_datasets = load_dataset(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    if "google/t5" in model_checkpoint:
        task_prefix = "translate English to Czech: "
    else:
        task_prefix = ""

    naively_tokenized_datasets = raw_datasets.map(
        lambda batch: {
            "text_input_ids": tokenizer(batch["text"])["input_ids"],
            "da_input_ids": tokenizer([task_prefix + da for da in batch["da"]])["input_ids"],
        },
        batched=True,
        num_proc=4,
        remove_columns=["text", "da", "delex_da", "delex_text"],
    )
    print()
    # max_input_length = max(len(sentence) for sentence in naively_tokenized_datasets["train"]["text_input_ids"])
    # max_target_length = max(len(sentence) for sentence in naively_tokenized_datasets["train"]["da_input_ids"])
    # print("maximum text token length", max_input_length)
    # print("maximum da token length", max_target_length)
    max_input_length = None
    max_target_length = None
    # for split in ["validation", "test", "train"]:
    #     max_input_length = max(max_input_length, max(len(sentence) for sentence in naively_tokenized_datasets[split]["text_input_ids"]))
    #     max_target_length = max(max_target_length, max(len(sentence) for sentence in naively_tokenized_datasets[split]["da_input_ids"]))
    #     print(split, "maximum text token length", max_input_length)
    #     print(split, "maximum da token length", max_target_length)

    # print("selected:", max_input_length, max_target_length)

    tokenized_datasets = raw_datasets.map(
        partial(preprocess_function, tokenizer=tokenizer, task_prefix=task_prefix, max_input_length=max_input_length, max_target_length=max_target_length),
        batched=True,
        num_proc=4,
        remove_columns=["text", "da", "delex_da", "delex_text"],
    )


    train_size = int(len(tokenized_datasets["train"])*train_size)
    tokenized_datasets["train"] = tokenized_datasets["train"].select(range(train_size))
    
    validation_size = int(len(tokenized_datasets["validation"])*validation_size)
    tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(validation_size))

    print(raw_datasets["train"][:3])
    print(tokenized_datasets["train"][:3])
    # Create logdir name
    wandb.init(project="MT-finetune-restaurant_cs", entity="kukas", dir=os.path.join(scratch_dir, "wandb"))
    # run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = wandb.run.name
    
    output_dir = os.path.join(scratch_dir, dataset_name, "text2text", run_name)
    os.makedirs(output_dir, exist_ok=False)


    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    print("Total number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    summary(model)

    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        output_dir,
        evaluation_strategy=logging_strategy if epochs > 1 else "no",
        logging_strategy=logging_strategy,
        save_strategy=logging_strategy,
        logging_steps=logging_steps,
        save_steps=logging_steps,
        eval_steps=logging_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=validation_batch_size,
        weight_decay=weight_decay,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        eval_accumulation_steps=1,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=True,
        report_to="wandb",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
    )

    if do_train:
        trainer.train()
    
    trainer.compute_metrics = partial(compute_final_metrics, model=model, tokenizer=tokenizer)

    if do_eval:
        trainer.evaluate(tokenized_datasets["validation"], num_beams=8, metric_key_prefix="eval")

    if do_predict:
        trainer.evaluate(tokenized_datasets["test"], num_beams=8, metric_key_prefix="test")
    # for sample in raw_datasets["validation"].select(range(20)):
    #     print(sample["da"])
    #     # article_en = "inform(food=Indian,good_for_meal='lunch or dinner',name='Kočár z Vídně')"
    #     article_en = sample["da"]
    #     model_inputs = tokenizer(article_en, return_tensors="pt").to("cuda")

    #     generated_tokens = model.generate(**model_inputs)
    #     res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    #     print("\t REF: " + sample["text"])
    #     print("\t GEN: " + res[0])


if __name__ == '__main__':

    ap = ArgumentParser()

    ap.add_argument('--scratch-dir', type=str,
                    help='Path to the working directory for experiment outputs', default="./experiments/")
    ap.add_argument('--batch-size', type=int,
                    help='Batch size per one device', default=32)
    ap.add_argument('--validation-batch-size', type=int,
                    help='Validation batch size per one device', default=8)
    ap.add_argument('--epochs', type=int,
                    help='Number of training epochs', default=10)
    ap.add_argument('--train-size', type=float,
                    help='Portion of the training data to use (default: 1.0)', default=1.0)
    ap.add_argument('--validation-size', type=float,
                    help='Portion of the validation data to use (default: 1.0)', default=1.0)

    ap.add_argument('--logging-steps', type=float,
                    help='Evaluate/Log/Save each N steps', default=100)
    ap.add_argument('--logging-strategy', type=str,
                    help='Evaluate/Log/Save each N steps or each epoch', default="epoch", choices=["steps", "epoch", "no"])

    ap.add_argument('--weight-decay', type=float,
                    help='Weight decay', default=0.01)
    ap.add_argument('--learning-rate', type=float,
                    help='Learning rate', default=4e-5)
    ap.add_argument('--checkpoint', type=str,
                    help='Which model checkpoint to use', default="Helsinki-NLP/opus-mt-en-cs")

    ap.add_argument('--do-train', default=False, action=BooleanOptionalAction)
    ap.add_argument('--do-eval', default=False, action=BooleanOptionalAction)
    ap.add_argument('--do-predict', default=False, action=BooleanOptionalAction)

    args = ap.parse_args()

    run_experiment(args.do_train, args.do_eval, args.do_predict, args.checkpoint, args.batch_size, args.validation_batch_size, args.epochs, args.learning_rate, args.weight_decay, args.train_size, args.validation_size, args.scratch_dir, args.logging_strategy, args.logging_steps)
