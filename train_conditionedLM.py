import wandb
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer

import subprocess, tempfile
import numpy as np

model_checkpoint = "MU-NLPC/CzeGPT-2"

raw_datasets = load_dataset("cs_restaurants")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# add special tokens
tokenizer.all_special_tokens
num_added_toks = tokenizer.add_tokens(["[SEP]", "[PAD]"], special_tokens=True)
tokenizer.pad_token = "[PAD]"
tokenizer.sep_token = "[SEP]"

max_input_length = 92  # computed from cs_restaurants training/validation sets


def preprocess_function(examples):
    conditioned = [
        da + tokenizer.sep_token + text
        for da, text in zip(examples["da"], examples["text"])
    ]
    model_inputs = tokenizer(
        conditioned, padding="max_length", max_length=max_input_length, truncation=True
    )
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs


tokenized_training_dataset = raw_datasets["train"].map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=["text", "da", "delex_da", "delex_text"],
)

bleu_metric = load_metric("sacrebleu")


def preprocess_for_metrics(eval_prediction):
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


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds, decoded_labels = preprocess_for_metrics(eval_preds)

    decoded_labels = [[labels] for labels in decoded_labels]

    result = bleu_metric.compute(
        predictions=decoded_preds, references=decoded_labels, lowercase=True
    )
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def compute_final_metrics(eval_preds):
    sys, ref = preprocess_for_metrics(eval_preds)

    with tempfile.NamedTemporaryFile(
        "w+", suffix=".txt", delete=False
    ) as sys_file, tempfile.NamedTemporaryFile(
        "w+", suffix=".txt", delete=False
    ) as ref_file:
        sys_file.writelines(line + "\n" for line in sys)
        ref_file.writelines(line + "\n" for line in ref)

    command = [
        "./e2e-metrics/measure_scores.py",
        "-t",
        "-p",
        ref_file.name,
        sys_file.name,
    ]
    eval_output = subprocess.run(command, capture_output=True, check=True)

    metric_names = ["BLEU", "NIST", "METEOR", "ROUGE_L", "CIDEr"]
    metric_scores = map(float, eval_output.stdout.decode().split()[1:])

    return dict(zip(metric_names, metric_scores))


from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

wandb.init(project="MT-finetune-restaurant_cs", entity="kukas")

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 64
epochs = 10
learning_rate = 4e-5

model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-cs_restaurants",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
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
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.compute_metrics = compute_final_metrics
trainer.evaluate()

for sample in raw_datasets["validation"].select(range(20)):
    print(sample["da"])
    # article_en = "inform(food=Indian,good_for_meal='lunch or dinner',name='Kočár z Vídně')"
    article_en = sample["da"]
    model_inputs = tokenizer(article_en, return_tensors="pt").to("cuda")

    generated_tokens = model.generate(**model_inputs)
    res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    print("\t REF: " + sample["text"])
    print("\t GEN: " + res[0])
