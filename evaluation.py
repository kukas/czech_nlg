
import subprocess, tempfile
import numpy as np
from datasets import load_metric

bleu_metric = load_metric("sacrebleu")

def run_fast_metrics(sys, ref):
    """Computes metrics using Huggingface built-in functions

    Args:
        sys (list[str]): NLG system outputs
        ref (list[str]): Reference outputs

    Returns:
        dict: dictionary with the metrics
    """

    decoded_refs = [[labels] for labels in ref]

    result = bleu_metric.compute(
        predictions=sys, references=decoded_refs, lowercase=True
    )
    metrics = {"bleu": result["score"]}

    #np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    prediction_lens = [len(pred) for pred in sys]
    metrics["gen_len"] = np.mean(prediction_lens)
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics

def run_e2e_metrics(sys, ref):
    """Computes metrics using the E2E script

    Args:
        sys (list): NLG system outputs
        ref (list): Reference outputs

    Returns:
        dict: dictionary with the metrics
    """

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
    try:
        eval_output = subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as err:
        print("return code", err.returncode)
        print("cmd", err.cmd)
        print("stderr:")
        print(err.stderr)
        print("stdout:")
        print(err.stdout)
        raise

    metric_names = ["BLEU", "NIST", "METEOR", "ROUGE_L", "CIDEr"]
    metric_scores = map(float, eval_output.stdout.decode().split()[1:])

    return dict(zip(metric_names, metric_scores))

