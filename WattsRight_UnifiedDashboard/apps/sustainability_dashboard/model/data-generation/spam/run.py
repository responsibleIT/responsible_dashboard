import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from evaluation import evaluate_thresholds
from plot import plot_comprehensive_results

models = [
    # {
    #     "model_name": "mshenoda/roberta-spam",
    #     "folder": "mshenoda_roberta-spam",
    # },
    # {
    #     "model_name": "AntiSpamInstitute/spam-detector-bert-MoE-v2.2",
    #     "folder": "AntiSpamInstitute_spam-detector-bert-MoE-v2.2",
    # },
    # {
    #     "model_name": "fzn0x/bert-spam-classification-model",
    #     "folder": "fzn0x_bert-spam-classification-model",
    # },
    # {
    #     "model_name": "mrm8488/bert-tiny-finetuned-sms-spam-detection",
    #     "folder": "mrm8488_bert-tiny-finetuned-sms-spam-detection",
    # },
    # {
    #     "model_name": "satish860/sms_spam_detection-manning",
    #     "folder": "satish860_sms_spam_detection-manning",
    # },
    # {
    #     "model_name": "SalehAhmad/roberta-base-finetuned-sms-spam-ham-detection",
    #     "folder": "SalehAhmad_roberta-base-finetuned-sms-spam-ham-detection",
    # },
    # {
    #     "model_name": "mrm8488/bert-tiny-finetuned-enron-spam-detection",
    #     "folder": "mrm8488_bert-tiny-finetuned-enron-spam-detection",
    # },
    # {
    #     "model_name": "skandavivek2/spam-classifier",
    #     "folder": "skandavivek2_spam-classifier",
    # },
    # {
    #     "model_name": "HJOK/task2_deberta_spamMLM_v1",
    #     "folder": "HJOK_task2_deberta_spamMLM_v1",
    # },
    # {
    #     "model_name": "sureshs/distilbert-large-sms-spam",
    #     "folder": "sureshs_distilbert-large-sms-spam",
    # },
    # {
    #     "model_name": "leeboykt/sms_spam_detection",
    #     "folder": "leeboykt_sms_spam_detection",
    # },
    # {
    #     "model_name": "cesullivan99/sms-spam-weighted",
    #     "folder": "cesullivan99_sms-spam-weighted",
    # },
    # {
    #     "model_name": "Ngadou/bert-sms-spam-dectector",
    #     "folder": "Ngadou_bert-sms-spam-dectector",
    # },
    # {
    #     "model_name": "smtriplett/bert_finetuned_sms_spam_pp5_1",
    #     "folder": "smtriplett_bert_finetuned_sms_spam_pp5_1",
    # },
    # {
    #     "model_name": "WonderfulAnalytics/distilbert-base-uncased-finetuned-spam",
    #     "folder": "WonderfulAnalytics_distilbert-base-uncased-finetuned-spam",
    # },
    # {
    #     "model_name": "nelsi/test_spam",
    #     "folder": "nelsi_test_spam",
    # },
# {
#     "model_name": "0x7o/roberta-base-spam-detector",
#     "folder": "0x7o_roberta-base-spam-detector",
# },
    # {
    #     "model_name": "mbruton/spa_mBERT",
    #     "folder": "mbruton_spa_mBERT",
    # },
    {
        "model_name": "dungnt/sms_spam_detection",
        "folder": "dungnt_sms_spam_detection",
    },
    {
        "model_name": "wesleyacheng/sms-spam-classification-with-bert",
        "folder": "wesleyacheng_sms-spam-classification-with-bert",
    },
]


VALIDATION_PATH = "datasets/benchmark.csv"

thresholds = [i * 0.1 for i in range(0, 101)]

for model in models:
    model_name = model["model_name"]
    folder = model["folder"]

    if not os.path.exists(f"runs/{folder}"):
        os.makedirs(f"runs/{folder}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        results = evaluate_thresholds(VALIDATION_PATH, thresholds, tokenizer, model)
        results.to_csv(f"runs/{folder}/pruning_results.csv", index=False)

        plot_comprehensive_results(results, folder=folder)
    except Exception as e:
        print(f"Error processing model {model_name}: {e}")
        continue