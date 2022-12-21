import pandas as pd
import numpy as np
import argparse
import torch
import sys
import ast
import os
import mlflow

from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader

from sklearn.metrics import classification_report
    
def openPrompt(data, model, path_model, label_words):
    
    classes = ["1", "0"]

    dataset =[ 
            InputExample(guid = i,
                         text_a = txt,
                         label = lab)     
                for i, (txt, lab) in enumerate(zip(data["text"].tolist(), data["label"].tolist()))
    ] 

    plm, tokenizer, model_config, WrapperClass = load_plm(model, path_model)

    promptTemplate = ManualTemplate(
        text = '{"placeholder":"text_a"} This text is {"mask"}',
        tokenizer = tokenizer,
    )

    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = label_words,
        tokenizer = tokenizer,
    )

    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )

    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass
    )

    promptModel = promptModel.cuda()

    predictions = []

    promptModel.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.cuda()
            logits = promptModel(batch)
            preds = torch.argmax(logits, dim = -1)
            predictions.extend([classes[p] for p in preds.cpu().numpy().tolist()])

    predictions = list(np.float_(predictions))

    report = classification_report(data['label'].tolist(), predictions, digits=2, output_dict=True)
    report.update({"accuracy": {"precision": None, "recall": None, "f1-score": report["accuracy"], "support": report['macro avg']['support']}})

    evaluation_metrics_df = pd.DataFrame(report).transpose()
    evaluation_metrics_df = evaluation_metrics_df.round(4)
    print(evaluation_metrics_df)

    return evaluation_metrics_df, report

def main():

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["SEED"] = "22"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    parser = argparse.ArgumentParser()
    
    # Requiered parameters
    parser.add_argument("--data_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data file. Should contain the .tsv file for the Hate Speech dataset.")
    
    parser.add_argument("--transformer", 
                        default="bert-based-cased",
                        required=True,
                        help="Transformer pre-trained model selected in the list: bert-based-cased, \
                        roberta-base")         

    parser.add_argument("--class1_prompt", 
                        nargs="+", 
                        default=[])
    
    parser.add_argument("--class0_prompt", 
                        nargs="+", 
                        default=[])

    parser.add_argument("--output_file",
                        default="ouput_file",
                        type=str,
                        required=True,
                        help="The output file where the model predictions will be written.")     
  
    args = parser.parse_args()
    data_file = args.data_file
    class1_prompt = args.class1_prompt
    class0_prompt = args.class0_prompt
    output_file = args.output_file
    transformer = args.transformer

    path_dataset = "/mnt/beegfs/fmplaza/22PromptHateSpeech/datasets/" + data_file
    path_model = "/mnt/beegfs/fmplaza/22PromptHateSpeech/models/" + transformer

    dataset = pd.read_csv(path_dataset, sep="\t")

    label_words = {
            "1": class1_prompt,
            "0": class0_prompt
    }

    f = open(output_file, "w")
    f.write("*"*100+"\n")
    f.write("dataset: " + data_file+"\n")
    f.write("transformer_model: " + transformer+"\n")
    f.write("label_words" + str(label_words)+"\n")
    f.write("*"*100+"\n\n")

    print(transformer.split("-")[0])
    print(path_model)

    evaluation_metrics_df, report = openPrompt(dataset, transformer.split("-")[0], path_model, label_words)

    dfAsString = evaluation_metrics_df.to_string(header=True, index=True)
    f.write("Evaluation metrics"+"\n")
    f.write(dfAsString)

    nameRun = output_file.split("/")[-1].split(".")[0]

    mlflow.set_experiment("OpenPromptAlbertDynabenchv")

    with mlflow.start_run(run_name=nameRun):
        mlflow.log_param("class0 prompt", label_words["0"])
        mlflow.log_param("class1 prompt", label_words["1"])

        mlflow.log_metric("precision_0", report['0']["precision"])
        mlflow.log_metric("recall_0", report['0']["recall"])
        mlflow.log_metric("f1_0", report['0']["f1-score"])
        mlflow.log_metric("support_0", report['0']["support"])

        mlflow.log_metric("precision_1", report['1']["precision"])
        mlflow.log_metric("recall_1", report['1']["recall"])
        mlflow.log_metric("f1_1", report['1']["f1-score"])
        mlflow.log_metric("support_1", report['1']["support"])

        mlflow.log_metric("macro-avg precision", report['macro avg']['precision'])
        mlflow.log_metric("macro-avg recall", report['macro avg']['recall'])
        mlflow.log_metric("macro-avg f1", report['macro avg']['f1-score'])

if __name__ == "__main__":
    main()