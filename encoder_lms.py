import pandas as pd
#from typing import List
import torch
#from datasets import Dataset
#from tqdm import tqdm

from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader

class prompting:
        def __init__(self, model="roberta-base"):
            if model == "roberta-base":
                self.checkpoint = ("roberta","roberta-base")
            elif model == "roberta-large":
                self.checkpoint = ("roberta","roberta-large")
            elif model == "bert":
                self.checkpoint = ("bert","bert-base-uncased")
            elif model == "deberta-base":
                self.checkpoint = ("deberta-v3","microsoft/deberta-v3-base")
            elif model == "deberta-large":
                self.checkpoint = ("deberta-v3","microsoft/deberta-v3-large")
            elif model == "xlm-roberta":
                self.checkpoint = ("xlm-roberta-base","xlm-roberta-base")
                
            else:
                raise Exception("Select one of the following models: roberta-base, roberta-large, bert, deberta-base, deberta-large, xlm")
             
        def predict(self, template, verb_h, verb_nh, data):
            plm, tokenizer, model_config, WrapperClass = load_plm(self.checkpoint[0], self.checkpoint[1])

            promptTemplate = ManualTemplate(
                 text = f'{{"placeholder":"text_a"}} {template} {{"mask"}}',
                 tokenizer = tokenizer,
                 )
        
            classes = ["1", "0"]
            
            label_words = {
                "1": verb_h,
                "0": verb_nh
                }

            if isinstance(data, pd.DataFrame):
                dataset = [InputExample(guid=i, text_a=txt) for i, txt in enumerate(data["text"].tolist())]
            elif isinstance(data, str):
                dataset = [InputExample(guid=0, text_a=data)]
            elif isinstance(data, list) and all(isinstance(t, str) for t in data):
                dataset = [InputExample(guid=0, text_a=txt) for i, txt in enumerate(data)]
            else:
                raise TypeError("The data parameter must be a pandas DataFrame or a string")

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
            return predictions
            
