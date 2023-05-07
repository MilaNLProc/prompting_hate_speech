import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)

class prompting:

        def __init__(self, model="flant5"):
            if model == "flant5":
                self.checkpoint = "google/flan-t5-xl"
            elif model == "mt0":
                self.checkpoint = "bigscience/mt0-xxl"
            else:
                raise Exception("Select one of the following models: flant5 or mt0")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint, torch_dtype="auto", device_map="auto")
            
        def build_prompt(self, prompt_template: str, output_indicator: str, input_text: str):
            if prompt_template:
                prompt = f"{prompt_template} {input_text} {output_indicator}"
            else:
                raise NotImplementedError("Insert a template")
            return prompt

        def predict(self, prompt_template: str, output_indicator: str, data):
            with torch.no_grad():

                if isinstance(data, str):
                    texts = [self.build_prompt(prompt_template, output_indicator, data)]
                elif isinstance(data, pd.DataFrame):
                    texts = data['text'].tolist()
                    texts = [self.build_prompt(prompt_template, output_indicator, t) for t in texts]
                elif isinstance(data, list) and all(isinstance(t, str) for t in data):
                    texts = [self.build_prompt(prompt_template, output_indicator, t) for t in data]
                else:
                    raise ValueError('Input data must be either a string or a pandas DataFrame.')

                raw_dataset = Dataset.from_dict({"text": texts})

                proc_dataset = raw_dataset.map(
                    lambda x: self.tokenizer(
                        x["text"], truncation=True
                    ),  # truncate by default to maximum model length
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                    remove_columns=["text"],
                )
                
                proc_dataset.set_format("torch")

                loader = torch.utils.data.DataLoader(
                    proc_dataset,
                    shuffle=False,
                    batch_size=512, #default
                    collate_fn=DataCollatorWithPadding(self.tokenizer),
                )

                predictions = []
                for i, batch in tqdm(
                    enumerate(loader), desc=self.checkpoint, total=len(texts) // 512
                ):
                    inputs = {k: v.to(self.model.device) for k, v in batch.items()}
                    outputs = self.model.generate(**inputs)

                    decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    predictions.extend(decoded)

                predictions =  list(map(str.lower, predictions))
            return predictions