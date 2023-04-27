from prompting_hate_speech.instruction_llms import instruction_LLMS
from prompting_hate_speech.encoder_lms import encoder_LLMS
import pandas as pd

hc = instruction_LLMS("flant5")

prompt_template = "Classify this text as hate or non-hate. Text:"
output_indicator = "Answer:"
#text = "I love you"
#hc.predict_text(prompt_template, output_indicator, text)

texts = ["hateful", "love"]
df = pd.DataFrame()
df['text'] = texts
#print(hc.predict(prompt_template, output_indicator, df))

# Encoder-based models with prompting

enc_p = encoder_LLMS("deberta-base")
prompt_template = "This text is"
verb_h = "hateful"
verb_nh = "love"
print(enc_p.predict(prompt_template, verb_h, verb_nh, df))