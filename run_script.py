from instruction_fine_tuned_lms import prompting
import pandas as pd

#inst_lms = prompting("flant5")
#prompt_template = "Classify this text as hate or non-hate. Text:"
#output_indicator = "Answer:"

#text = "I hate you"
#texts = ["I love you", "I hate you"]
#df = pd.DataFrame()
#df['text'] = texts

#print(inst_lms.predict(prompt_template, output_indicator, df))


# Encoder-based models with prompting

from encoder_lms import prompting

enc_lms = prompting("deberta-large")
prompt_template = "This text is"
verb_h = "toxic"
verb_nh = "love"

text = "My lovely cat, I love you so much, you are my favorite"
texts = ["Shut your dumbass up bitch we all know you a hoe", "My lovely cat, I love you so much, you are my favorite", "HATE YOU BITCH SHIT!"]
#df = pd.DataFrame()
#df['text'] = texts
print(enc_lms.predict(prompt_template, verb_h, verb_nh, texts))