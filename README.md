# Respectful or Toxic? Using Zero-Shot Learning with Language Models to Detect Hate Speech

This repository contains the code of the paper Respectful or Toxic? Using Zero-Shot Learning with Language Models to Detect Hate Speech.

License
-------

Code comes from HuggingFace and thus our License is an MIT license.

For models restrictions may apply on the data (which are derived from existing datasets) or Twitter (main data source). We refer users to the original licenses accompanying each dataset and Twitter regulations.

Installing
----------

    !git clone https://github.com/MilaNLProc/prompting_hate_speech
    !cd prompting_hate_speech
    pip install -e .

**Important**: If you want to use CUDA you need to install the correct version of
the CUDA systems that matches your distribution, see `PyTorch <https://pytorch.org/get-started/locally/>`__.

Instructions to run the code
--------

Features
--------

Encoder models

    from encoders import prompting

    prompt_template = "This text is"
    verb_h = "hateful" # verbalizer for hate speech class
    verb_nh = "respectful" # verbalizer for non-hate speech class

    enc_p = prompting("deberta-base")

    enc_p.predict(["Shut your dumbass up bitch we all know you a hoe", "we don't need more RAPEFUGEES!"])

    >> ["hate", "not-hate"]
    
Instruction fine-tuned models

    from llms import prompting

    prompt_template = "Classify this text as hate or non-hate. Text:"
    output_indicator = "Answer:"

    llms_p = prompting("flant5")
  
    llms_p.predict(prompt_template, output_indicator, ["Shut your dumbass up bitch we all know you a hoe", "we don't need more RAPEFUGEES!"])

    >> ["hate", "not-hate"]

Input = dataframe || text || list of texts

Note: the instances used as examples are from a hate speech corpus. We did not create them.

License
-------
`GNU GPLv3 <https://choosealicense.com/licenses/gpl-3.0/>`_
