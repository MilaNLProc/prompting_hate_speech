Hey Language Models

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

Features
--------

Encoder models

    from encoder_lms import prompting_enc_hs

    prompting_enc_hs.predict(["I hate this woman", "let's se if this muslim can eat pork"])

    >> ["hate", "not-hate"]
    
Instruction fine-tuned models

    from encoder_lms import prompting_inst_hs
  
    prompting_enc_hs.predict(["I hate this woman", "let's se if this muslim can eat pork"])
    >> ["hate", "not-hate"]

License
-------
`GNU GPLv3 <https://choosealicense.com/licenses/gpl-3.0/>`_
