Hey Language Models

License
-------

Code comes from HuggingFace and thus our License is an MIT license.

For models restrictions may apply on the data (which are derived from existing datasets) or Twitter (main data source). We refer users to the original licenses accompanying each dataset and Twitter regulations.

Installing
----------

.. code-block:: bash


    !git clone https://github.com/MilaNLProc/hate-ita/
    !cd hate-ita
    pip install -e .

**Important**: If you want to use CUDA you need to install the correct version of
the CUDA systems that matches your distribution, see `PyTorch <https://pytorch.org/get-started/locally/>`__.

Features
--------

Encoder models

.. code-block:: python


    from encoder_lms import prompting_enc_hs

    prompting_enc_hs.predict(["ti odio", "come si fa a rompere la lavatrice porca puttana"])

    >> ["hate", "not-hate"]
    
Instruction fine-tuned models

    from encoder_lms import prompting_inst_hs
  
    prompting_enc_hs.predict(["ti odio", "come si fa a rompere la lavatrice porca puttana"])

    >> ["hate", "not-hate"]

License
-------
`GNU GPLv3 <https://choosealicense.com/licenses/gpl-3.0/>`_
