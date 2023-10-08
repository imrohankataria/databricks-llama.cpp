# LLAMA2 - 13b CPP and CODELLAMA - 34b CPP Notebooks

This repository contains two Databricks notebooks: LLAMA2 - 13b CPP and CODELLAMA - 34b CPP. Both notebooks implement variants of the LLM (Language Model) architecture and are designed to run on Databricks clusters. 

## LLAMA2 - 13b CPP

**Cluster Used:** NC6s_v3 ([Azure VM Pricing](https://azureprice.net/vm/Standard_NC6s_v3))

**References** - 
1. [djliden - Inference Experiments - LLaMA v2](https://github.com/djliden/inference-experiments/tree/main/llama2)
2. [abetlen - llama-cpp-python Issue #707](https://github.com/abetlen/llama-cpp-python/issues/707) (Do this step if LLAMA-CPP Doesn't work, install pathspec==0.11.0 via Add Library option using PyPi)

**Description:**

The LLAMA2 - 13b CPP notebook is an implementation of a variant of the LLM (Language Model) architecture. The code uses the LangChain framework, but you're free to use any other framework that suits your needs.

The notebook includes instructions for installing necessary libraries and tools, building the LLAMA-CPP-PYTHON with specific arguments, and running the LLM model with specific parameters. It also includes a prompt template and an example of how to run the model with a series of questions.

Please note that it's important to ensure the model path is correct for your system, and the values for n_gpu_layers, n_batch, and n_ctx are appropriate for your model and GPU VRAM pool.

**Notebook:** [LLAMA2 - 13b CPP Notebook](https://github.com/username/repo/blob/main/LLAMA2_13b_CPP_Notebook.py)

## CODELLAMA - 34b CPP

**Cluster Used:** NC12s_v3 ([Azure VM Pricing](https://azureprice.net/vm/Standard_NC12s_v3))

**Models Used:** 
1. [Phind-CodeLlama-34B-v2-GGUF](https://huggingface.co/TheBloke/Phind-CodeLlama-34B-v2-GGUF)
2. [CodeLlama-34B-GGUF](https://huggingface.co/TheBloke/CodeLlama-34B-GGUF)

**References** - 
1. [djliden - Inference Experiments - LLaMA v2](https://github.com/djliden/inference-experiments/tree/main/llama2)
2. [abetlen - llama-cpp-python Issue #707](https://github.com/abetlen/llama-cpp-python/issues/707) (Do this step if LLAMA-CPP Doesn't work, install pathspec==0.11.0 via Add Library option using PyPi)
3. [Langchain Code Understanding](https://python.langchain.com/docs/use_cases/code_understanding)

**Description:**

The LLAMA2 - 13b CPP notebook is an implementation of a variant of the LLM (Language Model) architecture. The code uses the LangChain framework, but you're free to use any other framework that suits your needs.

The notebook includes instructions for installing necessary libraries and tools, building the LLAMA-CPP-PYTHON with specific arguments, and running the LLM model with specific parameters. It also includes a prompt template and an example of how to run the model with a series of questions.

Please note that it's important to ensure the model path is correct for your system, and the values for n_gpu_layers, n_batch, and n_ctx are appropriate for your model and GPU VRAM pool.

**Notebook:** [CODELLAMA - 34b CPP Notebook](https://github.com/username/repo/blob/main/CODELLAMA_34b_CPP_Notebook.py)
