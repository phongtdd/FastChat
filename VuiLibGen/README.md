## Install

### Method 1: With pip

```bash
pip3 install "fschat[model_worker,webui]"
```
## Fine-tuning
### Data
In folder data, there are dataset of VuiLibGen with top 1 RAG results
Additionally, it also includes the `maven_corpus.json`, as the descriptions of maven packages.

### Install dependency in origin repors
```bash
cd FastChat
pip3 install -e ".[train]"
```
### Install extra dependency
```bash
cd VuiLibGen
pip3 install -e requirements.txt
```
### Finetune
Use the script in the `scripts` folder to fine-tune the model.  
Update `Path-to-model`, `data-path`, and `eval-data-path` with the actual paths.

# Infer
## Install dependency
```
cd infer
pip3 install -e reuqirements.txt
```
## Set up local search 
In file post.py, replace maven path with actual maven_corpus path in your folder
## Testing 
Use the script in the `scripts` folder to test the model.
Update `model_path`, `finetuned_model_path`, `data_path` with the actual paths. Specify the model_type you want to test, e.g., llama2 or vicuna.