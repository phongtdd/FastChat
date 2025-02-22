## Install

### Method 1: With pip

```bash
pip3 install "fschat[model_worker,webui]"
```
## Fine-tuning
### Data
In folder data, there are dataset of VuiLibGen with top 1 RAG results
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