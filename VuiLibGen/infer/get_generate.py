import argparse
import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from functools import partial
from peft import PeftModel 
from post import closest_lib
import re

# Prompt template to use with Llama 2 or Vicuna
llama2_chat_template=dict(
        SYSTEM=(
            '[INST] <<SYS>>\n You are a helpful, respectful and honest '
            'assistant. Always answer as helpfully as possible, while being '
            'safe. Your answers should not include any harmful, unethical, '
            'racist, sexist, toxic, dangerous, or illegal content. Please '
            'ensure that your responses are socially unbiased and positive in '
            'nature.\n{system}\n<</SYS>>\n [/INST] '),
        INSTRUCTION='[INST] {input} [/INST]',
        SEP='\n')

vicuna_template = dict(
        SYSTEM=('A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n '),
        INSTRUCTION=('USER: {input} ASSISTANT:'),
        SEP='\n')

# Function to preprocess dataset
def map_fn(example):
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            input += msg['value']
        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation}

def template_fn(example, template):
    conversation = example.get('conversation', [])
    
    for i, single_turn_conversation in enumerate(conversation):
        input_text = single_turn_conversation.get('input', '') or ''
        input_text = template["INSTRUCTION"].format(input=input_text, round=i + 1)

        system_text = single_turn_conversation.get('system', '')
        if system_text:
            system_text = template["SYSTEM"].format(system=system_text)
            input_text = system_text + input_text

        single_turn_conversation['input'] = input_text

        if "SUFFIX" in template:
            output_text = single_turn_conversation.get('output', '')
            output_text += template["SUFFIX"]
            single_turn_conversation['output'] = output_text

        single_turn_conversation['need_eos_token'] = not template.get("SUFFIX_AS_EOS", False)
        single_turn_conversation['sep'] = template.get("SEP", "")

    return example


def template_map_fn_factory(template):
    return partial(template_fn, template=template)

# Function to load data
def load_data(file_path, template):
    dataset = load_dataset('json', data_files={'test': file_path})

    # Apply mapping functions
    dataset = dataset.map(map_fn, num_proc=32)  
    dataset = dataset.map(template_map_fn_factory(template), num_proc=32)
    dataset = dataset.remove_columns('conversations')
    return dataset

# Function to get the generate library of model
def inference(model, tokenizer, prompt: str, device) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt",max_length=512, padding=True, truncation=True).input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output_ids[0][(len(input_ids[0])+1):], skip_special_tokens=True)

# Function to get the local post preprocess
def local_search(response):
    match = re.match(r"^(.*?\s)([\w:.+-]+)\.$", response)
    if match:
        return match.group(1) + closest_lib(match.group(2))
    return response

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(
        base_model,
        args.finetuned_model,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_model)

    correct = 0

    if args.datapath:
        # Define the prompt template
        if args.model_type == 'llama2':
            template = llama2_chat_template
        elif args.model_type == 'vicuna':
            template = vicuna_template
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        dataset = load_data(args.datapath, template)['test']

        print("Running inference on test set...")
        results = []
        correct = 0

        # Generate the output
        for i in range(0, len(dataset)):
            ex = dataset['conversation'][i]
            input = ex[0]['input']
            output = ex[0]['output']
            response = inference(model, tokenizer, input, device)
            # Post-processing library name
            response = local_search(response)
            if output.strip().lower() == response.strip().lower():
                    correct += 1
            results.append({"input": input, "response": response, "output": output})
        acc = (correct/len(dataset))*100
        # Save results
        with open("test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        print("Inference completed. Results saved to test_results.json")
        sys.exit()
    return acc if args.datapath else None 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, required=True, help="Path to model")
    parser.add_argument("--finetuned_model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--model_type", type=str, default='llama2', help = "Type of model ")
    parser.add_argument("--datapath", type=str, required=True, help="Path to a data")
    args = parser.parse_args()
    main(args)
