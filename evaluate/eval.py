from seqeval.metrics import accuracy_score
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
import torch
import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel


def cvt_text_to_pred(text):
    if not text:
        return 'nan'
    pred_match = re.search(r'\d+(.\d+)', text)
    if pred_match is not None:
        pred = pred_match.group()
    else:
        print(text)
        pred = '0.0'
    return pred


def map_output(feature):

    label = cvt_text_to_pred(feature['output'])
    pred = cvt_text_to_pred(feature['out_text'])
    
    return {'label': label, 'pred': pred}

def test_fineval(args, model, tokenizer):

    dataset = load_dataset("christlurker/convqa_multiturn", split="test")
    # dataset = dataset.map(partial(test_mapping, args), load_from_cache_file=False)
    
    def collate_fn(batch):
        inputs = tokenizer(
            [f["prompt"] for f in batch], return_tensors='pt',
            padding=True, max_length=args.max_length,
            return_token_type_ids=False
        )
        return inputs
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    out_text_list = []
    log_interval = len(dataloader) // 5

    for idx, inputs in enumerate(tqdm(dataloader)):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        res = model.generate(**inputs, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        if (idx + 1) % log_interval == 0:
            tqdm.write(f'{idx}: {res_sentences[0]}')
        out_text = [o.split("Answer: ")[1] for o in res_sentences]
        out_text_list += out_text
        torch.cuda.empty_cache()
    
    dataset = dataset.add_column("out_text", out_text_list)
    dataset = dataset.map(map_output, load_from_cache_file=False)
    dataset = dataset.filter(lambda x: x['pred'] != 'nan')
    dataset = dataset.to_pandas()
    
    print(dataset)
    dataset.to_csv('tmp.csv')
    
    label = [float(d) for d in dataset['label']]
    pred = [float(d) for d in dataset['pred']]
    
    print('Accuracy: ', accuracy_score(label, pred))

    return dataset


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--max_length", type=int, default=1024)
    args = args.parse_args()

    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.model_parallel = True

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    print(f'pad: {tokenizer.pad_token_id}, eos: {tokenizer.eos_token_id}')

    # model = PeftModel.from_pretrained(model, args.peft_model)
    model = model.eval()
    with torch.no_grad():
        test_fineval(args, model, tokenizer)

    print('Done')