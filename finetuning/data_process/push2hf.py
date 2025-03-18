from datasets import load_from_disk
import huggingface_hub

dataset = load_from_disk("processed_data/convfinqa_sharegpt_refine")
dataset.push_to_hub("christlurker/findata_test")
