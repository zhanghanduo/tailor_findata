from datasets import load_from_disk

dataset = load_from_disk("processed_data/convfinqa_sharegpt_refine")
dataset.push_to_hub("christlurker/finqa_sharegpt")