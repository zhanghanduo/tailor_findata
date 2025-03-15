from datasets import load_from_disk
import huggingface_hub

dataset = load_from_disk("processed_data/convfinqa_sharegpt_refine")
dataset.push_to_hub("christlurker/finqa_sharegpt")

huggingface_hub.create_tag("christlurker/finqa_sharegpt", tag="v1.1", repo_type="dataset")