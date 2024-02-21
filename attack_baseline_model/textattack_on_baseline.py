import transformers
from textattack.models.wrappers import HuggingFaceModelWrapper
import textattack
from datasets import load_dataset
import argparse
# Original_40_run_prime
# ID = 'ACT_40_run_prime'

parser = argparse.ArgumentParser()
parser.add_argument("--id")
args = parser.parse_args()
print(args.id)

ID = args.id

# origin_folder = f"../original_results/GLUE/MRPC/ALBERT/{ID}"
origin_folder = f"../original_results/GLUE/MRPC/ALBERT/{ID}"
# https://huggingface.co/textattack
model = transformers.AutoModelForSequenceClassification.from_pretrained(origin_folder)

attack_bool = {"attack":True}

model.config.update(attack_bool)
print ('model configuration',model.config)
tokenizer = transformers.AutoTokenizer.from_pretrained(origin_folder)

# We wrap the model so it can be used by textattack


# model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MRPC")
# tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MRPC")



model_wrapper = HuggingFaceModelWrapper(model, tokenizer)


# dataset_load = load_dataset("glue", "mrpc", split="test")
# dataset = textattack.datasets.HuggingFaceDataset(dataset_load)


dataset = textattack.datasets.HuggingFaceDataset("glue", "mrpc", split="test")



attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

attack_args = textattack.AttackArgs(
    num_examples=1000,
    log_to_csv=f"{ID}.csv",
    log_to_txt=f"{ID}.txt",
    disable_stdout=True,
    # parallel=True,
    # num_workers_per_device=8,
)

attacker = textattack.Attacker(attack, dataset, attack_args)



attacker.attack_dataset()


# print (model_wrapper)
# print (dataset)
