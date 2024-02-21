# need to create a copy of the .tsv file?
# does this file need to be stripped down?
# maybe create a custom dataset from the Dataset dataloader



import textattack
import transformers

model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-mnli")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-mnli")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
dataset = textattack.datasets.HuggingFaceDataset("mnli", split="train")

# Attack 20 samples with CSV logging and checkpoint saved every 5 interval
attack_args = textattack.AttackArgs(
    num_examples=20,
    log_to_csv="log.csv",
    checkpoint_interval=5,
    checkpoint_dir="checkpoints",
    disable_stdout=True
)

attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()
