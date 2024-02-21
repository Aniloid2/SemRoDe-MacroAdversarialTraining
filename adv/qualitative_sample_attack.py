# load an attack from textattack 
# load model
# load a custom dataset which is the sample that i hardcode with the label
# attack the model using this hardcoded sample


import textattack 
from textattack.models.wrappers import ModelWrapper
import transformers
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper
# class CustomDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples
#         self.shuffled = False
#     def __getitem__(self, idx):
#         return self.samples[idx]
    
#     def __len__(self):
#         return len(self.samples)


model_max_length = 128
model_name = 'textattack/bert-base-uncased-ag-news'
# model_wrapper = ModelWrapper.from_pretrained(model_name)
model = transformers.models.bert.BertForSequenceClassification.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,model_max_length=model_max_length)
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
max_modification_rate = 0.3
cos_sim=0.5
sem_sim=0.8
no_cand=50
attack =  textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)



custom_sample = "Woods' Top Ranking on Line at NEC Invite (AP) AP - Tiger Woods already lost out on the majors. Next up could be his No. 1 ranking."
custom_label = 1
# custom_dataset = custom_dataset = CustomDataset([(custom_sample, custom_label)])

# custom_dataset = textattack.datasets.HuggingFaceDataset('ag_news', None,'train')
 
  
custom_dataset =   Dataset([(custom_sample,custom_label),(custom_sample,custom_label)] ) 
attacker = textattack.Attacker(attack, custom_dataset) 
results_iterable = attacker.attack_dataset()

print (results_iterable)