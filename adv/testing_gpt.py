# from transformers.models.gpt_neo import GPTNeoTokenizer, GPTNeoForCausalLM
import transformers
import torch
import local_models 
# Load tokenizer and model
tokenizer = transformers.GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125m')
model = local_models.GPTNeoForCausalLMOT.from_pretrained('EleutherAI/gpt-neo-125m')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

# Print the size of the vocabulary
vocab_size = model.config.vocab_size
print("Vocabulary Size:", vocab_size)


# Add special token to the tokenizer
tokenizer.add_special_tokens({'pad_token': 'PAD'})

# Encode the input text
# input_text = "The dog is very"#,"the time of the day is"]
input_text = ["The dog is very","the time of the day is"]
input_ids = tokenizer.batch_encode_plus(input_text,padding=True, truncation=True, return_tensors='pt').to(device) # if batch size is more than 1
# input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device) # if batch size is 1



output_tosk = [  464,  3290,   318,   845, 50257, 50257]
generated_text_out = tokenizer.decode(output_tosk, skip_special_tokens=False)

max_length = 20
 
# Generate text step-by-step
output_ids = input_ids['input_ids'].clone()
output_attention = input_ids['attention_mask'].clone()

for _ in range(max_length):
    # Feed the input to the model

    
    outputs = model(input_ids=output_ids,attention_mask=output_attention)
 
    
    # Get the logits for the last token
    logits = outputs.logits[:, -1, :]
 

    # Sample the next token
    probs = torch.softmax(logits, dim=-1)
    next_token_index = torch.argmax(probs, dim=-1) 
    next_token_temp = tokenizer.decode(next_token_index.item()) 
    
    next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
    
    # Append the generated token to the output sequence
    output_ids = torch.cat([output_ids, next_token], dim=-1)
    
    # Check if the generated token is an end-of-text token
    if next_token.item() == tokenizer.eos_token_id:
        break

# Decode the generated output
generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

# Print the generated text
print(generated_text)








# #prev method
# # Generate text using the model
# output = model.generate(input_ids, do_sample=True, min_length=20)

# # Decode the generated output
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# # Print the generated text
# print(generated_text)