"""
HuggingFace Model Wrapper
--------------------------
"""

import torch
import transformers

import textattack
from textattack.models.helpers import T5ForTextToText
from textattack.models.tokenizers import T5Tokenizer

from .pytorch_model_wrapper import PyTorchModelWrapper

torch.cuda.empty_cache()


class HuggingFaceModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer):
        assert isinstance(
            model, (transformers.PreTrainedModel, T5ForTextToText)
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
                T5Tokenizer,
            ),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

    def embedding_inference(self, inputs_dict):#,text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # # Default max length is set to be int(1e30), so we force 512 to enable batching.
        # max_length = (
        #     512
        #     if self.tokenizer.model_max_length == int(1e30)
        #     else self.tokenizer.model_max_length
        # )

        # print ('token ier input ',text_input_list)
        # inputs_dict = self.tokenizer(
        #     text_input_list,
        #     add_special_tokens=True,
        #     padding="max_length",
        #     max_length=max_length,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        model_device = next(self.model.parameters()).device
        inputs_dict = inputs_dict.to(model_device) # this is likely wrong since we need mask usually.
        
        # in_embeddings = embeddings.to(model_device) # this is likely wrong since we need mask usually.
        # attention_mask = (in_embeddings != 0).long().to(model_device) 
        # token_type_ids = torch.zeros_like(in_embeddings).to(model_device)

        # inputs_dict = {
        #     "input_ids": in_embeddings ,  # Add batch dimension if needed
        #     "attention_mask": attention_mask ,  # Add batch dimension if needed
        #     "token_type_ids": token_type_ids ,  # Add batch dimension if needed
        # }
        input_tokens = inputs_dict['input_ids']
        # input_indices_copy = input_tokens.clone().detach().requires_grad_(True)
         
        # print ('embeddings before',self.model.get_input_embeddings())
        self.model.get_input_embeddings().requires_grad = True
        # print ('embeddings after',self.model.get_input_embeddings(),self.model.get_input_embeddings().requires_grad)
        
        word_embedding_layer = self.model.get_input_embeddings()
        embeddings = word_embedding_layer(input_tokens)
        

        # embeddings = self.model.bert.embeddings(input_tokens) ## at the moment only works with bert, we need to make 
        # it work with roberta
        # print ('embedding init',embedding_init)
        # print ('embeddings',embeddings, embeddings.grad) 
        
        # inputs_dict = {
        #     "input_ids": input_tokens ,  # Add batch dimension if needed
        #     "attention_mask": inputs_dict['attention_mask'] ,  # Add batch dimension if needed
        #     "token_type_ids": inputs_dict['token_type_ids'] ,  # Add batch dimension if needed
        # }

        inputs_dict = {
            "inputs_embeds": embeddings ,  # Add batch dimension if needed
            "attention_mask": inputs_dict['attention_mask'] ,  # Add batch dimension if needed
            "token_type_ids": inputs_dict['token_type_ids'] ,  # Add batch dimension if needed
        }

        # retaining gradiant for the non leaf tensor node
        embeddings.retain_grad()

        # with torch.no_grad():
        outputs = self.model(**inputs_dict)
 
        
        

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs,embeddings
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits,embeddings,input_tokens





    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
