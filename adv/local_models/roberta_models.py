
from dataclasses import dataclass
from typing import Optional, Tuple

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)


from transformers.modeling_outputs import * # ModelOutput
# from transformers.models.bert.modeling_bert import * # BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import *# RobertaPreTrainedModel

from transformers.models.roberta.configuration_roberta import RobertaConfig


_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]

ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`RobertaTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""




@dataclass
class SequenceClassifierOutputAA(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    pooler_output: Optional[Tuple[torch.FloatTensor]] = None
    loss_original: Optional[torch.FloatTensor] = None
    KL_loss: Optional[torch.FloatTensor] = None
    MMD_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaClassificationHeadOT(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    # roberta sequence head classification
    def forward(self, features, **kwargs):
        # print (features.shape)
        # print ('featue shape',features[:, 0, :].shape)
        # sys.exit()
        # x = features[:, 0, :] 
        x = features # features[:, 0, :]  # take <s> token (equiv. to [CLS])
        pooled_dropout = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x,pooled_dropout

    


from collections import defaultdict
from transformers.models.bert.tokenization_bert import BertTokenizer
import json
import torch.nn.functional as F
import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__)) 
parent_path =  os.path.dirname(os.path.dirname(current_file_path)) 
sys.path.append(f'{parent_path}/src/TextDefender')  
from utils.luna import batch_pad

class RobertaForSequenceClassificationOT(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=True)
        # self.classifier = RobertaClassificationHead(config)
        
        # bert like classifier head
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)


        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        # self.dropout2 = nn.Dropout(classifier_dropout)
        # self.classifier2 = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) 

        # pooled_output = outputs[1]
        # logits,pooled_output = self.classifier(pooled_output)
        # pooled_output = self.classifier.dropout(pooled_output)

        # sequence_output = outputs[0]
        # logits = self.classifier(sequence_output)
         
        loss = None 
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        loss_total = loss
        
        return SequenceClassifierOutputAA(
            # loss=loss,
            # logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
            loss=loss_total,
            logits=logits,
            pooler_output=pooled_output,
            loss_original = loss,
            KL_loss = None,
            MMD_loss = None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



from transformers import RobertaTokenizer 


class ASCCRobertaModel(RobertaPreTrainedModel, nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.tokenizer = RobertaTokenizer.from_pretrained(config._name_or_path)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def build_nbrs_allannlp(self, nbr_file, vocab, alpha, num_steps):
        t2i = vocab.get_token_to_index_vocabulary("tokens")
        loaded = json.load(open(nbr_file))
        filtered = defaultdict(lambda: [], {})
        for k in loaded:
            if k in t2i:
                for v in loaded[k]:
                    if v in t2i:
                        filtered[k].append(v)
        nbrs = dict(filtered)

        nbr_matrix = []
        vocab_size = vocab.get_vocab_size("tokens")
        for idx in range(vocab_size):
            token = vocab.get_token_from_index(idx)
            nbr = [idx]
            if token in nbrs.keys():
                words = nbrs[token]
                for w in words:
                    assert w in t2i
                    nbr.append(t2i[w])
            nbr_matrix.append(nbr)
        nbr_matrix = batch_pad(nbr_matrix)
        self.nbrs = torch.tensor(nbr_matrix).cuda()
        self.max_nbr_num = self.nbrs.size()[-1]
        # self.weighting_param = nn.Parameter(torch.empty([self.num_embeddings, self.max_nbr_num], dtype=torch.float32),
        #                                     requires_grad=True).cuda()
        self.weighting_mask = self.nbrs != 0
        self.criterion_kl = nn.KLDivLoss(reduction="sum")
        self.alpha = alpha
        self.num_steps = num_steps

    def build_nbrs(self, nbr_file, vocab, alpha, num_steps, save_nbr_file):
        if not os.path.exists(save_nbr_file):
            # Instead of t2i = vocab.get_token_to_index_vocabulary("tokens")
            # Use a regular Python dictionary 
            t2i = vocab  # Assuming vocab is a dictionary now

            loaded = json.load(open(nbr_file))
            filtered = defaultdict(lambda: [], {})

            # Rest part remains same
            for k in loaded:
                if k in t2i:
                    for v in loaded[k]:
                        if v in t2i:
                            filtered[k].append(v)
            nbrs = dict(filtered)

            nbr_matrix = []
            # Instead of vocab_size = vocab.get_vocab_size("tokens")
            # Just use len() function
            vocab_size = len(vocab)  # Get the size of the vocab dictionary 

            for idx in range(vocab_size):
                # Instead of token = vocab.get_token_from_index(idx)
                # Find the token by the index using a list comprehension
                token = [k for k, v in vocab.items() if v == idx][0]  

                nbr = [idx]
                if token in nbrs.keys():
                    words = nbrs[token]
                    for w in words:
                        assert w in t2i
                        nbr.append(t2i[w])
                nbr_matrix.append(nbr)
            nbr_matrix = batch_pad(nbr_matrix)
            self.nbrs = torch.tensor(nbr_matrix).cuda()
            self.max_nbr_num = self.nbrs.size()[-1]
            # self.weighting_param = nn.Parameter(torch.empty([self.num_embeddings, self.max_nbr_num], dtype=torch.float32),
            #                                     requires_grad=True).cuda()
            self.weighting_mask = self.nbrs != 0
            self.criterion_kl = nn.KLDivLoss(reduction="sum")
            self.alpha = alpha
            self.num_steps = num_steps
            torch.save(self.nbrs, save_nbr_file)
        else:
            self.nbrs = torch.load(save_nbr_file)
            self.max_nbr_num = self.nbrs.size()[-1]
            # self.weighting_param = nn.Parameter(torch.empty([self.num_embeddings, self.max_nbr_num], dtype=torch.float32),
            #                                     requires_grad=True).cuda()
            self.weighting_mask = self.nbrs != 0
            self.criterion_kl = nn.KLDivLoss(reduction="sum")
            self.alpha = alpha
            self.num_steps = num_steps

    def forward(self, input_ids, attention_mask, token_type_ids= None, return_dict=True,output_hidden_states=True):
        # clean_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        # pooled_clean_output = self.dropout(clean_outputs[1])
        # clean_logits = self.classifier(pooled_clean_output)
        # return clean_logits, clean_logits
        print ('inptus',input_ids,attention_mask,token_type_ids)
        
        # 0 initialize w for neighbor weightings
        batch_size, text_len = input_ids.shape
        w = torch.empty(batch_size, text_len, self.max_nbr_num, 1).to(self.device).to(torch.float)
        nn.init.kaiming_normal_(w)
        w.requires_grad_()
        optimizer_w = torch.optim.Adam([w], lr=1, weight_decay=2e-5)

        # 1 forward and backward to calculate adv_examples
        # print ('input emb',self.get_input_embeddings())
        # print ('input ids',input_ids)
        # print ('nbrs',self.nbrs)
        # print ('nbrs',self.nbrs[input_ids])
        # sys.exit()
        input_nbr_embed = self.get_input_embeddings()(self.nbrs[input_ids])
        weighting_mask = self.weighting_mask[input_ids]
        # here we need to calculate clean logits with no grad, to find adv examples
        with torch.no_grad():
            clean_outputs = self.roberta(input_ids, attention_mask)
            clean_sequence_output = clean_outputs[0]
            clean_logits = self.classifier(clean_sequence_output)
        
        self.num_steps = 30
        for _ in range(self.num_steps):
            optimizer_w.zero_grad()
            with torch.enable_grad():
                w_after_mask = weighting_mask.unsqueeze(-1) * w + ~weighting_mask.unsqueeze(-1) * -999
                embed_adv = torch.sum(input_nbr_embed * F.softmax(w_after_mask, -2) * weighting_mask.unsqueeze(-1), dim=2)

                adv_outputs = self.roberta(attention_mask=attention_mask, inputs_embeds=embed_adv)
                # print ('last_hidden_state',adv_outputs.last_hidden_state)
                # sys.exit()
                adv_sequence_output = adv_outputs[0]
                adv_logits = self.classifier(adv_sequence_output)
                # print ('adv logits',adv_logits,adv_logits[0].shape,adv_logits[1].shape )
                # print ('clean logits',clean_logits,clean_logits[0].shape,clean_logits[1].shape) 
                # sys.exit()
                # print ('clean_logits',clean_logits[0].shape)
                # print ('adv logtis shape',adv_logits[0].shape)

                # prev
                # actual_adv_logits = adv_logits[0] 
                # actual_clean_logits = clean_logits[0]
                # adv_logits_after = actual_adv_logits[:,0,:] 
                # clean_logits_after = actual_clean_logits[:,0,:]


                # print ('adv_logits',adv_logits)
                # print ('clean_logits_after',clean_logits_after.shape)
                # print ('adv_logits_after shape',adv_logits_after.shape) 
                # adv_loss = - self.criterion_kl(F.log_softmax(adv_logits_after, dim=1), F.softmax(clean_logits_after.detach(), dim=1))
                
                adv_loss = - self.criterion_kl(F.log_softmax(adv_logits, dim=1), F.softmax(clean_logits.detach(), dim=1))
                
                loss_sparse = (-F.softmax(w_after_mask, -2) * weighting_mask.unsqueeze(-1) * F.log_softmax(w_after_mask, -2)).sum(-2).mean()
                loss = adv_loss + self.alpha * loss_sparse
                print ('loss ascc inner roberta',loss) 
            loss.backward(retain_graph=True)
            optimizer_w.step()
         
        optimizer_w.zero_grad()
        self.zero_grad()

        # 2 calculate clean data logits
        clean_outputs = self.roberta(input_ids, attention_mask)
        clean_sequence_output = clean_outputs[0]
        clean_logits = self.classifier(clean_sequence_output)
        # print ('clean_logits',clean_logits)

        # actual_clean_logits = clean_logits[0]
        # clean_logits = actual_clean_logits[:,0,:]

        # 3 calculate convex hull of each embedding
        w_after_mask = weighting_mask.unsqueeze(-1) * w + ~weighting_mask.unsqueeze(-1) * -999
        embed_adv = torch.sum(input_nbr_embed * F.softmax(w_after_mask, -2) * weighting_mask.unsqueeze(-1), dim=2)

        # 4 calculate adv logits
        adv_outputs = self.roberta(attention_mask=attention_mask, inputs_embeds=embed_adv)
        adv_sequence_output = adv_outputs[0]
        adv_logits = self.classifier(adv_sequence_output)
        # print ('adv_logits',adv_logits)
        # print ('clean_logits',clean_logits)
         
        # actual_adv_logits = adv_logits[0]
        # adv_logits = actual_adv_logits[:,0,:]

        # print ('logits',clean_logits.shape, adv_logits.shape)
        # sys.exit()
        return SequenceClassifierOutputAA(
                loss=None,
                logits=clean_logits,
                pooler_output=None,
                loss_original = None,
                KL_loss = adv_logits, # adv logits here, idealy we have a different object class for this with adv_logit entry specificaly for ASCC (TODO)
                MMD_loss = None,
                hidden_states=clean_outputs.hidden_states, # not needed but passing anyway because after it returns i have a call to hidden states
                attentions=clean_outputs.attentions, # not needed but passing anyway because after it returns i have a call to hidden states 
            )

   
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)

class RobertaForSequenceClassificationDSRM(RobertaPreTrainedModel,nn.Module):
    def __init__(self, args, model_name, config):
        super(RobertaForSequenceClassificationDSRM, self).__init__(config)
        # self.args = args
        self.config = config
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        
        
        self.dsrm_weights = nn.Linear(args.batch_size, 1, bias=False)
        nn.init.constant_(self.dsrm_weights.weight, 1)
        

    def refresh_weights(self):
        nn.init.constant_(self.dsrm_weights.weight, 1)

    # def forward(self, model_inputs, labels, requir_weight=False):
    def forward(self, **model_inputs):
        sequence_output_object = self.model(**model_inputs)#.logits  # (1) backward
        return sequence_output_object
        # losses = F.cross_entropy(logits, labels.squeeze(-1), reduction='none')
        # losses = torch.sum(logits, dim = -1)
        # if requir_weight:
        #     loss = self.weights(losses) / self.args.bsz
        #     return loss
        # else:
        #     loss = torch.mean(losses)
        #     return loss

    # def train_forward(self, model_inputs, labels):
    def train_forward(self, **model_inputs):#, labels):
        # logits = self.model(**model_inputs).logits  # (1) backward
        sequence_output_object = self.model(**model_inputs)
        # loss = F.cross_entropy(logits, labels.squeeze(-1), reduction='none')
        # return loss, logits
        return sequence_output_object