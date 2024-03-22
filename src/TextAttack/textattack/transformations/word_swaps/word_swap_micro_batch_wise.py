"""
Word Swap by Gradient
-------------------------------

"""
import torch

import textattack
from textattack.shared import utils
from textattack.shared.validators import validate_model_gradient_word_swap_compatibility
import sys
from .word_swap import WordSwap
from textattack.shared.utils import device
# from textattack.shared.utils import device, words_from_text
import numpy as np
class WordSwapMicroBW(WordSwap):
    """Uses the model's gradient to suggest replacements for a given word.

    Based off of HotFlip: White-Box Adversarial Examples for Text
    Classification (Ebrahimi et al., 2018).
    https://arxiv.org/pdf/1712.06751.pdf

    Arguments:
        model (nn.Module): The model to attack. Model must have a
            `word_embeddings` matrix and `convert_id_to_word` function.
        top_n (int): the number of top words to return at each index
    >>> from textattack.transformations import WordSwapGradientBased
    >>> from textattack.augmentation import Augmenter

    >>> transformation = WordSwapGradientBased()
    >>> augmenter = Augmenter(transformation=transformation)
    >>> s = 'I am fabulous.'
    >>> augmenter.augment(s)
    """

    def __init__(self, model_wrapper, top_n=1):
        # Unwrap model wrappers. Need raw model for gradient.
        if not isinstance(model_wrapper, textattack.models.wrappers.ModelWrapper):
            raise TypeError(f"Got invalid model wrapper type {type(model_wrapper)}")
        self.model = model_wrapper.model
        self.model_wrapper = model_wrapper
        self.tokenizer = self.model_wrapper.tokenizer
        # Make sure we know how to compute the gradient for this model.
        # validate_model_gradient_word_swap_compatibility(self.model)
        # Make sure this model has all of the required properties.
        if not hasattr(self.model, "get_input_embeddings"):
            raise ValueError(
                "Model needs word embedding matrix for gradient-based word swap"
            )
        if not hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id:
            raise ValueError(
                "Tokenizer needs to have `pad_token_id` for gradient-based word swap"
            )

        self.top_n = top_n
        self.is_black_box = False

        # self.counter_fitted_dic = self._extract_dictionary('counter')[0]
        # self.contextualized = self._extract_dictionary('contextualized')[0]

        self.counter_fitted_dic = utils.extract_dictionary(self.model_wrapper,'counter')[0]
        self.contextualized = utils.extract_dictionary(self.model_wrapper,'contextualized')[0]

        # print ('context shape',self.contextualized[0] )
        # print ('counter',self.counter_fitted_dic[0])
        # print ('model input emb',self.model.get_input_embeddings().weight.data.shape)
        # sys.exit()
        # self.filtered_vocab = self._filter_dictionary(dictionary=[self.counter_fitted_dic,self.contextualized.keys()])
        self.filtered_vocab = utils.filter_dictionary(dictionary=[self.counter_fitted_dic,self.contextualized.keys()])
        
        
        # self.contextualized = self._extract_word_embeddings('contextualized')
         
        # self.counter_fitted_dic = self._extract_word_embeddings('counter')
        # self._filter_dictionary
    def gen_indexes(self,list1, list2):
        list1_to_list2 = {}
        list2_to_list1 = {}

        list1_indexes = {word: [] for word in list1}
        list2_indexes = {word: [] for word in list2}

        for i, word in enumerate(list1):
            list1_indexes[word].append(i)

        for i, word in enumerate(list2):
            list2_indexes[word].append(i)

        for word in list1_indexes:
            if word in list2_indexes:
                for idx1 in list1_indexes[word]:
                    if list2_indexes[word]:
                        idx2 = list2_indexes[word].pop(0)
                        list1_to_list2[idx1] = idx2
                        list2_to_list1[idx2] = idx1            
        return list1_to_list2, list2_to_list1

    def _get_replacement_words_by_grad(self, attacked_text, indices_to_replace,**kwargs):
        """Returns returns a list containing all possible words to replace
        `word` with, based off of the model's gradient.

        Arguments:
            attacked_text (AttackedText): The full text input to perturb
            word_index (int): index of the word to replace
        """
        max_iter = 30
        grad_update_interval = 1
        batch_size = 1
        word_max_len = 128 
        counter = 0
        debug = False 

        unsuccesful_mask = kwargs['unsuccesful_mask']
        skip_this_token_mask = kwargs['skip_this_token_mask']
        skipped_sample_mask = kwargs['skipped_sample_mask']

        print ('attkt 4',attacked_text[4].text)
        print ('attkt 5',attacked_text[5].text)
        
        words_from_text_list = []
        for b in range(len(attacked_text)):
            words_from_text = utils.words_from_text(attacked_text[b].text)
            words_from_text_list.append(words_from_text)

        lookup_table = self.model.get_input_embeddings().weight.data.cpu()
        if debug: print ('lookup',lookup_table.shape)
        if debug: print ('tokenizer input',attacked_text.tokenizer_input )
        grad_output_list = []
        for b in range(len(attacked_text)):
            grad_output = self.model_wrapper.get_grad(attacked_text[b].tokenizer_input)
            grad_output_list.append(grad_output)
            

        emb_grad = torch.tensor( [grad["gradient"] for grad in grad_output_list ])
        
        text_ids = [grad["ids"][0].detach().cpu().tolist() for grad in grad_output_list ]
        
        tokenized_text_ids = [self.tokenizer.convert_ids_to_tokens(txt_ids) for txt_ids in text_ids]
        if debug: print ('text ids',text_ids,text_ids[13],self.tokenizer.convert_ids_to_tokens(text_ids),self.tokenizer.convert_ids_to_tokens(text_ids)[13])
        
        
        indexTotokens = {}
        tokensToindex = {}
        
        
        batch_wise_indexTotokens = []
        batch_wise_tokensToindex = []
        print ('words_from_text_list',words_from_text_list[4]) 
        print ('words_from_text_list',words_from_text_list[5]) 
        for b in range(len(attacked_text)):     
            indexTotokens, tokensToindex = self.gen_indexes(words_from_text_list[b], tokenized_text_ids[b])
            batch_wise_indexTotokens.append(indexTotokens)
            batch_wise_tokensToindex.append(tokensToindex)
 
 
        candidates = []
        
        unsuccesful_mask = unsuccesful_mask
        # print ('indeces to replace',indices_to_replace) 
        pre_transformation_mask = np.array([True if len(indx)>0 else False for indx in indices_to_replace ])
        # print ('pre_transformation_mask',pre_transformation_mask) 
        logical_and_operation = unsuccesful_mask & pre_transformation_mask
        logical_and_operation = logical_and_operation & skip_this_token_mask
        logical_and_operation = logical_and_operation & skipped_sample_mask
        

        # print  ('inner',batch_wise_indexTotokens,len(batch_wise_indexTotokens) )
        
        tokenizer_indices_to_replace = []
        for b in range(len(attacked_text)):
            
            if len(indices_to_replace[b]) > 0:# and logical_and_operation[b] :
                # print ('logical_and_operation[b]',logical_and_operation[b])
                # print ('minimums ',b,batch_wise_indexTotokens[b],min(indices_to_replace[b]))
                # print ('tokentoindex',batch_wise_tokensToindex[b])
                # print ('wft',words_from_text_list[b], tokenized_text_ids[b])
                tokenizer_indices_to_replace.append([batch_wise_indexTotokens[b][min(indices_to_replace[b])] ])
            else:
                tokenizer_indices_to_replace.append([0])
                 
        jacobian = kwargs['gradient']
        ids = kwargs['ids']
        synonyms = kwargs['synonyms']
        projected_mask = kwargs['projected_mask']
        adv_xs = text_ids
        
        
        adv_xs,synonym,synonyms,unsucessful_mask,indices_masked = self._project_synonyms(adv_xs,tokenizer_indices_to_replace,synonyms,projected_mask,logical_and_operation)
        # print ('tokenizer_indices_to_replace',tokenizer_indices_to_replace)
        synonym_indices = indices_masked[:,0].tolist()
        expanded_synonym_list  = []
        b = 0
        print ('synonym_indices',synonym_indices)
        while b < len(attacked_text):
            if b in synonym_indices: 
                first_element = synonym.pop(0)
                expanded_synonym_list.append(first_element)
            else:
                expanded_synonym_list.append(-1)
            b+=1
        print ('expanded_synonym_list',expanded_synonym_list[4])
        print ('expanded_synonym_list',expanded_synonym_list[5]) 
        # print ('synonym_indices',synonym_indices) 
        for b in range(len(attacked_text)):
            # print ('expanded_synonym_list',expanded_synonym_list[b])
            if expanded_synonym_list[b] == '[PAD]':
                print ('expanded_synonym_list pad',expanded_synonym_list[b])
                candidates.append((-1,-1)) 
                # sys.exit()
            elif logical_and_operation[b] == False:
                candidates.append((-1,-1)) 
            else:
                if b in synonym_indices:  
                    candidates.append((expanded_synonym_list[b],batch_wise_tokensToindex[b][tokenizer_indices_to_replace[b][0]]))
                else:
                    candidates.append((-1,-1))
            
        
        print ('candidates len',len(candidates))
        return candidates, unsuccesful_mask,synonyms


        token_to_replace = [ v for k,v in indexTotokens.items() if k in indices_to_replace ]

        if debug:print ('index to tokens',indexTotokens)
        if debug:print ('token to index',tokensToindex)
        if debug:print ('token to replace',token_to_replace)
        
        vocab_size = lookup_table.size(0)
        diffs = torch.zeros(len(indices_to_replace), vocab_size)
        if debug:print ('diffs shape',diffs.shape, len(emb_grad), emb_grad.shape)
        
        indices_to_replace = list(indices_to_replace)
        if debug:print ('list to replace',indices_to_replace)

        token_to_replace = list(token_to_replace)
        if debug:print ('token to replace',token_to_replace)

        if debug:print ('lookup table x emb',lookup_table.shape,emb_grad[1].shape )

        # for j, word_idx in enumerate(indices_to_replace):
        for j, word_idx in enumerate(token_to_replace):
            # Make sure the word is in bounds.
            if word_idx >= len(emb_grad):
                continue
            # Get the grad w.r.t the one-hot index of the word.
            b_grads = lookup_table.mv(emb_grad[word_idx]).squeeze() 
            # print ('word idx', word_idx)
            a_grad = b_grads[text_ids[word_idx]] 
            # for i in b_grads:
            #     if i == a_grad: 
            #         print('they are same', i,a_grad)
            #         print ('minus',i - a_grad) 
            #         print ('where are they same index',b_grads.tolist().index(a_grad.item()))
            diffs[j] = b_grads - a_grad
            # print ('minimums',min(diffs[j]),diffs[j].tolist().index(min(diffs[j]).item()) )
        if debug:print ('bgrads sahpe',b_grads.shape)
        if debug:print ('agrad shape',a_grad.shape)
        if debug:print ('diff shape',diffs.shape,self.tokenizer.pad_token_id)
        # print ('diff zero',diffs[13][0],diffs[13][1],diffs[13][2],diffs[13][-1])
        # Don't change to the pad token.
        diffs[:, self.tokenizer.pad_token_id] = float("-inf")

        # print ('diffs 13 argsort',(diffs[13]).argsort(), (diffs[13]).argsort()[0] )
        # Find best indices within 2-d tensor by flattening.
        word_idxs_sorted_by_grad = (-diffs).flatten().argsort()

        if debug:print ('words sorted',word_idxs_sorted_by_grad,len(word_idxs_sorted_by_grad))

        candidates = []
        num_words_in_text, num_words_in_vocab = diffs.shape
        for idx in word_idxs_sorted_by_grad.tolist():
            idx_in_diffs = idx // num_words_in_vocab
            # print ('idx in diffd',idx_in_diffs)
            idx_in_vocab = idx % (num_words_in_vocab)
            # print ('indx in vocab',idx_in_vocab)
            # idx_in_sentence = indices_to_replace[idx_in_diffs]
            idx_in_sentence = token_to_replace[idx_in_diffs]
            # print ('idx_in_sent',idx_in_sentence)
            word = self.tokenizer.convert_ids_to_tokens(idx_in_vocab)
            if (not utils.has_letter(word)) or (len(utils.words_from_text(word)) != 1):
                # Do not consider words that are solely letters or punctuation.
                continue
            if '##' in word:
                continue
            if '[CLS]' in word:
                continue
            if '[SEP]' in word:
                continue
            if word not in self.filtered_vocab:
                continue
                
            candidates.append((word, tokensToindex[idx_in_sentence]))
            if len(candidates) == self.top_n:
                break
        if debug:print ('candidates',candidates)   
        return candidates

    def _project_synonyms(self,token_ids,pos,synonyms,projection_masked,unsuccessful_mask):
        max_iter = 30
        grad_update_interval = 1
        batch_size =  len(token_ids) 
        word_max_len = 128
        counter = 0

        # synonyms = torch.tensor(synonyms).long()
        # # jacobian = torch.tensor(jacobian)
        # token_ids = torch.tensor(token_ids).long()
        
        # # we need to do this for 30 iterations
        # # modified_num = torch.sum(modified_mask, dim=-1)
        # # modified_ratio = (modified_num + 1) / len(token_ids)
        # # print ('mods',modified_num,modified_ratio,len(token_ids))
        # # sys.exit()

        # # idealy precompute the token ids, at this time it's dynamic? maybe this is good
        # # token_embeddings = self.embedding_matrix[token_ids] # shape [128, 768]
        # # synonyms_embeddings = self.embedding_matrix[synonyms] # shape [128, 4, 768]

        # token_embeddings = self.victim_model.model.get_input_embeddings()( token_ids.to(device)) 
        # synonyms_embeddings = self.victim_model.model.get_input_embeddings()( synonyms.to(device)) 
        

        # # compute projection
        # token_embeddings_expanded = token_embeddings.unsqueeze(1) # shape [128, 1, 768]
        # jacobian_squeezed = jacobian.squeeze() # shape [128, 768]
        # jacobian_expanded = jacobian_squeezed.unsqueeze(1) # shape [128, 1, 768]

        # multi_jacob = (synonyms_embeddings - token_embeddings_expanded) * jacobian_expanded
        # projection = torch.sum(multi_jacob, dim=-1) # final shape [128, 4]

        # # create subword/not real word mask
        # print ('filtered vocab',self.filtered_vocab_ids)
        # print ('token_ids',token_ids)

        # # just check at the end if origin exists in fitlered ids, if it dosn't return the new maks with this id
        # # blocked
        # for p,tok in enumerate(token_ids): 
        #     if tok.item() not in self.filtered_vocab_ids:
                
        #         synonyms[p] = torch.zeros(synonyms.shape[1])
        # # an alternative way is to rank all substitutions, instead of keeping the first 
                
        # # the alternative is to go though it all 
        # # Step 3: Mask Projection. Substitution can only occur on known words.
        # synonym_mask = (synonyms == 0).to(device) # mask where token id is 0 i.e., no synonyms found
        # # print ('synonym_mask',synonym_mask)
        # inf_tensor = torch.full(projection.shape, -1e9).to(device) # tensor filled with large negative values
        # # print ('inf t',inf_tensor)
        # projection_masked = projection.clone() # create a clone of the projection tensor housing the mask
        # # print ('projection_masked',projection_masked)
        # # apply mask to projection tensor
        
        # projection_masked[synonym_mask] = inf_tensor[synonym_mask]
        
        # # Step 4: Substitution
        # print ('projection_masked',projection_masked,projection_masked.shape)
        # reduced = torch.max(projection_masked, dim=-1)[0] 
        
        # print ('reduced',reduced,reduced.shape) 
        # # _, pos = torch.topk(reduced, k=grad_update_interval, dim=-1) # pos will be of shape [128, grad_update_interval]
        # values, pos = torch.topk(reduced, k=4, dim=-1)
        # if one_step:
        #     return pos
        
        # print ('pos0',pos)  
        original_pos = pos
        pos = torch.tensor(pos).to(device) 
        # print ('pos',pos,pos.shape)  
         
        serial = torch.arange(start=0, end=batch_size).unsqueeze(-1).repeat_interleave(1, grad_update_interval).to(device)
        # print ('serial',serial,serial.shape)   
        indices = torch.stack([serial, pos], dim=-1)
        # print ('indices',indices,indices.shape)
         
        # print ('unsuccessful_mask',unsuccessful_mask)
        # print ('indices',indices)
        indices_masked = indices[unsuccessful_mask] # shape depends on unsuccessful_mask
        # print ('indices_masked',indices_masked,indices_masked.shape)
        
        indices_masked = indices_masked.squeeze(1) 
        # print ('indices_masked',indices_masked, indices_masked.shape) 
       
        token_ids = torch.tensor(token_ids)
      
        origin = token_ids[indices_masked[:,0],indices_masked[:,1]]
        # print ('origin',origin)
        
        # print ('projection_masked',projection_masked, projection_masked.shape)  
        argmax_projection = torch.argmax(projection_masked, dim=-1, keepdim=True)#[unsuccessful_mask] # shape depends on unsuccessful_mask
        # print ('argmax proj',argmax_projection,argmax_projection.shape)
        argmax_projection = argmax_projection.squeeze(-1) 
        # print ('argmax proj',argmax_projection,argmax_projection.shape)
        # sys.exit()
        # print ('indices_masked',indices_masked,indices_masked[:,0], indices_masked[:,1]) 
        
        gather_nd_projection = argmax_projection[indices_masked[:,0], indices_masked[:,1]] 
        # print ('gather_nd_projection',gather_nd_projection) 
        
        synonym_pos = gather_nd_projection.unsqueeze(-1) 


        
        # print ('synonym_pos',synonym_pos, synonym_pos.shape)   
        # print ('indices masked',indices_masked) 
        synonym_indices_masked = torch.cat([indices_masked, synonym_pos], dim=-1).detach().cpu() 
        # print ('synonym_indices_masked',synonym_indices_masked)
        
        # print ('synonyms',synonyms,synonyms.shape,synonyms[0,12])
        synonym = synonyms[synonym_indices_masked[:,0], synonym_indices_masked[:,1],synonym_indices_masked[:,2]]
        # print ('synonym',synonym,synonym.shape)
        
        if isinstance(synonym,np.int64):
            synonym = [synonym]



        
        synonym = torch.tensor(synonym) 
       
        delta = synonym - origin
        # print ('delta',delta,token_ids.shape, origin)
        # print ('synonym', self.tokenizer.convert_ids_to_tokens(synonym.tolist()))
        # print ('origin', self.tokenizer.convert_ids_to_tokens(origin.tolist()))

        # print ('token_ids',token_ids) 
        token_ids[indices_masked[:,0],indices_masked[:,1]] = origin + delta
        
        # print ('token ids',token_ids)#,self.tokenizer.convert_ids_to_tokens(token_ids.tolist()))
        
        indices_masked_synonyms = indices_masked.detach().cpu()
        # print ('synonym_indices_masked',synonym_indices_masked, synonym_indices_masked.shape)
        # print ('synonyms',synonyms[indices_masked_synonyms[:,0],indices_masked_synonyms[:,1]])

        synonym_subtract_mask = torch.zeros(synonyms.shape).long()
        # print ('synonym_subtract_mask',synonym_subtract_mask,type(synonym_subtract_mask))
        # sys.exit()
        # print ('synonym_subtract_mask',synonym_subtract_mask,synonym_subtract_mask.shape )
        # print ('synonym',synonym.unsqueeze(dim=-1))

        index_x = synonym_indices_masked[:, 0]
        index_y = synonym_indices_masked[:, 1]
        index_z = synonym_indices_masked[:, 2]
        # synonym_select = torch.index_select(synonym, 0, index_z)
        synonym_select = -synonym
        # print ('synonym_select',synonym_select,synonym_select.shape)
        synonym_select = synonym_select#.float()
        # print ('index_z',index_z)
        synonym_subtract_mask.index_put_((index_x, index_y, index_z), synonym_select) 
        # print ('synonym_select',synonym_select.shape)
        # print ('synonym_subtract_mask',synonym_subtract_mask,synonym_subtract_mask.shape)
        # print ('-1 position',synonym_subtract_mask[-1])
        synonym_subtract_mask = synonym_subtract_mask.numpy()
 
        # synonym_subtract_mask[:,:,synonym_indices_masked[0][2]] = -synonym.unsqueeze(dim=-1)
        # print ('synonym_subtract_mask2',synonym_subtract_mask)
        # synonyms[indices_masked_synonyms[:,0],indices_masked_synonyms[:,1]] = [0]*synonyms.shape[2]
        # print ('synonyms2',synonyms[indices_masked_synonyms[:,0],indices_masked_synonyms[:,1]])
        # sys.exit()
        # print ('synonyms - 1', synonyms[-1])
        synonyms = np.add(synonyms,synonym_subtract_mask)
        # print ('synonyms - 1 after', synonyms[-1]) 
        # sys.exit()
        return token_ids ,self.tokenizer.convert_ids_to_tokens(synonym.tolist()),synonyms, unsuccessful_mask, indices_masked_synonyms.detach().cpu()
        return  token_ids, synonyms
     


    def _get_transformations(self, attacked_text, indices_to_replace, **kwargs):
        """Returns a list of all possible transformations for `text`.

        If indices_to_replace is set, only replaces words at those
        indices.
        """  

        # print ('attacked_text',attacked_text)
        # print ('indices_to_replace',indices_to_replace) 

        temp_transformations,unsuccesful_mask,synonyms = self._get_replacement_words_by_grad( attacked_text, indices_to_replace,**kwargs  )
        # print ('temp_transformations',temp_transformations)
        
        transformations = [] 
        for b, (word, idx) in enumerate(temp_transformations):
            
            if word == -1:
                transformations.append([])
                # continue
            else:
                transformations.append([attacked_text[b].replace_word_at_index(idx, word)])
            # transformations.append(mid_transformations)
    
        
        
        return transformations,unsuccesful_mask,synonyms
    
    def _get_replacement_words_by_grad_hotflip(self, attacked_text, indices_to_replace,**kwargs):
        """Returns returns a list containing all possible words to replace
        `word` with, based off of the model's gradient.

        Arguments:
            attacked_text (AttackedText): The full text input to perturb
            word_index (int): index of the word to replace
        """
        debug = False 

         

        if debug: print ('attacked_text',attacked_text.attack_attrs)
        if debug: print ('word from text',utils.words_from_text(attacked_text.text))
        words_from_text = utils.words_from_text(attacked_text.text)
        lookup_table = self.model.get_input_embeddings().weight.data.cpu()
        if debug: print ('lookup',lookup_table.shape)
        if debug: print ('tokenizer input',attacked_text.tokenizer_input )
        grad_output = self.model_wrapper.get_grad(attacked_text.tokenizer_input)
        
        # emb_grad = torch.tensor(kwargs['gradient']) #grad_output["gradient"])
        # text_ids = kwargs['ids'] #grad_output["ids"][0]
        emb_grad = torch.tensor( grad_output["gradient"])
        text_ids =  grad_output["ids"][0]
        tokenized_text_ids = self.tokenizer.convert_ids_to_tokens(text_ids)
        if debug: print ('text ids',text_ids,text_ids[13],self.tokenizer.convert_ids_to_tokens(text_ids),self.tokenizer.convert_ids_to_tokens(text_ids)[13])
        # grad differences between all flips and original word (eq. 1 from paper)
        
        # index translator
        # dictionary {index from words:index to tokens}
        # dictionary {index of toeksn: index from words}
        indexTotokens = {}
        tokensToindex = {}
        # for tok_id, tok in enumerate(self.tokenizer.convert_ids_to_tokens(text_ids)):
        #     if tok in words_from_text:
        #         idx = words_from_text.index(tok)
                
        #         tok_idx = tok_id
        #         indexTotokens[idx] = tok_idx
        #         tokensToindex[tok_idx] = idx

        # Create dictionaries to store mappings
        indexTotokens, tokensToindex = self.gen_indexes(words_from_text, tokenized_text_ids)

        token_to_replace = [ v for k,v in indexTotokens.items() if k in indices_to_replace ]

        if debug:print ('index to tokens',indexTotokens)
        if debug:print ('token to index',tokensToindex)
        if debug:print ('token to replace',token_to_replace)
        
        vocab_size = lookup_table.size(0)
        diffs = torch.zeros(len(indices_to_replace), vocab_size)
        if debug:print ('diffs shape',diffs.shape, len(emb_grad), emb_grad.shape)
        
        indices_to_replace = list(indices_to_replace)
        if debug:print ('list to replace',indices_to_replace)

        token_to_replace = list(token_to_replace)
        if debug:print ('token to replace',token_to_replace)

        if debug:print ('lookup table x emb',lookup_table.shape,emb_grad[1].shape )

        # for j, word_idx in enumerate(indices_to_replace):
        for j, word_idx in enumerate(token_to_replace):
            # Make sure the word is in bounds.
            if word_idx >= len(emb_grad):
                continue
            # Get the grad w.r.t the one-hot index of the word.
            b_grads = lookup_table.mv(emb_grad[word_idx]).squeeze() 
            # print ('word idx', word_idx)
            a_grad = b_grads[text_ids[word_idx]] 
            # for i in b_grads:
            #     if i == a_grad: 
            #         print('they are same', i,a_grad)
            #         print ('minus',i - a_grad) 
            #         print ('where are they same index',b_grads.tolist().index(a_grad.item()))
            diffs[j] = b_grads - a_grad
            # print ('minimums',min(diffs[j]),diffs[j].tolist().index(min(diffs[j]).item()) )
        if debug:print ('bgrads sahpe',b_grads.shape)
        if debug:print ('agrad shape',a_grad.shape)
        if debug:print ('diff shape',diffs.shape,self.tokenizer.pad_token_id)
        # print ('diff zero',diffs[13][0],diffs[13][1],diffs[13][2],diffs[13][-1])
        # Don't change to the pad token.
        diffs[:, self.tokenizer.pad_token_id] = float("-inf")

        # print ('diffs 13 argsort',(diffs[13]).argsort(), (diffs[13]).argsort()[0] )
        # Find best indices within 2-d tensor by flattening.
        word_idxs_sorted_by_grad = (-diffs).flatten().argsort()

        if debug:print ('words sorted',word_idxs_sorted_by_grad,len(word_idxs_sorted_by_grad))

        candidates = []
        num_words_in_text, num_words_in_vocab = diffs.shape
        for idx in word_idxs_sorted_by_grad.tolist():
            idx_in_diffs = idx // num_words_in_vocab
            # print ('idx in diffd',idx_in_diffs)
            idx_in_vocab = idx % (num_words_in_vocab)
            # print ('indx in vocab',idx_in_vocab)
            # idx_in_sentence = indices_to_replace[idx_in_diffs]
            idx_in_sentence = token_to_replace[idx_in_diffs]
            # print ('idx_in_sent',idx_in_sentence)
            word = self.tokenizer.convert_ids_to_tokens(idx_in_vocab)
            if (not utils.has_letter(word)) or (len(utils.words_from_text(word)) != 1):
                # Do not consider words that are solely letters or punctuation.
                continue
            if '##' in word:
                continue
            if '[CLS]' in word:
                continue
            if '[SEP]' in word:
                continue
            if word not in self.filtered_vocab:
                continue
                
            candidates.append((word, tokensToindex[idx_in_sentence]))
            if len(candidates) == self.top_n:
                break
        if debug:print ('candidates',candidates)   
        return candidates

    
    def _extract_dictionary(self,name_embedding = 'glove'):
        import json
        import nltk
        from nltk.corpus import wordnet
        if name_embedding == 'glove':
            nbr_file = '/home/brianformento/security_project/defense/attention_align/tmd/TextDefender/glove.6B.300d.txt'
    
            glove_embeddings = {}

            glove_embedding_file = open(nbr_file, 'r', encoding='utf-8')
            glove_embedding_lines = glove_embedding_file.readlines()
            # with open(nbr_file, 'r', encoding='utf-8') as f:
            for l,line in enumerate(glove_embedding_lines):
                # print ('r l',line)
                values = line.strip().split()
                word = values[0] 
                embedding = np.asarray(values[1:], dtype='float32') 
                glove_embeddings[word] = embedding  
            glove_embedding_file.close()
            return list(glove_embeddings.keys()) , glove_embeddings
        elif name_embedding == 'contextualized':
            # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_vocab = self.tokenizer.vocab
            # bert_vocab = tokenizer.vocab
            return bert_vocab, ''
        elif name_embedding == 'wordnet':
            # import nltk
            nltk.download('wordnet')
            # from nltk.corpus import wordnet

            wordnet_vocabulary = list(wordnet.words())
            return wordnet_vocabulary , ''
        elif name_embedding == 'counter':
            nbr_file = '/home/brianformento/security_project/defense/attention_align/tmd/TextDefender/counterfitted_neighbors.json'
            loaded = json.load(open(nbr_file))
            # filtered = defaultdict(lambda: [], {})

            return loaded.keys(), loaded
    
    def _extract_word_embeddings(self,name_embedding = 'glove',dictionary=None):
        if dictionary == None:
            ValueError ('Need at least 1 dictionary to extract word embeddings')
        
        if name_embedding == 'glove':
            _, glove_embeddings =  self.extract_dictionary('glove')
            embedding_matrix = []
            word_list = []

            for word, embedding in glove_embeddings.items():
                if word in dictionary:
                    embedding_matrix.append(embedding)
                    word_list.append(word)
                else:
                    pass

            embedding_matrix = torch.tensor(embedding_matrix)

            # Calculate cosine similarity at the tensor level
            norms = torch.norm(embedding_matrix, dim=1)
            normalized_embeddings = embedding_matrix / norms[:, None]
            similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

            # Set diagonal elements to -1 to exclude the word itself from the closest words
            # torch.fill_diagonal_(similarity_matrix, -1)

            glove_closest_words = {}
            for row_idx, word in enumerate(word_list):
                # Get the indices of the 50 closest words
                closest_indices = torch.argsort(similarity_matrix[row_idx], descending=True)[:50]
                closest_words = [(word_list[idx], similarity_matrix[row_idx, idx]) for idx in closest_indices]
                glove_closest_words[word] = closest_words 
            print ('Glove Complete')
            return glove_closest_words

        elif name_embedding == 'contextualized':
            from itertools import zip_longest
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            tokenizer = self.tokenizer #BertTokenizer.from_pretrained('bert-base-uncased')
            model = self.model_wrapper #BertModel.from_pretrained('bert-base-uncased')
            bert_word_embeddings = {}
            batch_size = 64  # Number of words to process in each batch
            bert_vocab = dictionary
            for batch_words in zip_longest(*[iter(bert_vocab)] * batch_size, fillvalue=None):
                
                
                batch_words = [word for word in batch_words if word is not None]
                if len(batch_words) == 0:
                    continue
                tokenized_text = tokenizer.batch_encode_plus(batch_words, padding=True, truncation=True, return_tensors='pt')
                tokens_tensor = tokenized_text['input_ids']
                attention_mask = tokenized_text['attention_mask']
                
                # Move the tensors to GPU if available
                if torch.cuda.is_available():
                    tokens_tensor = tokens_tensor.to(device)
                    attention_mask = attention_mask.to(device)
                    model = model.to(device)
                
                with torch.no_grad():
                    hidden_states = model(tokens_tensor, attention_mask=attention_mask)[0].cpu()
                    
                for i, word in enumerate(batch_words):
                    # print('word',word)
                    if word is not None:
                        word_embedding = torch.mean(hidden_states[i], dim=0).numpy()
                        bert_word_embeddings[word] = word_embedding

                
            embedding_matrix_bert = []
            word_list_bert = []

            for word, embedding in bert_word_embeddings.items():
                embedding_matrix_bert.append(embedding)
                word_list_bert.append(word)

            embedding_matrix_bert = torch.tensor(embedding_matrix_bert)

            # Calculate cosine similarity at the tensor level
            norms = torch.norm(embedding_matrix_bert, dim=1)
            normalized_embeddings = embedding_matrix_bert / norms[:, None]
            similarity_matrix_bert = torch.mm(normalized_embeddings, normalized_embeddings.t())

            # Set diagonal elements to -1 to exclude the word itself from the closest words
            # torch.fill_diagonal_(similarity_matrix_bert, -1)

            bert_closest_words = {}
            for row_idx, word in enumerate(word_list_bert):
                # Get the indices of the 50 closest words
                closest_indices = torch.argsort(similarity_matrix_bert[row_idx], descending=True)[:50]
                closest_words = [(word_list_bert[idx], similarity_matrix_bert[row_idx, idx]) for idx in closest_indices]
                bert_closest_words[word] = closest_words 
            print ('BERT Embeddings Complete') 
            return bert_closest_words
        elif name_embedding == 'wordnet':
            wordnet_vocabulary = dictionary
            wordnet_closest_words = {}
            for word in wordnet_vocabulary:
                # Find synonyms for the word
                synonyms = []
                synsets = wordnet.synsets(word)
                for synset in synsets:
                    synonyms.extend(synset.lemmas())

                # Extract the names of the synonyms and their frequencies
                synonym_names = [synonym.name() for synonym in synonyms]

                # Take the top 50 most common synonyms
                closest_words = [(synonym, 1) for synonym in synonym_names[:50]]

                wordnet_closest_words[word] = closest_words
            return wordnet_closest_words
        elif name_embedding == 'counter':
            _, word_embeddings = self.extract_dictionary('counter')
            counter_vocabulary = dictionary
            counter_closest_words = {}
            for word in counter_vocabulary:
                if word in word_embeddings:
                    # Find synonyms for the word
                    synonyms = word_embeddings[word]



                    # Take the top 50 most common synonyms
                    closest_words = [(synonym, 1) for synonym in synonyms[:50]]

                    counter_closest_words[word] = closest_words
            return counter_closest_words

    def compute_overlap(set1 = None,set2= None):
        if set1 == None or set2 == None:
            ValueError ('Needs at least two sets to find the jaccard similarity')

            #list of words

            # function that gets two dictionarys containing a word:[list of synonyms]
            # returns the average jaccard similarity between the two sets
            # need to remove the same word, form the ranking, this should be easey just pop word from list of synonyms
        list_of_words = list(set1.keys())

        per_word_overlap_percentage = []
        
        for i,word in enumerate(list_of_words): 
            set1_at_word = set1[word]
            set2_at_word = set2[word] 
                

            set1_at_word_set = set([tup[0] for tup in set1_at_word])
            set2_at_word_set = set([tup[0] for tup in set2_at_word]) 
            intersection = set1_at_word_set & set2_at_word_set
            union = set1_at_word_set | set2_at_word_set
            num_intersection = len(intersection)/len(union) 
            per_word_overlap_percentage.append(num_intersection)

            # if i < 10:
            #     print (i,word,intersection)
            #     print ('word set 1',set1_at_word)
            #     print ('word set 2',set2_at_word)
            # if num_intersection > 0.70:
            #     print (i,word,intersection)
            #     print ('word set 1',set1_at_word)
            #     print ('word set 2',set2_at_word)
        overlap = sum(per_word_overlap_percentage) / len(per_word_overlap_percentage)
        return {'min':min(per_word_overlap_percentage),'max':max(per_word_overlap_percentage),'overlap':overlap }

    def _filter_dictionary(self,dictionary = None):
        if dictionary == None:
            ValueError ('Need at least 2 dictionaries to find the union set')
        else:
            intersection = set()
            for i,dic in enumerate(dictionary):
                set_dic = set(dic)
                if len(intersection) == 0:
                    intersection = set_dic
                else:
                    intersection =  intersection & set_dic
        
        return intersection
    def extra_repr_keys(self):
        return ["top_n"]