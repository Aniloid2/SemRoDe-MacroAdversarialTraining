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
# from textattack.shared.utils import device, words_from_text

class WordSwapHotFlipMicro(WordSwap):
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

    def _get_transformations(self, attacked_text, indices_to_replace, **kwargs):
        """Returns a list of all possible transformations for `text`.

        If indices_to_replace is set, only replaces words at those
        indices.
        """
        transformations = []
        for word, idx in self._get_replacement_words_by_grad(
            attacked_text, indices_to_replace,**kwargs
        ):
            transformations.append(attacked_text.replace_word_at_index(idx, word))
        return transformations
    
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