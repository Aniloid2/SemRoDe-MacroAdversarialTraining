"""
Beam Search
===============

"""
import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus,GoalFunctionResult
from textattack.search_methods import SearchMethod
import sys
from torch.nn import CrossEntropyLoss
import time
import torch
from textattack.shared.utils import device
from textattack.shared import utils
import os
class MicroSearchBW(SearchMethod):
    """An attack that maintains a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """

    def __init__(self, beam_width=8):
        self.beam_width = beam_width
        self.loss_fn = CrossEntropyLoss()
        self.victim_model = None# self.get_victim_model()
        # self.counter_fitted_dic = utils.extract_dictionary(self.victim_model,'counter')[0]
        # self.contextualized = utils.extract_dictionary(self.victim_model,'contextualized')[0]
        # self.filtered_vocab = utils.filter_dictionary(dictionary=[self.counter_fitted_dic,self.contextualized.keys()])
        
    def _initialize_model_and_vocab(self):
        if self.victim_model == None:
            self.victim_model = self.get_victim_model()
            counter_fitted_dic_keys, self.counter_fitted_dic = utils.extract_dictionary(self.victim_model,'counter_vector')
            self.contextualized = utils.extract_dictionary(self.victim_model,'contextualized')[0]
            self.filtered_vocab = utils.filter_dictionary(dictionary=[counter_fitted_dic_keys,self.contextualized.keys()])
            self.index_to_token = {}
            self.token_to_index = {}

            for word,index in self.contextualized.items():
                 
                self.index_to_token[index]= word
                self.token_to_index[word] = index  

            self.filtered_vocab_ids = set()
            for word in self.filtered_vocab:
                
                self.filtered_vocab_ids.add(self.token_to_index[word])  
            
            self.vocab_size = len(self.victim_model.tokenizer)
            feature_size = 768  # Replace with the actual feature size
            feature_size_counter = 300

            config = self.victim_model.model.config 
            if not os.path.exists(os.path.join(  'aux_files',config.finetuning_task)):
                os.makedirs(os.path.join(  'aux_files', config.finetuning_task))



            if not os.path.isfile(os.path.join( 'aux_files',config.finetuning_task, 'small_dist_counter_%s_%s.npy' % (config.finetuning_task, config.model_type))):
                def compute_dist_matrix(vocab_size, feature_size,feature_size_counter):
                    # Initialize a tensor of shape (vocab size, feature size) with zeros
                    # embedding_tensor = torch.zeros(( feature_size,vocab_size), device=device)
                    embedding_tensor = np.zeros(shape = (( feature_size,vocab_size)))
                    embedding_tensor_counter = np.zeros(shape = (( feature_size_counter,vocab_size)))
                    print ('embedding_tensor shape',embedding_tensor.shape)
                    # Fill the tensor with the features
                    cnt = 0
                    unfound_ids = []
                    unfound_words = []   
                    for word,id in self.contextualized.items(): 
                        if word in self.filtered_vocab:
                            if word =='good' or word == 'fine' or word == 'great' or word == 'bad':
                                print (word,id)
                            feature_vector = self.victim_model.model.get_input_embeddings()(torch.tensor(id, device=device)) 
                            embedding_tensor[:,id] = feature_vector.detach().cpu().numpy()
                            embedding_tensor_counter[:,id] = self.counter_fitted_dic[word]

                            # can also do the same as above but with glove feature vectors?
                        else:
                            cnt+=1
                            unfound_ids.append(id)
                            unfound_words.append(word)
        
                    # unfound_ids = np.array(unfound_ids)   
                    print("Number of not found words = ", cnt)

                    def compute_distance(embedding_tensor):
                        

                        INFINITY = 100000
                        embedding_tensor = embedding_tensor.astype(np.float32)
                        c_ = -2 * np.dot(embedding_tensor.T, embedding_tensor)
                        a = np.sum(np.square(embedding_tensor), axis=0).reshape((1, -1))
                        
                        b = a.T
                        dist = a + b + c_
                        dist[0, :] = INFINITY
                        dist[:, 0] = INFINITY
                        dist[unfound_ids, :] = INFINITY
                        dist[:, unfound_ids] = INFINITY
                        return dist

                    # INFINITY = 100000
                    # embedding_tensor_counter = embedding_tensor_counter.astype(np.float32)
                    # c_ = -2 * np.dot(embedding_tensor_counter.T, embedding_tensor_counter)
                    # a = np.sum(np.square(embedding_tensor_counter), axis=0).reshape((1, -1))
                    
                    # b = a.T
                    # dist = a + b + c_
                    # dist[0, :] = INFINITY
                    # dist[:, 0] = INFINITY
                    # dist[unfound_ids, :] = INFINITY
                    # dist[:, unfound_ids] = INFINITY
                    dist = compute_distance(embedding_tensor)
                    dist_counter = compute_distance(embedding_tensor_counter)
                    return dist, dist_counter
                

                dist,dist_counter = compute_dist_matrix(self.vocab_size,feature_size,feature_size_counter)
                print ('dist','good->great',dist[2204,2307],'great->fine',dist[2307,2986],'good->fine',dist[2204,2986])
                print ('dist_counter','good->great',dist_counter[2204,2307],'great->fine',dist_counter[2307,2986],'good->fine',dist_counter[2204,2986])
                print ('dist','bad->great',dist[2919,2307],'bad->fine',dist[2919,2986],'bad->good',dist[2919,2204])
                print ('dist_counter','bad->great',dist_counter[2919,2307],'bad->fine',dist_counter[2919,2986],'bad->good',dist_counter[2919,2204])
                self.dist = dist
                self.dist_counter = dist_counter

                def create_small_embedding_matrix(dist_mat,vocab_size,threshold=1.5, retain_num=50):
                    small_embedding_matrix = np.zeros(shape=((vocab_size , retain_num, 2)))
                    for i in range(vocab_size ):
                        if i % 1000 == 0:
                            print("%d/%d processed." % (i, vocab_size))
                        dist_order = np.argsort(dist_mat[i, :])[1 : 1 + retain_num]
                        
                        dist_list = dist_mat[i][dist_order]
                        mask = np.ones_like(dist_list)
                        if threshold is not None:
                            mask = np.where(dist_list < threshold)
                            dist_order, dist_list = dist_order[mask], dist_list[mask]
                        n_return = len(dist_order)
                        dist_order_arr = np.pad(
                            dist_order, (0, retain_num - n_return), "constant", constant_values=(-1, -1)
                        )
                        dist_list_arr = np.pad(
                            dist_list, (0, retain_num - n_return), "constant", constant_values=(-1, -1)
                        )
                        small_embedding_matrix[i, :, 0] = dist_order_arr
                        small_embedding_matrix[i, :, 1] = dist_list_arr
                    return small_embedding_matrix
                
                
                self.smaller_dist = create_small_embedding_matrix(self.dist,self.vocab_size)
                self.smaller_dist_counter = create_small_embedding_matrix(self.dist_counter,self.vocab_size)
                np.save(os.path.join( 'aux_files',config.finetuning_task, 'small_dist_%s_%s.npy' % (config.finetuning_task, config.model_type)), self.smaller_dist)
                np.save(os.path.join( 'aux_files',config.finetuning_task, 'small_dist_counter_%s_%s.npy' % (config.finetuning_task, config.model_type)), self.smaller_dist_counter)
            
            else:
                
                self.smaller_dist = np.load(os.path.join( 'aux_files',config.finetuning_task, 'small_dist_%s_%s.npy' %  (config.finetuning_task, config.model_type)))
                self.smaller_dist_counter = np.load(os.path.join( 'aux_files',config.finetuning_task, 'small_dist_counter_%s_%s.npy' %  (config.finetuning_task, config.model_type)))
            
            
            
        else:
            pass

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
 

    def _get_index_order(self,initial_result, inputs_dict,synonyms,predictions,embeddings,number_of_synonyms):
        
        # initial_text = initial_result.attacked_text
        # len_text, indices_to_order = self.get_indices_to_order(initial_text) 
        # self._initialize_model_and_vocab()
        victim_model = self.victim_model# self.get_victim_model()
        # index_scores = np.zeros(len_text)

        #  # Default max length is set to be int(1e30), so we force 512 to enable batching.
        # max_length = (
        #     512
        #     if self.victim_model.tokenizer.model_max_length == int(1e30)
        #     else self.victim_model.tokenizer.model_max_length
        # )

        # print ('token ier input ',initial_text.tokenizer_input)
        # inputs_dict = self.victim_model.tokenizer(
        #     initial_text.tokenizer_input,
        #     add_special_tokens=True,
        #     padding="max_length",
        #     max_length=max_length,
        #     truncation=True,
        #     return_tensors="pt",
        # )
         
 
        input_tokens = inputs_dict['input_ids']
        
        input_tokens.tolist()[0]
        # print ('input token sample 0',self.victim_model.tokenizer.convert_ids_to_tokens(input_tokens.tolist()[0]) ) 
        synonyms = self.get_output_tensor(input_tokens.tolist(),number_of_synonyms)
        # print ('synonyms',synonyms,torch.tensor(synonyms).shape,synonyms[0][12],self.victim_model.tokenizer.convert_ids_to_tokens(synonyms[0][12]))
        # sys.exit()
        counter = 0
        max_iter= 1
        grad_update_interval = 1
        # batch_size = 1
        batch_size =len(initial_result) 
        word_max_len = 128
        modified_mask  =  torch.zeros((batch_size, word_max_len))
        adv_xs = input_tokens.tolist() 
        # predictions , embeddings, input_tokens= victim_model.embedding_inference(inputs_dict)
        # print ('initial result',initial_result[0].ground_truth_output)  
        predictions_argmax = torch.argmax(predictions,dim=-1)
        

        # gradient = victim_model.get_grad(initial_text.tokenizer_input)["gradient"]
        ground_truth_output_list = []
        for init_result in initial_result:


            if not isinstance(init_result.ground_truth_output, torch.LongTensor):
                ground_truth_output_list.append(init_result.ground_truth_output)

        ground_truth_output = torch.LongTensor(ground_truth_output_list).to(device)
        # print ('ground_truth_output',ground_truth_output)
        
        # equals = torch.eq(predictions_argmax, ground_truth_output) 
        # if equals == False:
        #     # achived a missclassification
        #     break

        loss = self.loss_fn(predictions, ground_truth_output) 
        loss.backward()
        jacobian = embeddings.grad 
        # print ('jacobian shape',jacobian.shape) 
        

        # token_embedding = {id: victim_model.model.get_input_embeddings()(torch.tensor(id).to(device))  for token, id in victim_model.tokenizer.get_vocab().items()}
        # print ('token embedding size0', token_embedding[0][:4])
        # print ('token embedding size1', token_embedding[1][:4])
        # sys.exit()
        
        # find synonyms
        # print ('input tokens',input_tokens)
        token_ids = self.victim_model.tokenizer.convert_ids_to_tokens(input_tokens.tolist()[0])
        # print ('token_ids tokens',token_ids) 
        
        # print ('shape dist mat',self.smaller_dist.shape)
    

        
        synonym_order,projected_mask = self.get_order(adv_xs,synonyms,jacobian,modified_mask,number_of_synonyms)
        return synonym_order,False,jacobian,input_tokens,projected_mask

     
    
    def perform_search(self, initial_result_gfs):
        number_of_synonyms = 5
        # print ('initial_result',initial_result_gfs)
        initial_result= initial_result_gfs['goal_function_result']
        skipped_sample_mask = initial_result_gfs['skipped_sample_mask'] 
        # initial result will include a skipped mask

        batch_size = len(initial_result) 


        attacked_text =  [i.attacked_text for i in initial_result]

        initial_text = [i.attacked_text for i in initial_result] 
        len_text, indices_to_order = self.get_indices_to_order_batchwise(initial_text) 
        # print ('initial text',initial_text)  
        self._initialize_model_and_vocab() 
        victim_model = self.victim_model# self.get_victim_model()
        # print ('len_text',len_text)
        # print ('max(len_text)',max(len_text))
        index_scores = np.zeros(shape=(batch_size,max(len_text)))


         # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.victim_model.tokenizer.model_max_length == int(1e30)
            else self.victim_model.tokenizer.model_max_length
        )



        tokenizer_inputs = [init_text.tokenizer_input for init_text in initial_text]
        # print ('tokenizer_inputs',tokenizer_inputs)
        inputs_dict = self.victim_model.tokenizer(
            tokenizer_inputs,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )


        
        # print ('input dict',inputs_dict)
        

        ### tokenize words from text directly
        ### tokenize words from text directly
 
        input_tokens = inputs_dict['input_ids'] 
        
        synonyms = self.get_output_tensor(input_tokens.tolist(),number_of_synonyms)
        
        # Sort words by order of importance
        
        
        i = 0
        cur_result = initial_result
        results = None
        # if len(index_order) == 1:
        #     index_order = index_order.item()
        # elif len(index_order) > 1:
        #     index_order = index_order.tolist()
        # else:
        #     ValueError()


        # print ('attacked_text',attacked_text)   
        # sys.exit() 
        attacked_text_list = []
        for atk_txt in attacked_text:
            words_from_text = utils.words_from_text(atk_txt.text)
            attacked_text_list.append(words_from_text)
        # print ('input_tokens',input_tokens.tolist()[0])
        # print ('attacked_text_list',attacked_text_list) 
        tokenized_text_ids_list = []
        for ipt_tok in input_tokens:
            tokenized_text_ids = self.victim_model.tokenizer.convert_ids_to_tokens(ipt_tok.tolist())  
            tokenized_text_ids_list.append(tokenized_text_ids)

        batch_wise_indexTotokens = []
        batch_wise_tokensToindex = []
        for b in range(len(attacked_text_list)):     
            indexTotokens, tokensToindex = self.gen_indexes(attacked_text_list[b], tokenized_text_ids_list[b])
            batch_wise_indexTotokens.append(indexTotokens)
            batch_wise_tokensToindex.append(tokensToindex)

        # print ('word from text',words_from_text)
        # print ('token text ids',tokenized_text_ids)
        # print ('idxtotoken',batch_wise_indexTotokens,'toks to ind',batch_wise_tokensToindex) 
        
        # print ('outer indextotokens',batch_wise_indexTotokens)
 
        i = 0
        unsuccesful_mask = [True]*batch_size
        # print ('index_order',index_order) 
        while i < 4:
            a = time.time() 
            # print ('i',i, index_order[b][i].item() ,batch_wise_tokensToindex[b])
            initial_text = [i.attacked_text for i in cur_result] 
            tokenizer_inputs = [init_text.tokenizer_input for init_text in initial_text]
            # print ('tokenizer_inputs',tokenizer_inputs)
            inputs_dict = self.victim_model.tokenizer(
                tokenizer_inputs,
                add_special_tokens=True,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            predictions , embeddings, input_tokens= victim_model.embedding_inference(inputs_dict)  
            index_order, search_over, gradient,ids,projected_mask= self._get_index_order(cur_result,inputs_dict,synonyms,predictions,embeddings,number_of_synonyms)

            
            # print ('cur result',cur_result)
            # print ('indiial',initial_text)
            
            indices_to_modify = index_order[:,0].tolist() 
            # print ('indices_to_modify',indices_to_modify)
            # print ('batch_wise_tokensToindex',batch_wise_tokensToindex)

            # print ('index_order',index_order)
            print ('synonyms shape',synonyms.shape) 
            i+=1
            for b in range(len(batch_wise_tokensToindex)):
                id_now = indices_to_modify[b]
                if id_now not in batch_wise_tokensToindex[b].keys():
                    
                    synonyms[b][id_now] = np.zeros(synonyms.shape[-1]) 
                    



            
            skip_this_token_mask = [True]*batch_size
            skip_this_token_mask = [True if token in batch_wise_tokensToindex[b].keys() else False for b, token in enumerate(indices_to_modify) ]
            
            # skip_this_index_mask = [True if batch_wise_tokensToindex[b][token] in batch_wise_indexTotokens[b].keys() else False for b, token in enumerate(indices_to_modify) ]
            
            
            skip_this_index_mask = []

            for b, token in enumerate(indices_to_modify):
                
                if token in batch_wise_tokensToindex[b].keys():

                    if batch_wise_tokensToindex[b][token] in batch_wise_indexTotokens[b].keys():
                        skip_this_index_mask.append(True)
                    else:
                        skip_this_index_mask.append(False)
                else:
                    skip_this_index_mask.append(False) 
            
            
            # print ('skip_this_index_mask',skip_this_index_mask,skip_this_index_mask[50],batch_wise_tokensToindex[50],indices_to_modify[50]) 



            indices_to_modify_index = [batch_wise_tokensToindex[b][token] if skip_this_token_mask[b] is True else -1  for b, token in enumerate(indices_to_modify) ]
            # print ('indices_to_modify_index',indices_to_modify_index)
             
            transformed_text_candidates,unsuccesfull_mask,synonyms = self.get_transformations_batch_wise(
                [cur_rlt.attacked_text for cur_rlt in cur_result ],
                original_text=initial_text,
                indices_to_modify=  indices_to_modify_index,
                **{'gradient':gradient,'ids':ids,'synonyms':synonyms,'predictions':predictions,'embeddings':embeddings,'projected_mask':projected_mask,'unsuccesful_mask':unsuccesful_mask,'skip_this_token_mask':skip_this_token_mask,'skipped_sample_mask':skipped_sample_mask,'indices_to_modify_index':indices_to_modify_index}
            )
            # print ('transformed_text_candidates',transformed_text_candidates)
            print ('transformations time taken:',time.time()-a)
            
            
             
            print ('len transformed_text_candidates',len(transformed_text_candidates))
            # print ('transformed_text_candidates 4',transformed_text_candidates[4])
            # print ('transformed_text_candidates 5',transformed_text_candidates[5])
            bt = time.time()
            updated_mask = []
            results_list = []
            for b in range(len(transformed_text_candidates)): 
                results, search_over = self.get_goal_results(transformed_text_candidates[b])
                results = sorted(results, key=lambda x: -x.score)
                # print ('result',results)
                results_list.append(results)
                updated_mask.append(search_over)
            
            # print ('results_list',results_list)
            # print ('updated_mask',updated_mask)
            
            # print ('transformed_text_candidates',transformed_text_candidates)
            # print ('results_list',results_list,i)
            # if i == 50:
            #     sys.exit()
            # print ('r1',results_list[0])
            # print ('r2',results_list[1])
            # print ('r3',results_list[1][0])
            
            for b in range(len(transformed_text_candidates)): 
                if len(results_list[b]) > 0:
                    # if isinstance(results_list[b][0],GoalFunctionResult):
                    
                    # print ('score is greater ','b',b,'i',i,results_list[b][0].score,cur_result[b].score)
                    if results_list[b][0].score > cur_result[b].score:
                        
                        cur_result[b] = results_list[b][0]
                    # else:
                    #     continue
                # if len(results_list[b]) > 0:
                #     print ('b',b,'i',i, results_list[b][0].score, cur_result[b].score)

                # else:
                #     print ('b',b,'i',i,'blank')
                    
                if cur_result[b].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    # need to create a mask that we pass to get_transformations_batch_wise that can say if a batch sample
                    # has been successful and no longer needs substitutions
                    # print ('cur_result[b].goal_status',b,cur_result[b].goal_status,GoalFunctionResultStatus.SUCCEEDED,'i',i)
                    unsuccesfull_mask[b] = False
                    # best_result = cur_result[b]
                    
                    # return best_result
                else:
                    continue

            

                # If we succeeded, return the index with best similarity.
                # if cur_result[b].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                #     best_result = cur_result
                #     # @O: Use vectorwise operations
                #     max_similarity = -float("inf")
                #     for result in results:
                #         if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                #             break
                #         candidate = result.attacked_text
                #         try:
                #             similarity_score = candidate.attack_attrs["similarity_score"]
                #         except KeyError:
                #             # If the attack was run without any similarity metrics,
                #             # candidates won't have a similarity score. In this
                #             # case, break and return the candidate that changed
                #             # the original score the most.
                #             break
                #         if similarity_score > max_similarity:
                #             max_similarity = similarity_score
                #             best_result = result
                #     return best_result
            # print ('cur_result',cur_result)
        cur_result[b].num_queries = 0 #self.goal_function.num_queries     
        
        
        return cur_result
    
    # def perform_search(self, initial_result):
    #     beam = [initial_result.attacked_text]
    #     best_result = initial_result
    #     print ('beam',beam)
    #     attacked_text = initial_result.attacked_text
    #     index_order, search_over = self._get_index_order(attacked_text)
    #     i = 0
    #     # while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
    #     while i < len(index_order) and not search_over:
    #         potential_next_beam = []
    #         for text in beam:
    #             print ('text',text)
    #             transformations = self.get_transformations(
    #                 text, original_text=initial_result.attacked_text, indices_to_modify=[index_order[i]]
    #             )
    #             print ('transformations',len(transformations))
    #             for i,tran in enumerate(transformations):
    #                 print (i,tran)
                    
                
    #             i+=1
    #             potential_next_beam += transformations 
    #             print ('potential next beams',len(potential_next_beam))
                
    #             if i == 3: 
    #                 sys.exit()
    #         if len(potential_next_beam) == 0:
    #             # If we did not find any possible perturbations, give up.
    #             return best_result
    #         results, search_over = self.get_goal_results(potential_next_beam)
    #         scores = np.array([r.score for r in results])
    #         best_result = results[scores.argmax()]
    #         if search_over:
    #             return best_result

    #         # Refill the beam. This works by sorting the scores
    #         # in descending order and filling the beam from there.
    #         best_indices = (-scores).argsort()[: self.beam_width]
    #         beam = [potential_next_beam[i] for i in best_indices]

    #     return best_result


    def perform_search_get_index_out(self, initial_result_gfs):
        number_of_synonyms = 50
        # print ('initial_result',initial_result_gfs)
        initial_result= initial_result_gfs['goal_function_result']
        skipped_sample_mask = initial_result_gfs['skipped_sample_mask'] 
        # initial result will include a skipped mask

        batch_size = len(initial_result) 


        attacked_text =  [i.attacked_text for i in initial_result]

        initial_text = [i.attacked_text for i in initial_result] 
        len_text, indices_to_order = self.get_indices_to_order_batchwise(initial_text) 
        # print ('initial text',initial_text)  
        self._initialize_model_and_vocab() 
        victim_model = self.victim_model# self.get_victim_model()
        # print ('len_text',len_text)
        # print ('max(len_text)',max(len_text))
        index_scores = np.zeros(shape=(batch_size,max(len_text)))


         # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.victim_model.tokenizer.model_max_length == int(1e30)
            else self.victim_model.tokenizer.model_max_length
        )


        tokenizer_inputs = [init_text.tokenizer_input for init_text in initial_text]
        # print ('tokenizer_inputs',tokenizer_inputs)
        inputs_dict = self.victim_model.tokenizer(
            tokenizer_inputs,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        # print ('input dict',inputs_dict)
        

        ### tokenize words from text directly
        ### tokenize words from text directly
 
        input_tokens = inputs_dict['input_ids'] 
        predictions , embeddings, input_tokens= victim_model.embedding_inference(inputs_dict)  
        synonyms = self.get_output_tensor(input_tokens.tolist(),number_of_synonyms)
        
        # Sort words by order of importance
        index_order, search_over, gradient,ids,projected_mask= self._get_index_order(initial_result,inputs_dict,synonyms,predictions,embeddings,number_of_synonyms)
 
        i = 0
        cur_result = initial_result
        results = None
        # if len(index_order) == 1:
        #     index_order = index_order.item()
        # elif len(index_order) > 1:
        #     index_order = index_order.tolist()
        # else:
        #     ValueError()


        # print ('attacked_text',attacked_text)   
        # sys.exit() 
        attacked_text_list = []
        for atk_txt in attacked_text:
            words_from_text = utils.words_from_text(atk_txt.text)
            attacked_text_list.append(words_from_text)
        # print ('input_tokens',input_tokens.tolist()[0])
        # print ('attacked_text_list',attacked_text_list) 
        tokenized_text_ids_list = []
        for ipt_tok in input_tokens:
            tokenized_text_ids = self.victim_model.tokenizer.convert_ids_to_tokens(ipt_tok.tolist())  
            tokenized_text_ids_list.append(tokenized_text_ids)

        batch_wise_indexTotokens = []
        batch_wise_tokensToindex = []
        for b in range(len(attacked_text_list)):     
            indexTotokens, tokensToindex = self.gen_indexes(attacked_text_list[b], tokenized_text_ids_list[b])
            batch_wise_indexTotokens.append(indexTotokens)
            batch_wise_tokensToindex.append(tokensToindex)

        # print ('word from text',words_from_text)
        # print ('token text ids',tokenized_text_ids)
        # print ('idxtotoken',batch_wise_indexTotokens,'toks to ind',batch_wise_tokensToindex) 
        
 
        i = 0
        unsuccesful_mask = [True]*batch_size
        # print ('index_order',index_order) 
        while i < len(index_order[b][:10]) and not search_over:
            a = time.time() 
            # print ('i',i, index_order[b][i].item() ,batch_wise_tokensToindex[b])

            
            # print ('cur result',cur_result)
            # print ('indiial',initial_text)
            
            indices_to_modify = index_order[:,i].tolist() 
            # print ('indices_to_modify',indices_to_modify)
            # print ('batch_wise_tokensToindex',batch_wise_tokensToindex)
            
            skip_this_token_mask = [True]*batch_size
            skip_this_token_mask = [True if token in batch_wise_tokensToindex[b].keys() else False for b, token in enumerate(indices_to_modify) ]
            # print ('skip_this_token_mask',skip_this_token_mask)

            indices_to_modify_index = [batch_wise_tokensToindex[b][token] if skip_this_token_mask[b] is True else -1  for b, token in enumerate(indices_to_modify) ]
            # print ('indices_to_modify_index',indices_to_modify_index)
            transformed_text_candidates,unsuccesfull_mask = self.get_transformations_batch_wise(
                [cur_rlt.attacked_text for cur_rlt in cur_result ],
                original_text=initial_text,
                indices_to_modify=  indices_to_modify_index,
                **{'gradient':gradient,'ids':ids,'synonyms':synonyms,'predictions':predictions,'embeddings':embeddings,'projected_mask':projected_mask,'unsuccesful_mask':unsuccesful_mask,'skip_this_token_mask':skip_this_token_mask,'skipped_sample_mask':skipped_sample_mask}
            )
            # print ('transformed_text_candidates',transformed_text_candidates)
            print ('transformations time taken:',time.time()-a)
            
            i+=1 
            
             
            bt = time.time()
            updated_mask = []
            results_list = []
            for b in range(len(transformed_text_candidates)):
                results, search_over = self.get_goal_results(transformed_text_candidates[b])
                results = sorted(results, key=lambda x: -x.score)
                # print ('result',results)
                results_list.append(results)
                updated_mask.append(search_over)
            
            # print ('results_list',results_list)
            # print ('updated_mask',updated_mask)
            
            # print ('transformed_text_candidates',transformed_text_candidates)
            # print ('results_list',results_list,i)
            # if i == 50:
            #     sys.exit()
            # print ('r1',results_list[0])
            # print ('r2',results_list[1])
            # print ('r3',results_list[1][0])
            
            for b in range(len(transformed_text_candidates)): 
                if len(results_list[b]) > 0:
                    # if isinstance(results_list[b][0],GoalFunctionResult):
                    
                    print ('score is greater ','b',b,'i',i,results_list[b][0].score,cur_result[b].score)
                    if results_list[b][0].score > cur_result[b].score:
                        
                        cur_result[b] = results_list[b][0]
                    # else:
                    #     continue
                # if len(results_list[b]) > 0:
                #     print ('b',b,'i',i, results_list[b][0].score, cur_result[b].score)

                # else:
                #     print ('b',b,'i',i,'blank')
                    
                if cur_result[b].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    # need to create a mask that we pass to get_transformations_batch_wise that can say if a batch sample
                    # has been successful and no longer needs substitutions
                    # print ('cur_result[b].goal_status',b,cur_result[b].goal_status,GoalFunctionResultStatus.SUCCEEDED,'i',i)
                    unsuccesfull_mask[b] = False
                    # best_result = cur_result[b]
                    
                    # return best_result
                else:
                    continue

            

                # If we succeeded, return the index with best similarity.
                # if cur_result[b].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                #     best_result = cur_result
                #     # @O: Use vectorwise operations
                #     max_similarity = -float("inf")
                #     for result in results:
                #         if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                #             break
                #         candidate = result.attacked_text
                #         try:
                #             similarity_score = candidate.attack_attrs["similarity_score"]
                #         except KeyError:
                #             # If the attack was run without any similarity metrics,
                #             # candidates won't have a similarity score. In this
                #             # case, break and return the candidate that changed
                #             # the original score the most.
                #             break
                #         if similarity_score > max_similarity:
                #             max_similarity = similarity_score
                #             best_result = result
                #     return best_result
            # print ('cur_result',cur_result)
        cur_result[b].num_queries = 0 #self.goal_function.num_queries     
        
        
        return cur_result

    def find_synonyms(self, word_idx,number_of_synonyms):
            # note that depending on how your distance_matrix is structured, 
            # this function might need to be adjusted
            word_info = self.smaller_dist_counter[word_idx]
            # synonyms_info = word_info[np.argsort(word_info[:, 1])[:4]]  # holds synonym id and distance
            synonyms_info = word_info[:number_of_synonyms,0]
            # print ('synonyms',synonyms_info)
            synonyms_info[synonyms_info == -1] = 0
            return synonyms_info



    def get_output_tensor(self, batch,number_of_synonyms ):
        batch_size = len(batch)

        output_tensor = np.zeros((batch_size,128, number_of_synonyms), dtype=np.int)

        for i in range(batch_size): #for i, sentence in enumerate(batch):
            for j,token in enumerate(batch[i]): #for j,token in enumerate(sentence):
            
                if token in self.filtered_vocab_ids:
                    # print ('token',token)
                    output_tensor[i,j] = self.find_synonyms(token,number_of_synonyms)
                    # print ('token',i,output_tensor[i])
                    # for j in output_tensor[i]:
                    #     print ('original',self.index_to_token[token],'word sub',self.index_to_token[j])
        
        return output_tensor

    def get_order(self,token_ids,synonyms,jacobian,modified_mask,number_of_synonyms):
        max_iter = 30
        grad_update_interval = 1
        batch_size = 1
        word_max_len = 128
        counter = 0

 
        # print ('shapes synonyms',synonyms.shape)
        # print ('shapes jacobian',jacobian.shape)
        

        synonyms = torch.tensor(synonyms).long() 
        token_ids = torch.tensor(token_ids).long()
        
        # we need to do this for 30 iterations
        # modified_num = torch.sum(modified_mask, dim=-1)
        # modified_ratio = (modified_num + 1) / len(token_ids)
        # print ('mods',modified_num,modified_ratio,len(token_ids))
        # sys.exit()

        # idealy precompute the token ids, at this time it's dynamic? maybe this is good
        # token_embeddings = self.embedding_matrix[token_ids] # shape [128, 768]
        # synonyms_embeddings = self.embedding_matrix[synonyms] # shape [128, 4, 768]

        token_embeddings = self.victim_model.model.get_input_embeddings()( token_ids.to(device)) 
        synonyms_embeddings = self.victim_model.model.get_input_embeddings()( synonyms.to(device)) 
        
        # print ('token emb shape',token_embeddings.shape)
        # print ('synonym emb shape',synonyms_embeddings.shape )
        
        # compute projection
        token_embeddings_expanded = token_embeddings.unsqueeze(2) # shape [128, 1, 768]
        jacobian_squeezed = jacobian.squeeze() # shape [128, 768]
        jacobian_expanded = jacobian_squeezed.unsqueeze(2) # shape [128, 1, 768]

        # print ('token_embeddings_expanded shape',token_embeddings_expanded.shape)
        # print ('jacobian shape',jacobian.shape)
        # print ('jacobian_squeezed shape',jacobian_squeezed.shape)
        # print ('jacobian_expanded shape', jacobian_expanded.shape)
        multi_jacob = (synonyms_embeddings - token_embeddings_expanded) * jacobian_expanded
        projection = torch.sum(multi_jacob, dim=-1) # final shape [128, 4]
        # print ('projection shpae',projection.shape)

        # create subword/not real word mask
        # print ('filtered vocab',self.filtered_vocab_ids)
        # print ('token_ids',token_ids)
        


        # just check at the end if origin exists in fitlered ids, if it dosn't return the new maks with this id
        # blocked
        # print ('synonyms shape',synonyms.shape)
        for b,tok_ids in enumerate(token_ids):
            for p,tok in enumerate(tok_ids): 
                if tok.item() not in self.filtered_vocab_ids:
                
                    synonyms[b][p] = torch.zeros(synonyms.shape[2])
        # an alternative way is to rank all substitutions, instead of keeping the first 
        
        
        # the alternative is to go though it all 
        # Step 3: Mask Projection. Substitution can only occur on known words.
        synonym_mask = (synonyms == 0).to(device) # mask where token id is 0 i.e., no synonyms found
        # print ('synonym_mask',synonym_mask)
        inf_tensor = torch.full(projection.shape, -1e9).to(device) # tensor filled with large negative values
        # print ('inf t',inf_tensor)
        projection_masked = projection.clone() # create a clone of the projection tensor housing the mask
        # print ('projection_masked',projection_masked)
        # apply mask to projection tensor
        
        projection_masked[synonym_mask] = inf_tensor[synonym_mask]
        
        # Step 4: Substitution
        # print ('projection_masked',projection_masked,projection_masked.shape)
        reduced = torch.max(projection_masked, dim=-1)[0] 
        # 
        # print ('reduced',reduced,reduced.shape) 
        # _, pos = torch.topk(reduced, k=grad_update_interval, dim=-1) # pos will be of shape [128, grad_update_interval]
        values, pos = torch.topk(reduced, k=number_of_synonyms, dim=-1) 
        # print ('pos',pos)
        return pos, projection_masked
    
    def project_synonyms2(self,token_ids,synonyms,jacobian,modified_mask):
        max_iter = 30
        grad_update_interval = 1
        batch_size = 1
        word_max_len = 128
        counter = 0

        synonyms = torch.tensor(synonyms).long()
        # jacobian = torch.tensor(jacobian)
        token_ids = torch.tensor(token_ids).long()
        
        # we need to do this for 30 iterations
        # modified_num = torch.sum(modified_mask, dim=-1)
        # modified_ratio = (modified_num + 1) / len(token_ids)
        # print ('mods',modified_num,modified_ratio,len(token_ids))
        # sys.exit()

        # idealy precompute the token ids, at this time it's dynamic? maybe this is good
        # token_embeddings = self.embedding_matrix[token_ids] # shape [128, 768]
        # synonyms_embeddings = self.embedding_matrix[synonyms] # shape [128, 4, 768]

        token_embeddings = self.victim_model.model.get_input_embeddings()( token_ids.to(device)) 
        synonyms_embeddings = self.victim_model.model.get_input_embeddings()( synonyms.to(device)) 
        

        # compute projection
        token_embeddings_expanded = token_embeddings.unsqueeze(1) # shape [128, 1, 768]
        jacobian_squeezed = jacobian.squeeze() # shape [128, 768]
        jacobian_expanded = jacobian_squeezed.unsqueeze(1) # shape [128, 1, 768]

        multi_jacob = (synonyms_embeddings - token_embeddings_expanded) * jacobian_expanded
        projection = torch.sum(multi_jacob, dim=-1) # final shape [128, 4]

        # create subword/not real word mask
        print ('filtered vocab',self.filtered_vocab_ids)
        print ('token_ids',token_ids)

        # just check at the end if origin exists in fitlered ids, if it dosn't return the new maks with this id
        # blocked
        for p,tok in enumerate(token_ids): 
            if tok.item() not in self.filtered_vocab_ids:
                
                synonyms[p] = torch.zeros(synonyms.shape[1])
        # an alternative way is to rank all substitutions, instead of keeping the first 
                
        # the alternative is to go though it all 
        # Step 3: Mask Projection. Substitution can only occur on known words.
        synonym_mask = (synonyms == 0).to(device) # mask where token id is 0 i.e., no synonyms found
        # print ('synonym_mask',synonym_mask)
        inf_tensor = torch.full(projection.shape, -1e9).to(device) # tensor filled with large negative values
        # print ('inf t',inf_tensor)
        projection_masked = projection.clone() # create a clone of the projection tensor housing the mask
        # print ('projection_masked',projection_masked)
        # apply mask to projection tensor
        
        projection_masked[synonym_mask] = inf_tensor[synonym_mask]
        
        # Step 4: Substitution
        print ('projection_masked',projection_masked,projection_masked.shape)
        reduced = torch.max(projection_masked, dim=-1)[0] 
        
        print ('reduced',reduced,reduced.shape) 
        # _, pos = torch.topk(reduced, k=grad_update_interval, dim=-1) # pos will be of shape [128, grad_update_interval]
        values, pos = torch.topk(reduced, k=4, dim=-1)
        if one_step:
            return pos
        pos = pos.unsqueeze(-1)
        print ('pos',pos,pos.shape)  
        print ('vals',values)
        sys.exit()
        # Creating serial tensor which is equivalent to creating a tensor of indices ranging from 0 to batch size
        
        serial = torch.arange(start=0, end=batch_size).unsqueeze(-1).repeat_interleave(1, grad_update_interval).to(device)
        print ('serial',serial,serial.shape) 

        # Creating a combined tensor of serial and pos
        indices = torch.stack([serial, pos], dim=-1)
        print ('indices',indices,indices.shape)
        # Unsuccessful mask is assumed to be a 1D boolean tensor of size 128
        indices_masked = indices # [unsuccessful_mask] # shape depends on unsuccessful_mask
        indices_masked = indices_masked.squeeze(1) 
        print ('indices_masked',indices_masked, indices_masked.shape)
        # Select original token_ids where substitution was unsuccessful
        origin = token_ids[indices_masked[0][1]] #  adv_xs # [unsuccessful_mask] # shape depends on unsuccessful_mask

        # For each token_id (where substitution was unsuccessful), find the synonym id that had the maximum projection_masked
        # synonym_pos = torch.argmax(projection_masked, dim=-1, keepdim=True)# [unsuccessful_mask] # shape depends on unsuccessful_mask
        # past this position, defo dos't work with batches
        
        print ('projection_masked',projection_masked, projection_masked.shape)
        argmax_projection = torch.argmax(projection_masked, dim=-1, keepdim=True)# [unsuccessful_mask] # shape depends on unsuccessful_mask
        argmax_projection = argmax_projection.squeeze(-1)
        argmax_projection = argmax_projection.unsqueeze(0)
        print ('argmax proj',argmax_projection,argmax_projection.shape)
        print ('indices_masked',indices_masked,indices_masked[0][0], indices_masked[0][1]) 
        gather_nd_projection = argmax_projection[indices_masked[0][0], indices_masked[0][1]]
        # gather_nd_projection = argmax_projection[indices_masked[:,0] , indices_masked[:,1] ]
        print ('gather_nd_projection',gather_nd_projection) 
        
        synonym_pos = gather_nd_projection.unsqueeze(-1)
        synonym_pos = synonym_pos.unsqueeze(-1)
        # indices_m_expanded = indices_masked.unsqueeze(-1)



        
        print ('synonym_pos',synonym_pos, synonym_pos.shape)  
        # Select the synonyms using the masked indices
        print ('indices masked',indices_masked) 
        synonym_indices_masked = torch.cat([indices_masked, synonym_pos], dim=-1) 
        print ('synonym_indices_masked',synonym_indices_masked)
        print ('synonyms',synonyms,synonyms.shape)
        synonym = synonyms[synonym_indices_masked[0][1], synonym_indices_masked[0][2]]
        print ('synonym',synonym,synonym.shape) 
        # Create delta tensor
        # delta = torch.sparse.FloatTensor(indices_masked.t(), (synonym.item() - origin.item()), (batch_size, word_max_len))
        delta = synonym - origin
        print ('delta',delta,token_ids.shape, origin)
        print ('synonym', self.victim_model.tokenizer.convert_ids_to_tokens(synonym.item()))
        print ('origin', self.victim_model.tokenizer.convert_ids_to_tokens(origin.item()))
        token_ids[indices_masked[0][1]] = origin + delta
        print ('token ids',token_ids,self.victim_model.tokenizer.convert_ids_to_tokens(token_ids.tolist()))
        
        synonyms[indices_masked[0][1]] = torch.zeros(synonyms.shape[1])
        
        # Perform substitution
        # adv_xs = adv_xs + delta.to_dense()
        
        return  token_ids, synonyms, modified_mask

        # return adv_xs



        

    @property
    def is_black_box(self):
        return False

    def extra_repr_keys(self):
        return ["beam_width"]


# ### tokenize words from text directly
#         def zero_out_larger_entries(tensor):
#             # Sum along the inner dimension to create a mask of zeros and non-zeros
#             non_zero_counts = tensor.ne(0).sum(dim=-1)
#             print ('non_zero_counts',non_zero_counts.shape,non_zero_counts)
#             # Build a mask where the non zero elements are more than 3 (2 padding + 1 embedding).
#             mask = non_zero_counts <= 3
#             print ('mask',mask.shape,mask)

#             mask = mask.unsqueeze(-1).expand_as(tensor)
#             print ('mask',mask.shape,mask)
#             # Use this mask to zero out tensor's elements where corresponding mask value is False
#             tensor[~mask] = 0
#             print ('tensor',tensor.shape)
#             return tensor
        
#         def keep_embeddings_only(tensor):
#             # Get the middle value by slicing the tensor
#             embeddings = tensor[:, 1]

#             return embeddings
#         words_from_text = utils.words_from_text(attacked_text.text)
        
#         print ('words_from_text',words_from_text)
#         inputs_dict = self.victim_model.tokenizer(
#             words_from_text ,
#             add_special_tokens=True,
#             padding="max_length",
#             max_length=max_length,
#             truncation=True,
#             return_tensors="pt",
#         )  
        
#         input_tensor = zero_out_larger_entries(inputs_dict['input_ids'])
#         input_tensor = keep_embeddings_only(input_tensor)