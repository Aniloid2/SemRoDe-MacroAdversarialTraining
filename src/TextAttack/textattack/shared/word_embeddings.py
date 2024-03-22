"""
Shared loads word embeddings and related distances
=====================================================
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import os
import pickle

import numpy as np
import torch
import sys

from textattack.shared import utils


class AbstractWordEmbedding(utils.ReprMixin, ABC):
    """Abstract class representing word embedding used by TextAttack.

    This class specifies all the methods that is required to be defined
    so that it can be used for transformation and constraints. For
    custom word embedding not supported by TextAttack, please create a
    class that inherits this class and implement the required methods.
    However, please first check if you can use `WordEmbedding` class,
    which has a lot of internal methods implemented.
    """

    @abstractmethod
    def __getitem__(self, index):
        """Gets the embedding vector for word/id
        Args:
            index (Union[str|int]): `index` can either be word or integer representing the id of the word.
        Returns:
            vector (ndarray): 1-D embedding vector. If corresponding vector cannot be found for `index`, returns `None`.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_mse_dist(self, a, b):
        """Return MSE distance between vector for word `a` and vector for word
        `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): MSE (L2) distance
        """
        raise NotImplementedError()

    @abstractmethod
    def get_cos_sim(self, a, b):
        """Return cosine similarity between vector for word `a` and vector for
        word `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): cosine similarity
        """
        raise NotImplementedError()

    @abstractmethod
    def word2index(self, word):
        """
        Convert between word to id (i.e. index of word in embedding matrix)
        Args:
            word (str)
        Returns:
            index (int)
        """
        raise NotImplementedError()

    @abstractmethod
    def index2word(self, index):
        """
        Convert index to corresponding word
        Args:
            index (int)
        Returns:
            word (str)
        """
        raise NotImplementedError()

    @abstractmethod
    def nearest_neighbours(self, index, topn):
        """
        Get top-N nearest neighbours for a word
        Args:
            index (int): ID of the word for which we're finding the nearest neighbours
            topn (int): Used for specifying N nearest neighbours
        Returns:
            neighbours (list[int]): List of indices of the nearest neighbours
        """
        raise NotImplementedError()


class WordEmbedding(AbstractWordEmbedding):
    """Object for loading word embeddings and related distances for TextAttack.
    This class has a lot of internal components (e.g. get consine similarity)
    implemented. Consider using this class if you can provide the appropriate
    input data to create the object.

    Args:
        emedding_matrix (ndarray): 2-D array of shape N x D where N represents size of vocab and D is the dimension of embedding vectors.
        word2index (Union[dict|object]): dictionary (or a similar object) that maps word to its index with in the embedding matrix.
        index2word (Union[dict|object]): dictionary (or a similar object) that maps index to its word.
        nn_matrix (ndarray): Matrix for precomputed nearest neighbours. It should be a 2-D integer array of shape N x K
            where N represents size of vocab and K is the top-K nearest neighbours. If this is set to `None`, we have to compute nearest neighbours
            on the fly for `nearest_neighbours` method, which is costly.
    """

    PATH = "word_embeddings"

    def __init__(self, embedding_matrix, word2index, index2word, nn_matrix=None):
        self.embedding_matrix = embedding_matrix
        self._word2index = word2index
        self._index2word = index2word
        self.nn_matrix = nn_matrix

        # Dictionary for caching results
        self._mse_dist_mat = defaultdict(dict)
        self._cos_sim_mat = defaultdict(dict)
        self._nn_cache = {}

    def __getitem__(self, index):
        """Gets the embedding vector for word/id
        Args:
            index (Union[str|int]): `index` can either be word or integer representing the id of the word.
        Returns:
            vector (ndarray): 1-D embedding vector. If corresponding vector cannot be found for `index`, returns `None`.
        """
        if isinstance(index, str):
            try:
                index = self._word2index[index]
            except KeyError:
                return None
        try:
            return self.embedding_matrix[index]
        except IndexError:
            # word embedding ID out of bounds
            return None

    def word2index(self, word):
        """
        Convert between word to id (i.e. index of word in embedding matrix)
        Args:
            word (str)
        Returns:
            index (int)
        """
        return self._word2index[word]

    def index2word(self, index):
        """
        Convert index to corresponding word
        Args:
            index (int)
        Returns:
            word (str)

        """
        return self._index2word[index]

    def get_mse_dist(self, a, b):
        """Return MSE distance between vector for word `a` and vector for word
        `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): MSE (L2) distance
        """
        if isinstance(a, str):
            a = self._word2index[a]
        if isinstance(b, str):
            b = self._word2index[b]
        a, b = min(a, b), max(a, b)
        try:
            mse_dist = self._mse_dist_mat[a][b]
        except KeyError:
            e1 = self.embedding_matrix[a]
            e2 = self.embedding_matrix[b]
            e1 = torch.tensor(e1).to(utils.device)
            e2 = torch.tensor(e2).to(utils.device)
            mse_dist = torch.sum((e1 - e2) ** 2).item()
            self._mse_dist_mat[a][b] = mse_dist

        return mse_dist

    def get_cos_sim(self, a, b):
        """Return cosine similarity between vector for word `a` and vector for
        word `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): cosine similarity
        """
        if isinstance(a, str):
            a = self._word2index[a]
        if isinstance(b, str):
            b = self._word2index[b]
        a, b = min(a, b), max(a, b)
        try:
            cos_sim = self._cos_sim_mat[a][b]
        except KeyError:
            e1 = self.embedding_matrix[a]
            e2 = self.embedding_matrix[b]
            e1 = torch.tensor(e1).to(utils.device)
            e2 = torch.tensor(e2).to(utils.device)
            cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2).item()
            self._cos_sim_mat[a][b] = cos_sim
        return cos_sim

    def nearest_neighbours(self, index, topn):
        """
        Get top-N nearest neighbours for a word
        Args:
            index (int): ID of the word for which we're finding the nearest neighbours
            topn (int): Used for specifying N nearest neighbours
        Returns:
            neighbours (list[int]): List of indices of the nearest neighbours
        """
        if isinstance(index, str):
            index = self._word2index[index]
        if self.nn_matrix is not None:
            nn = self.nn_matrix[index][1 : (topn + 1)]
        else:
            try:
                nn = self._nn_cache[index]
            except KeyError:
                embedding = torch.tensor(self.embedding_matrix).to(utils.device)
                vector = torch.tensor(self.embedding_matrix[index]).to(utils.device)
                dist = torch.norm(embedding - vector, dim=1, p=None)
                # Since closest neighbour will be the same word, we consider N+1 nearest neighbours
                
                # print ('dist',dist.topk(topn + 1, largest=False).indices)
                # nn = dist.topk(topn + 1, largest=False)[:1].tolist()
                nn = dist.topk(topn + 1, largest=False).indices.tolist()
                self._nn_cache[index] = nn

        return nn

    @staticmethod
    def counterfitted_GLOVE_embedding():
        """Returns a prebuilt counter-fitted GLOVE word embedding proposed by
        "Counter-fitting Word Vectors to Linguistic Constraints" (Mrkšić et
        al., 2016)"""
        if (
            "textattack_counterfitted_GLOVE_embedding" in utils.GLOBAL_OBJECTS
            and isinstance(
                utils.GLOBAL_OBJECTS["textattack_counterfitted_GLOVE_embedding"],
                WordEmbedding,
            )
        ):
            # avoid recreating same embedding (same memory) and instead share across different components
            return utils.GLOBAL_OBJECTS["textattack_counterfitted_GLOVE_embedding"]

        word_embeddings_folder = "paragramcf"
        word_embeddings_file = "paragram.npy"
        word_list_file = "wordlist.pickle"
        mse_dist_file = "mse_dist.p"
        cos_sim_file = "cos_sim.p"
        nn_matrix_file = "nn.npy"


        
        print(WordEmbedding.PATH)
        
        # Download embeddings if they're not cached.
        word_embeddings_folder = os.path.join(
            WordEmbedding.PATH, word_embeddings_folder
        ).replace("\\", "/")


        word_embeddings_folder = utils.download_from_s3(word_embeddings_folder)
        
        # Concatenate folder names to create full path to files.
        word_embeddings_file = os.path.join(
            word_embeddings_folder, word_embeddings_file
        )
        word_list_file = os.path.join(word_embeddings_folder, word_list_file)
        mse_dist_file = os.path.join(word_embeddings_folder, mse_dist_file)
        cos_sim_file = os.path.join(word_embeddings_folder, cos_sim_file)
        nn_matrix_file = os.path.join(word_embeddings_folder, nn_matrix_file)

        # loading the files
        embedding_matrix = np.load(word_embeddings_file)
 
        word2index = np.load(word_list_file, allow_pickle=True) 
        index2word = {}
        for word, index in word2index.items():
            index2word[index] = word
        nn_matrix = np.load(nn_matrix_file)

        
        
        embedding = WordEmbedding(embedding_matrix, word2index, index2word, nn_matrix)

        with open(mse_dist_file, "rb") as f:
            mse_dist_mat = pickle.load(f)
        with open(cos_sim_file, "rb") as f:
            cos_sim_mat = pickle.load(f)
            
        embedding._mse_dist_mat = mse_dist_mat
        embedding._cos_sim_mat = cos_sim_mat

        utils.GLOBAL_OBJECTS["textattack_counterfitted_GLOVE_embedding"] = embedding

        return embedding


# EMBEDDING_PATH = "word_embeddings/glove200"

# def __init__(self, emb_layer_trainable=True):
#     glove_path = utils.download_from_s3(GloveEmbeddingLayer.EMBEDDING_PATH)
#     glove_word_list_path = os.path.join(glove_path, "glove.wordlist.npy")
#     word_list = np.load(glove_word_list_path)
#     glove_matrix_path = os.path.join(glove_path, "glove.6B.200d.mat.npy")
#     embedding_matrix = np.load(glove_matrix_path)
#     super().__init__(embedding_matrix=embedding_matrix, word_list=word_list)
#     self.embedding.weight.requires_grad = emb_layer_trainable



    @staticmethod
    def GLOVE_embedding():
        """Returns a prebuilt counter-fitted GLOVE word embedding proposed by
        "Counter-fitting Word Vectors to Linguistic Constraints" (Mrkšić et
        al., 2016)"""
        
        if (
            "textattack_GLOVE_embedding" in utils.GLOBAL_OBJECTS
            and isinstance(
                utils.GLOBAL_OBJECTS["textattack_GLOVE_embedding"],
                WordEmbedding,
            )
        ):
            # avoid recreating same embedding (same memory) and instead share across different components
            return utils.GLOBAL_OBJECTS["textattack_GLOVE_embedding"]


        link_address = 'https://nlp.stanford.edu/data/glove.6B.zip'
       
        word_embeddings_folder = "glove_folder_intersection"
        word_embeddings_file = "glove.6B.200d.txt"
        word_embeddings_matrix = "glove.6B.200d.npy"
        word_list_file = "glove.wordlist.npy"
        mse_dist_file = "glove_mse_dist.p"
        cos_sim_file = "glove_cos_sim.p"
        nn_matrix_file = "glove_nn.npy"
        size_d = 200
        topn = 100

        
        print(WordEmbedding.PATH)
        
        # Download embeddings if they're not cached.
        word_embeddings_folder = os.path.join(
            WordEmbedding.PATH, word_embeddings_folder
        ).replace("\\", "/")

         
        # word_embeddings_folder = utils.download_from_url(link_address,word_embeddings_folder) 

        os.makedirs(  os.path.join(utils.TEXTATTACK_CACHE_DIR, word_embeddings_folder), exist_ok=True) 
        word_embeddings_origin_folder = os.path.join(
             utils.TEXTATTACK_CACHE_DIR,WordEmbedding.PATH
        ).replace("\\", "/")

        word_embeddings_final_folder = os.path.join(
            utils.TEXTATTACK_CACHE_DIR,word_embeddings_folder
        ).replace("\\", "/")

        list_glove = ['glove.6B.100d.txt',  'glove.6B.200d.txt',  'glove.6B.300d.txt',  'glove.6B.50d.txt']
        for L in list_glove:
            # print ('osjoin',os.path.join( word_embeddings_origin_folder, L), os.path.join( word_embeddings_final_folder,L))
            # sys.exit()
            if os.path.isfile(os.path.join( word_embeddings_origin_folder, L)):
                os.rename(os.path.join( word_embeddings_origin_folder, L), os.path.join( word_embeddings_final_folder,L))
            
            
            # sys.exit()
        # move the extracted files to word_embeddings_folder

        # Concatenate folder names to create full path to files.
        word_embeddings_file = os.path.join(
            utils.TEXTATTACK_CACHE_DIR,word_embeddings_folder, word_embeddings_file
        )
        word_embeddings_matrix = os.path.join(
            utils.TEXTATTACK_CACHE_DIR,word_embeddings_folder, word_embeddings_matrix
        )
        word_list_file = os.path.join(utils.TEXTATTACK_CACHE_DIR,word_embeddings_folder, word_list_file)
        mse_dist_file = os.path.join(utils.TEXTATTACK_CACHE_DIR,word_embeddings_folder, mse_dist_file)
        cos_sim_file = os.path.join(utils.TEXTATTACK_CACHE_DIR,word_embeddings_folder, cos_sim_file)
        nn_matrix_file = os.path.join(utils.TEXTATTACK_CACHE_DIR,word_embeddings_folder, nn_matrix_file)

        with open(word_embeddings_file, 'r', encoding="utf-8") as f:
            total_file = f.readlines() 
        # build components here from scratch
        
        

        # if os.path.isfile(os.path.join( word_embeddings_final_folder, word_embeddings_matrix)):
        if os.path.isfile( word_embeddings_matrix):
            # load
            embedding_matrix = np.load(word_embeddings_matrix,allow_pickle=True)
            word2index = np.load(word_list_file,allow_pickle=True)
        else:
            embedding_matrix = np.zeros((len(total_file), size_d))
            word2index = {}
            counter = 0
            with open(word_embeddings_file, 'r', encoding="utf-8") as f:
                
                for line in f: 
                    values = line.split()
                    word = values[0]
                    word2index[word] = counter
                    vector = np.asarray(values[1:], dtype=np.float32)
                    embedding_matrix[counter] = vector
                    counter +=1 
            
            np.save(word_embeddings_matrix, embedding_matrix,allow_pickle=True) 
            np.save(word_list_file, word2index,allow_pickle=True) 

        

          
        
        index2word = {}
        word2index = word2index.item()
        for word, index in word2index.items():
            index2word[index] = word



        embedding_matrix,word2index,index2word  = WordEmbedding.clean_glove_embeddings(embedding_matrix,word2index,index2word)
        # counter_fit = WordEmbedding.counterfitted_GLOVE_embedding()
        # cf_word2index = counter_fit._word2index



        # print ('len cf',len(cf_word2index),cf_word2index['hi'] )

        # f_word2index = {x:word2index[x] for x in word2index if x in cf_word2index}
        # print ('final dic',len(f_word2index))

        # f_index2word = {}
        # f_word2index = f_word2index.item()
        # for word, index in f_word2index.items():
        #     f_index2word[index] = word
        
        # print ('f ind2wor',len(f_index2word))
        

        # print ('start exploring') 
        # words_to_ignore = ['rank','—', 'honest','…','stickiness','_____________','cheeky','____________________________________________']
        # index_to_ignore = []
        # for i in range(400000):
        #     word = index2word[i]
            
        #     if word in words_to_ignore :
        #         index_to_ignore.append(i)
        #         print ('elipsi',i,word)
                
        # print ('end exploring')
        
        # print ('emb mat shape before',embedding_matrix.shape)
        # for wti in index_to_ignore:

        #     embedding_matrix = np.delete(embedding_matrix, wti, 0)
        # print ('emb mat shape aft',embedding_matrix.shape)

        
        
        
        

        embedding = WordEmbedding(embedding_matrix, word2index, index2word, None)
        
        # # neeed to do nn matrix
        # self._index2word = index2word
        # self._word2index = word2index
        # self.embedding_matrix = embedding_matrix
        # if os.path.isfile(os.path.join( word_embeddings_final_folder, nn_matrix_file)):
        
        
        if os.path.isfile(nn_matrix_file):
            # load
            nn_matrix = np.load(nn_matrix_file,allow_pickle=True) 
            
        else:
            
            nn_matrix = np.zeros((len(embedding_matrix), topn+1))
            
            
            for i in range( len(embedding_matrix)):
                print ('nn matrix',i)
                nn_local = embedding.nearest_neighbours(i,topn = topn) 
                nn_matrix[i] = torch.Tensor(nn_local)
                
            embedding.nn_matrix = nn_matrix
            np.save(nn_matrix_file,nn_matrix,allow_pickle=True)


        # for wti in index_to_ignore:
        #     nn_matrix = np.delete(nn_matrix, wti, 0)
        embedding.nn_matrix = nn_matrix 

        
        # print ('nn matrix loaded')


        indexs, topk = embedding.nn_matrix.shape
         
        # can try without cos sim mat and mse dist mat
        # mse_dist_mat = np.zeros((len(total_file), len(total_file)))
        # if os.path.isfile(os.path.join( word_embeddings_final_folder, cos_sim_file)):
        
        
        if os.path.isfile(cos_sim_file):
            # outer_dic = np.load(cos_sim_file+'.npy',allow_pickle=True) 
            fileObj = open(cos_sim_file, 'rb')
            cos_sim_mat = pickle.load(fileObj)
            fileObj.close() 
        else:
            outer_dic = {} 
            for i in range(indexs): #int(len(total_file)//2) 
                print ('cos',i)
                acc_i = int(i) 
                inner_dic = {}
                list_topk = embedding.nn_matrix[acc_i]
                # print ('i and top k',acc_i,list_topk)
                for j in list_topk: 
                    acc_j = int(j)
                    get_cos = embedding.get_cos_sim(acc_i,acc_j)
                    # print ('cos',acc_j,get_cos)
                    if acc_j in outer_dic:
                        pass
                    else:
                        inner_dic[acc_j] = get_cos
                    
                    # try:
                    #     # outer_dic[acc_i][acc_j] = get_cos
                    #     pass
                    # except Exception as e:
                                            
                    #     inner_dic[acc_j] = get_cos

                    
                outer_dic[acc_i] = inner_dic 
                
            # np.save(cos_sim_file,outer_dic,allow_pickle=True)  
            fileObj = open(cos_sim_file, 'wb')
            cos_sim_mat = pickle.dump(outer_dic,fileObj)
            fileObj.close()

        embedding._cos_sim_mat = cos_sim_mat 

         
 
        if os.path.isfile(mse_dist_file):
            # outer_dic = np.load(mse_dist_file+'.npy',allow_pickle=True) 
            fileObj = open(mse_dist_file, 'rb')
            mse_dist_mat = pickle.load(fileObj)
            fileObj.close() 
        else:
            outer_dic = {} 
            for i in range(indexs): #int(len(total_file)//2) 
                print ('mse',i)
                acc_i = int(i) 
                inner_dic = {}
                list_topk = embedding.nn_matrix[acc_i]
                # print ('i and top k',acc_i,list_topk) 
                for j in list_topk:  
                    acc_j = int(j)
                    get_cos = embedding.get_mse_dist(acc_i,acc_j)
                    # print ('cos',acc_j,get_cos)
                    if acc_j in outer_dic:
                        pass
                    else:
                        inner_dic[acc_j] = get_cos
                    
                    # try:
                    #     # outer_dic[acc_i][acc_j] = get_cos
                    #     pass
                    # except Exception as e:
                                            
                    #     inner_dic[acc_j] = get_cos 
                outer_dic[acc_i] = inner_dic 
                
            # np.save(mse_dist_file,outer_dic,allow_pickle=True)  
            fileObj = open(mse_dist_file, 'wb')
            mse_dist_mat = pickle.dump(outer_dic,fileObj)
            fileObj.close()
 
        embedding._mse_dist_mat = mse_dist_mat 

        # clean globe embeddings 
         
        utils.GLOBAL_OBJECTS["textattack_GLOVE_embedding"] = embedding

        return embedding

    def clean_glove_embeddings(embedding_matrix,word2index,index2word):
        counter_fit = WordEmbedding.counterfitted_GLOVE_embedding()
        cf_word2index = counter_fit._word2index
        cf_index2word = counter_fit._index2word
        cf_nn_matrix = counter_fit.nn_matrix
        cf_embedding_matrix = counter_fit.embedding_matrix
        cf_mse_dist_mat = counter_fit._mse_dist_mat
        cf_cos_sim_mat = counter_fit._cos_sim_mat 
 
        # word2index = embeddings._word2index
        # index2word = embeddings._index2word
        # nn_matrix = embeddings.nn_matrix
        # embedding_matrix = embeddings.embedding_matrix
        # mse_dist_mat = embeddings._mse_dist_mat
        # cos_sim_mat = embeddings._cos_sim_mat 
        
        f_word2index = {}
        index_to_ignore = []

        for i,x in enumerate(word2index.items()):
            
            if x[0] in cf_word2index.keys():
                f_word2index[x[0]] = word2index[x[0]]
            else:
                index_to_ignore.append(i)
        # print ('id ignore',index_to_ignore)        
        
        # breakpoint()
        # f_word2index = {x:word2index[x] for x in word2index if x in cf_word2index}


 
        f_index2word = {}
        
        for word, index in f_word2index.items():
            f_index2word[index] = word
            

        # def remove_index(dictionary1, dictionary2, index):
        #     # Remove index from both dictionaries
        #     # del dictionary1[index]
        #     word_to_index = {word: i for word, i in dictionary2.items() if i != index}
            
        #     # Update index values in the second dictionary
        #     updated_dictionary2 = {}
        #     for word, i in word_to_index.items():
        #         if i > index:
        #             updated_dictionary2[word] = i - 1
        #         else:
        #             updated_dictionary2[word] = i
            
        #     # Update index values in the first dictionary
        #     updated_dictionary1 = {i: word for i, word in enumerate(dictionary1.values())}
            
        #     return updated_dictionary1, updated_dictionary2

        # updated_dict1 = f_index2word
        # updated_dict2 = f_word2index

     
     
        # for removed_index in index_to_ignore:
        #     updated_dict1, updated_dict2 = remove_index(updated_dict1, updated_dict2, removed_index)
  
        
        # f_index2word = updated_dict1
        # f_word2index = updated_dict2
        # print ('len id2word',len(f_index2word), len(f_word2index))
        # print ('first id',f_index2word[0])
        # print ('reset ids')
        # breakpoint()
        intersection = list(f_index2word.keys())
        embedding_matrix = np.take(embedding_matrix, intersection, 0) 


        # f_keys = f_index2word.keys()
        # print ('f keys',f_keys)
        sored_list = list(f_index2word.keys())
        # sored_list.sort()
        
        
        counter = 0
        n_index2word = {}
        n_word2index = {}
        while f_index2word:
            old_index = sored_list[counter]
            # print ('old index',old_index)
            old_word = f_index2word[old_index]
            n_index2word[counter] = old_word
            n_word2index[old_word] = counter
            f_index2word.pop(old_index)
            counter+=1
        # print ('index word',n_index2word)
        # breakpoint()



        # for wti in index_to_ignore:

        #     embedding_matrix = np.delete(embedding_matrix, wti, 0) 

        # print ('shape embd',embedding_matrix.shape)
        
        # delete
        # for wti in index_to_ignore:
        #     nn_matrix = np.delete(nn_matrix, wti, 0)    
        
        # print (nn_matrix.shape)


        # # we need a index to index dic, that can translate between the index from old 400kembeddings to new 62kmebeddings
        # # to build it we use the words2index from both (f_word2index and word2index) 
        # index2index_map
        # for each word in word2index:
        #     if word in f_word2index
        #         index2index_map[word2index[word]] = f_word2index[word]




        # for each token_mse_values entry in mse_dic ,
        #     for each token_token in token_mse_values:
        #         # need to have mapping from old to new index to word
        #         if index2index_map[token_token] in index_to_ignore:
        #             token_mse_values.pop(token_token)
        #         else:
        #             tmp_vals = token_token.val
        #             token_mse_values.pop(token_token)
        #             token_mse_values[index2index_map[token_token]] = tmp_vals

        return embedding_matrix,n_word2index,n_index2word 

class GensimWordEmbedding(AbstractWordEmbedding):
    """Wraps Gensim's `models.keyedvectors` module
    (https://radimrehurek.com/gensim/models/keyedvectors.html)"""

    def __init__(self, keyed_vectors):
        gensim = utils.LazyLoader("gensim", globals(), "gensim")

        if isinstance(
            keyed_vectors, gensim.models.keyedvectors.WordEmbeddingsKeyedVectors
        ):
            self.keyed_vectors = keyed_vectors
        else:
            raise ValueError(
                "`keyed_vectors` argument must be a "
                "`gensim.models.keyedvectors.WordEmbeddingsKeyedVectors` object"
            )

        self.keyed_vectors.init_sims()
        self._mse_dist_mat = defaultdict(dict)
        self._cos_sim_mat = defaultdict(dict)

    def __getitem__(self, index):
        """Gets the embedding vector for word/id
        Args:
            index (Union[str|int]): `index` can either be word or integer representing the id of the word.
        Returns:
            vector (ndarray): 1-D embedding vector. If corresponding vector cannot be found for `index`, returns `None`.
        """
        if isinstance(index, str):
            try:
                index = self.keyed_vectors.vocab.get(index).index
            except KeyError:
                return None
        try:
            return self.keyed_vectors.vectors_norm[index]
        except IndexError:
            # word embedding ID out of bounds
            return None

    def word2index(self, word):
        """
        Convert between word to id (i.e. index of word in embedding matrix)
        Args:
            word (str)
        Returns:
            index (int)
        """
        vocab = self.keyed_vectors.vocab.get(word)
        if vocab is None:
            raise KeyError(word)
        return vocab.index

    def index2word(self, index):
        """
        Convert index to corresponding word
        Args:
            index (int)
        Returns:
            word (str)

        """
        try:
            # this is a list, so the error would be IndexError
            return self.keyed_vectors.index2word[index]
        except IndexError:
            raise KeyError(index)

    def get_mse_dist(self, a, b):
        """Return MSE distance between vector for word `a` and vector for word
        `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): MSE (L2) distance
        """
        try:
            mse_dist = self._mse_dist_mat[a][b]
        except KeyError:
            e1 = self.keyed_vectors.vectors_norm[a]
            e2 = self.keyed_vectors.vectors_norm[b]
            e1 = torch.tensor(e1).to(utils.device)
            e2 = torch.tensor(e2).to(utils.device)
            mse_dist = torch.sum((e1 - e2) ** 2).item()
            self._mse_dist_mat[a][b] = mse_dist
        return mse_dist

    def get_cos_sim(self, a, b):
        """Return cosine similarity between vector for word `a` and vector for
        word `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): cosine similarity
        """
        if not isinstance(a, str):
            a = self.keyed_vectors.index2word[a]
        if not isinstance(b, str):
            b = self.keyed_vectors.index2word[b]
        cos_sim = self.keyed_vectors.similarity(a, b)
        return cos_sim

    def nearest_neighbours(self, index, topn, return_words=True):
        """
        Get top-N nearest neighbours for a word
        Args:
            index (int): ID of the word for which we're finding the nearest neighbours
            topn (int): Used for specifying N nearest neighbours
        Returns:
            neighbours (list[int]): List of indices of the nearest neighbours
        """
        word = self.keyed_vectors.index2word[index]
        return [
            self.word2index(i[0])
            for i in self.keyed_vectors.similar_by_word(word, topn)
        ]
