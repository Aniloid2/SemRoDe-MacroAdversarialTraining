"""

HotFlip
===========
(HotFlip: White-Box Adversarial Examples for Text Classification)

"""
from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    InputColumnModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import BeamSearch,GreedyWordSwapWIR
from textattack.transformations import WordSwapGradientBased 
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from .attack_recipe import AttackRecipe
import math
from textattack.constraints.overlap import MaxWordsPerturbed 

# class HotFlipEbrahimi2017(AttackRecipe):
#     """Ebrahimi, J. et al. (2017)

#     HotFlip: White-Box Adversarial Examples for Text Classification

#     https://arxiv.org/abs/1712.06751

#     This is a reproduction of the HotFlip word-level attack (section 5 of the
#     paper).
#     """

#     @staticmethod
#     def build(model_wrapper):
#         #
#         # "HotFlip ... uses the gradient with respect to a one-hot input
#         # representation to efficiently estimate which individual change has the
#         # highest estimated loss."
#         transformation = WordSwapGradientBased(model_wrapper, top_n=1)
#         #
#         # Don't modify the same word twice or stopwords
#         #
#         constraints = [RepeatModification(), StopwordModification()]
#         #
#         # 0. "We were able to create only 41 examples (2% of the correctly-
#         # classified instances of the SST test set) with one or two flips."
#         #
#         constraints.append(MaxWordsPerturbed(max_num_words=2))
#         #
#         # 1. "The cosine similarity between the embedding of words is bigger than a
#         #   threshold (0.8)."
#         #
#         constraints.append(WordEmbeddingDistance(min_cos_sim=0.8))
#         #
#         # 2. "The two words have the same part-of-speech."
#         #
#         constraints.append(PartOfSpeech())
#         #
#         # Goal is untargeted classification
#         #
#         goal_function = UntargetedClassification(model_wrapper)
#         #
#         # "HotFlip ... uses a beam search to find a set of manipulations that work
#         # well together to confuse a classifier ... The adversary uses a beam size
#         # of 10."
#         #
#         search_method = BeamSearch(beam_width=10)
        

#         return Attack(goal_function, constraints, transformation, search_method)



class HotFlipEbrahimi2017(AttackRecipe):
    """Ebrahimi, J. et al. (2017)

    HotFlip: White-Box Adversarial Examples for Text Classification

    https://arxiv.org/abs/1712.06751

    This is a reproduction of the HotFlip word-level attack (section 5 of the
    paper).
    """

    @staticmethod
    def build(model_wrapper,max_modification_rate=None,cos_sim=0.5,sem_sim=0.8,no_cand=50):
        #
        # "HotFlip ... uses the gradient with respect to a one-hot input
        # representation to efficiently estimate which individual change has the
        # highest estimated loss."
        transformation = WordSwapGradientBased(model_wrapper, top_n=1)
        #
        # Don't modify the same word twice or stopwords
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "<SPLIT>"]
        )


        # stopwords=None
        # fmt: on
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]#,MinWordLength(min_length = 2)]
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        if max_modification_rate :
            # constraints.append(MaxNumWordsModified(max_num_words,min_num_words))

            constraints.append(MaxWordsPerturbed(max_percent=max_modification_rate))
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )


        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["question1", "question2"], {"question1"}
        )
        constraints.append(input_column_modification)

        input_column_modification = InputColumnModification(
            ["sentence1", "sentence2"], {"sentence1"}
        )
        constraints.append(input_column_modification)

        #
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))

        threshold = 1 - ((sem_sim)/math.pi)
        use_constraint = UniversalSentenceEncoder(
            threshold=threshold,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper,query_budget = no_cand)
        #
        # "HotFlip ... uses a beam search to find a set of manipulations that work
        # well together to confuse a classifier ... The adversary uses a beam size
        # of 10."
        #
        # search_method = BeamSearch(beam_width=10)
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)
