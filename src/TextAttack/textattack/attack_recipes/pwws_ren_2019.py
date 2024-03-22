"""

PWWS
=======

(Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency)

"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    InputColumnModification,
    MaxModificationRate,
)
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR,GreedyWordSwapWIRPWWS
from textattack.transformations import WordSwapWordNet
import math
from textattack.transformations import WordSwapEmbedding
from .attack_recipe import AttackRecipe
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder

class PWWSRen2019(AttackRecipe):
    """An implementation of Probability Weighted Word Saliency from "Generating
    Natural Language Adversarial Examples through Probability Weighted Word
    Saliency", Ren et al., 2019.

    Words are prioritized for a synonym-swap transformation based on
    a combination of their saliency score and maximum word-swap effectiveness.
    Note that this implementation does not include the Named
    Entity adversarial swap from the original paper, because it requires
    access to the full dataset and ground truth labels in advance.

    https://www.aclweb.org/anthology/P19-1103/
    """

    @staticmethod
    def build(model_wrapper,max_modification_rate=None,sem_sim=0.5,no_cand = 50):
        transformation = WordSwapWordNet()



        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "<SPLIT>"]
        )
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]


        if max_modification_rate:
             # input_column_modification = MaxModificationRate(max_modification_rate)
             input_column_modification =  MaxWordsPerturbed(max_percent=max_modification_rate)
             constraints.append(input_column_modification)

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

        threshold = 1 - ((sem_sim)/math.pi)
        use_constraint = UniversalSentenceEncoder(
            threshold=threshold,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)


        goal_function = UntargetedClassification(model_wrapper,query_budget = no_cand)
        # search over words based on a combination of their saliency score, and how efficient the WordSwap transform is
        search_method = GreedyWordSwapWIR("weighted-saliency")
        # search_method = GreedyWordSwapWIRPWWS(transformation)

        # transformation = WordSwapEmbedding(max_candidates=no_cand)

        return Attack(goal_function, constraints, transformation, search_method)
