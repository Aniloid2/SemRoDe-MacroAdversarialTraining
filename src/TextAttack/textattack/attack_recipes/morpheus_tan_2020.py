"""
MORPHEUS2020
===============
(It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations)


"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    InputColumnModification,
    
)
from textattack.goal_functions import MinimizeBleu
from textattack.search_methods import GreedySearch
from textattack.transformations import WordSwapInflections
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
import math
from .attack_recipe import AttackRecipe


class MorpheusTan2020(AttackRecipe):
    """Samson Tan, Shafiq Joty, Min-Yen Kan, Richard Socher.

    It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations

    https://www.aclweb.org/anthology/2020.acl-main.263/
    """

    @staticmethod
    def build(model_wrapper,max_modification_rate=None,cos_sim=0.5,sem_sim=0.8,no_cand=50):
        # we change everything except the word swap inflection technique
        transformation = WordSwapInflections()
        
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "<SPLIT>"]
        )

        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]

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

        # In our experiment, we first use the Universal Sentence
        # Encoder [7], a model trained on a number of natural language
        # prediction tasks that require modeling the meaning of word
        # sequences, to encode sentences into high dimensional vectors.
        # Then, we use the cosine similarity to measure the semantic
        # similarity between original texts and adversarial texts.
        # ... "Furthermore, the semantic similarity threshold \eps is set
        # as 0.8 to guarantee a good trade-off between quality and
        # strength of the generated adversarial text."
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
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")
        # search_method = GreedySearch()
        return Attack(goal_function, constraints, transformation, search_method)



# class MorpheusTan2020(AttackRecipe):
#     """Samson Tan, Shafiq Joty, Min-Yen Kan, Richard Socher.

#     It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations

#     https://www.aclweb.org/anthology/2020.acl-main.263/
#     """

#     @staticmethod
#     def build(model_wrapper ):

#         #
#         # Goal is to minimize BLEU score between the model output given for the
#         # perturbed input sequence and the reference translation
#         #
#         goal_function = MinimizeBleu(model_wrapper)

#         # Swap words with their inflections
#         transformation = WordSwapInflections()

#         #
#         # Don't modify the same word twice or stopwords
#         #
#         constraints = [RepeatModification(), StopwordModification()]

#         #
#         # Greedily swap words (see pseudocode, Algorithm 1 of the paper).
#         #
#         search_method = GreedySearch()

#         return Attack(goal_function, constraints, transformation, search_method)
