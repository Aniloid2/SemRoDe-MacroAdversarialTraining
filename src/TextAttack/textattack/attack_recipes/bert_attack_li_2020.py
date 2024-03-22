"""
BERT-Attack:
============================================================

(BERT-Attack: Adversarial Attack Against BERT Using BERT)

.. warning::
    This attack is super slow
    (see https://github.com/QData/TextAttack/issues/586)
    Consider using smaller values for "max_candidates".

"""
from textattack import Attack
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    InputColumnModification,
)
import math
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapMaskedLM

from .attack_recipe import AttackRecipe


class BERTAttackLi2020(AttackRecipe):
    """Li, L.., Ma, R., Guo, Q., Xiangyang, X., Xipeng, Q. (2020).

    BERT-ATTACK: Adversarial Attack Against BERT Using BERT

    https://arxiv.org/abs/2004.09984

    This is "attack mode" 1 from the paper, BAE-R, word replacement.
    """

    @staticmethod
    def build(model_wrapper,max_modification_rate=None,sem_sim=0.5,no_cand=50):
        # [from correspondence with the author]
        # Candidate size K is set to 48 for all data-sets.
        transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=no_cand)
        #
        # Don't modify the same word twice or stopwords.
        #
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "<SPLIT>"]
        )

        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]

        # "We only take ε percent of the most important words since we tend to keep
        # perturbations minimum."
        #
        # [from correspondence with the author]
        # "Word percentage allowed to change is set to 0.4 for most data-sets, this
        # parameter is trivial since most attacks only need a few changes. This
        # epsilon is only used to avoid too much queries on those very hard samples."
        if max_modification_rate:
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
        # "As used in TextFooler (Jin et al., 2019), we also use Universal Sentence
        # Encoder (Cer et al., 2018) to measure the semantic consistency between the
        # adversarial sample and the original sequence. To balance between semantic
        # preservation and attack success rate, we set up a threshold of semantic
        # similarity score to filter the less similar examples."
        #
        # [from correspondence with author]
        # "Over the full texts, after generating all the adversarial samples, we filter
        # out low USE score samples. Thus the success rate is lower but the USE score
        # can be higher. (actually USE score is not a golden metric, so we simply
        # measure the USE score over the final texts for a comparison with TextFooler).
        # For datasets like IMDB, we set a higher threshold between 0.4-0.7; for
        # datasets like MNLI, we set threshold between 0-0.2."
        #
        # Since the threshold in the real world can't be determined from the training
        # data, the TextAttack implementation uses a fixed threshold - determined to
        # be 0.2 to be most fair.
        threshold = 1 - ((sem_sim)/math.pi)
        use_constraint = UniversalSentenceEncoder(
            threshold=threshold, # used to be 0.2
            metric="cosine",
            compare_against_original=True,
            window_size=None,
        )
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification.
        #
        goal_function = UntargetedClassification(model_wrapper,query_budget = no_cand )
        #
        # "We first select the words in the sequence which have a high significance
        # influence on the final output logit. Let S = [w0, ··· , wi ··· ] denote
        # the input sentence, and oy(S) denote the logit output by the target model
        # for correct label y, the importance score Iwi is defined as
        # Iwi = oy(S) − oy(S\wi), where S\wi = [w0, ··· , wi−1, [MASK], wi+1, ···]
        # is the sentence after replacing wi with [MASK]. Then we rank all the words
        # according to the ranking score Iwi in descending order to create word list
        # L."
        search_method = GreedyWordSwapWIR(wir_method="unk")

        return Attack(goal_function, constraints, transformation, search_method)
