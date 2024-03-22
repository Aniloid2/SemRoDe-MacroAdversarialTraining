"""

Max Modification Rate
-----------------------------

"""

from textattack.constraints import PreTransformationConstraint


class MaxNumWordsModified(PreTransformationConstraint):
    def __init__(self, max_num_words: int,min_num_words: int):
        self.max_num_words = max_num_words
        self.min_num_words = min_num_words

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in current_text which are able to be
        modified."""

        # print ('curr text len',current_text.attack_attrs["modified_indices"], len(current_text.attack_attrs["modified_indices"]))

        if len(current_text.attack_attrs["modified_indices"]) >= self.max_num_words:
            # print ('return emtpy set since we are above the threshold')
            return set()
        else:
            # print ('return the following',current_text.words,set(range(len(current_text.words))) )
            return set(range(len(current_text.words)))

        # print ('max min',self.max_num_words, self.min_num_words, len(current_text.attack_attrs["modified_indices"]), current_text.attack_attrs["modified_indices"]  )
        # if (len(current_text.attack_attrs["modified_indices"]) > self.max_num_words) or (len(current_text.attack_attrs["modified_indices"]) < self.min_num_words):
        #     return set()
        # else:
        #     return set(range(len(current_text.words)))


        # print ('max min',self.max_num_words, self.min_num_words, len(current_text.attack_attrs["modified_indices"]), current_text.attack_attrs["modified_indices"]  )
        # if (len(current_text.attack_attrs["modified_indices"]) >= self.max_num_words):
        #     return set()
        # else:
        #     return set(range(len(current_text.words)))


    def extra_repr_keys(self):
        return ["max_num_words"]
