"""

Stopword Modification
--------------------------

"""

import nltk

from textattack.constraints import PreTransformationConstraint
from textattack.shared.validators import transformation_consists_of_word_swaps


class TokenizerAlignment(PreTransformationConstraint):
    """A constraint disallowing the modification of stopwords."""

    def __init__(self,tokenizer):
        # print ('stopwords',stopwords)
        # if stopwords is not None:
        #     self.stopwords = set(stopwords)
        # else:
        #     self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.tokenizer = tokenizer



    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        # print ('current text',current_text.text)
        tokens = self.tokenizer.tokenize(current_text.text)
        # print ('tokens',tokens)
        # print ('current words',current_text.words)
        # sys.exit()
        aligned_indices = set()
        for i, word in enumerate(current_text.words):
            if word in tokens:
                aligned_indices.add(i)
        # print ('aligned_indices',aligned_indices) 
        return aligned_indices

    def check_compatibility(self, transformation):
        """The stopword constraint only is concerned with word swaps since
        paraphrasing phrases containing stopwords is OK.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        return transformation_consists_of_word_swaps(transformation)
