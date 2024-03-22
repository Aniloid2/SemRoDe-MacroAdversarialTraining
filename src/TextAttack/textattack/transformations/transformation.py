"""
Transformation Abstract Class
============================================

"""

from abc import ABC, abstractmethod

from textattack.shared.utils import ReprMixin


class Transformation(ReprMixin, ABC):
    """An abstract class for transforming a sequence of text to produce a
    potential adversarial example."""

    def __call__(
        self,
        current_text,
        pre_transformation_constraints=[],
        indices_to_modify=None,
        shifted_idxs=False,
        return_indices=False,
        **kwargs
    ):
        """Returns a list of all possible transformations for ``current_text``.
        Applies the ``pre_transformation_constraints`` then calls
        ``_get_transformations``.

        Args:
            current_text: The ``AttackedText`` to transform.
            pre_transformation_constraints: The ``PreTransformationConstraint`` to apply before
                beginning the transformation.
            indices_to_modify: Which word indices should be modified as dictated by the
                ``SearchMethod``.
            shifted_idxs (bool): Whether indices could have been shifted from
                their original position in the text.
            return_indices (bool): Whether the function returns indices_to_modify
                instead of the transformed_texts.
        """
        # print ('indices_to_modify',indices_to_modify)
        # print ('current text',current_text)
        # batch_size = 2
        if isinstance(current_text,list):
            indices_to_modify_sets = []
            for b,indices in enumerate(indices_to_modify):
                for constraint in pre_transformation_constraints:
                    indices_to_modify = set([indices]) & constraint(current_text[b], self)
                indices_to_modify_sets.append(indices_to_modify)
                if return_indices:
                    print ('returing indices?')
                    ValueError('returning indices?')
                    return indices_to_modify


            
            transformed_texts,unsuccesful_mask,synonyms = self._get_transformations(current_text, indices_to_modify_sets, **kwargs)
            
            
            for b in range(len(transformed_texts)): 
                for transform in transformed_texts[b]:
                    transform.attack_attrs["last_transformation"] = self
                
            return transformed_texts,unsuccesful_mask,synonyms
        else:

            if indices_to_modify is None:
                indices_to_modify = set(range(len(current_text.words)))
                # If we are modifying all indices, we don't care if some of the indices might have been shifted.
                shifted_idxs = False
            else:
                indices_to_modify = set(indices_to_modify)

            if shifted_idxs:
                indices_to_modify = set(
                    current_text.convert_from_original_idxs(indices_to_modify)
                ) 
            for constraint in pre_transformation_constraints:
                indices_to_modify = indices_to_modify & constraint(current_text, self)

            if return_indices:
                return indices_to_modify
            
            # print ('kwargs',kwargs['gradient'])
            
            transformed_texts = self._get_transformations(current_text, indices_to_modify, **kwargs)
            # print ('transformed_texts',transformed_texts)
            
            if isinstance(transformed_texts,dict):
                temp_transformed_text = transformed_texts['transformed_texts']
                for text in temp_transformed_text:
                    text.attack_attrs["last_transformation"] = self
                transformed_texts['transformed_texts'] = temp_transformed_text
                return transformed_texts
            else:
                for text in transformed_texts:
                    text.attack_attrs["last_transformation"] = self
                return transformed_texts

    @abstractmethod
    def _get_transformations(self, current_text, indices_to_modify):
        """Returns a list of all possible transformations for ``current_text``,
        only modifying ``indices_to_modify``. Must be overridden by specific
        transformations.

        Args:
            current_text: The ``AttackedText`` to transform.
            indicies_to_modify: Which word indices can be modified.
        """
        raise NotImplementedError()

    @property
    def deterministic(self):
        return True
