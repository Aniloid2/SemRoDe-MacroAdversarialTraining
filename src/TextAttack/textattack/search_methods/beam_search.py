"""
Beam Search
===============

"""
import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod


class BeamSearch(SearchMethod):
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

    # def _get_index_order(self, initial_text):
    #     # gradient lookup ranking
    #     len_text, indices_to_order = self.get_indices_to_order(initial_text)

    #     victim_model = self.get_victim_model()
    #     index_scores = np.zeros(len_text)
    #     grad_output = victim_model.get_grad(initial_text.tokenizer_input)
    #     gradient = grad_output["gradient"]
    #     word2token_mapping = initial_text.align_with_model_tokens(victim_model)
    #     for i, index in enumerate(indices_to_order):
    #         matched_tokens = word2token_mapping[index]
    #         if not matched_tokens:
    #             index_scores[i] = 0.0
    #         else:
    #             agg_grad = np.mean(gradient[matched_tokens], axis=0)
    #             index_scores[i] = np.linalg.norm(agg_grad, ord=1)

    #     search_over = False
    #     index_order = np.array(indices_to_order)[(-index_scores).argsort()]
    #     return index_order, search_over
    
    def perform_search(self, initial_result):
        beam = [initial_result.attacked_text]
        best_result = initial_result
        # print ('beam',beam)
        # index_order, search_over = self._get_index_order(attacked_text)
        counter = 0
        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            for text in beam:
                # print ('text',text)
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                # print ('transformations',len(transformations))
                for i,tran in enumerate(transformations):
                    print (i,tran)
                    
                
                counter+=1
                potential_next_beam += transformations 
                # print ('potential next beams',len(potential_next_beam))
                
                # if counter == 3: 
                #     sys.exit()
            if len(potential_next_beam) == 0:
                # If we did not find any possible perturbations, give up.
                return best_result
            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            best_result = results[scores.argmax()]
            if search_over:
                return best_result

            # Refill the beam. This works by sorting the scores
            # in descending order and filling the beam from there.
            best_indices = (-scores).argsort()[: self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]

        return best_result

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]
