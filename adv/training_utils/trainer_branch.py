"""
Trainer Class
=============
"""

import collections
import json
import logging
import math
import os
import copy
import scipy
import torch
import tqdm
import transformers
import nvidia_smi
import textattack
from textattack.shared.utils import logger
from pathlib import Path
from textattack.attack import Attack
from textattack.attack_args import AttackArgs
from textattack.attack_results import MaximizedAttackResult, SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult
from textattack.attacker import Attacker
from textattack.model_args import HUGGINGFACE_MODELS
from textattack.models.helpers import LSTMForClassification, WordCNNForClassification
from textattack.models.wrappers import ModelWrapper
from textattack.training_args import CommandLineTrainingArgs
from .training_args import  TrainingArgs
import training_utils.training_utils_functions as training_utils_functions
import torch.nn.functional as F
import itertools
import ot
from geomloss import SamplesLoss
import warnings
from datasets import load_dataset, Dataset
import sys
import higher
import time
 
current_file_path = os.path.dirname(os.path.abspath(__file__)) 
parent_path =  os.path.dirname(os.path.dirname(current_file_path)) 
sys.path.append(f'{parent_path}/src/TextDefender') 
# from src.TextDefender.trainer.ascc import ASCCTrainer
from trainer.ascc import ASCCTrainer
from utils.ascc_utils import WarmupMultiStepLR
# from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """Trainer is training and eval loop for adversarial training.

    It is designed to work with PyTorch and Transformers models.

    Args:
        model_wrapper (:class:`~textattack.models.wrappers.ModelWrapper`):
            Model wrapper containing both the model and the tokenizer.
        task_type (:obj:`str`, `optional`, defaults to :obj:`"classification"`):
            The task that the model is trained to perform.
            Currently, :class:`~textattack.Trainer` supports two tasks: (1) :obj:`"classification"`, (2) :obj:`"regression"`.
        attack (:class:`~textattack.Attack`):
            :class:`~textattack.Attack` used to generate adversarial examples for training.
        train_dataset (:class:`~textattack.datasets.Dataset`):
            Dataset for training.
        eval_dataset (:class:`~textattack.datasets.Dataset`):
            Dataset for evaluation
        training_args (:class:`~textattack.TrainingArgs`):
            Arguments for training.

    Example::

        >>> import textattack
        >>> import transformers

        >>> model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        >>> tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

        >>> # We only use DeepWordBugGao2018 to demonstration purposes.
        >>> attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
        >>> train_dataset = textattack.datasets.HuggingFaceDataset("imdb", split="train")
        >>> eval_dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

        >>> # Train for 3 epochs with 1 initial clean epochs, 1000 adversarial examples per epoch, learning rate of 5e-5, and effective batch size of 32 (8x4).
        >>> training_args = textattack.TrainingArgs(
        ...     num_epochs=3,
        ...     num_clean_epochs=1,
        ...     num_train_adv_examples=1000,
        ...     learning_rate=5e-5,
        ...     per_device_train_batch_size=8,
        ...     gradient_accumulation_steps=4,
        ...     log_to_tb=True,
        ... )

        >>> trainer = textattack.Trainer(
        ...     model_wrapper,
        ...     "classification",
        ...     attack,
        ...     train_dataset,
        ...     eval_dataset,
        ...     training_args
        ... )
        >>> trainer.train()

    .. note::
        When using :class:`~textattack.Trainer` with `parallel=True` in :class:`~textattack.TrainingArgs`,
        make sure to protect the “entry point” of the program by using :obj:`if __name__ == '__main__':`.
        If not, each worker process used for generating adversarial examples will execute the training code again.
    """

    def __init__(
        self,
        model_wrapper,
        task_type="classification",
        attack=None,
        train_dataset=None,
        eval_dataset=None,
        training_args=None,
    ):
        assert isinstance(
            model_wrapper, ModelWrapper
        ), f"`model_wrapper` must be of type `textattack.models.wrappers.ModelWrapper`, but got type `{type(model_wrapper)}`."

        # TODO: Support seq2seq training
        assert task_type in {
            "classification",
            "regression",
            "optimal_transport",
            "optimal_transport_geomloss",
            "OT_GL",
            "OT_GL_CC",
            "MMD",
        }, '`task_type` must either be "classification","optimal_transport", "regression" or "OT_GL"'

        if attack:
            assert isinstance(
                attack, Attack
            ), f"`attack` argument must be of type `textattack.Attack`, but got type of `{type(attack)}`."

            if id(model_wrapper) != id(attack.goal_function.model):
                logger.warn(
                    "`model_wrapper` and the victim model of `attack` are not the same model."
                )

        

        if train_dataset:
            assert isinstance(
                train_dataset, textattack.datasets.Dataset
            ), f"`train_dataset` must be of type `textattack.datasets.Dataset`, but got type `{type(train_dataset)}`."

        if eval_dataset:
            assert isinstance(
                eval_dataset, textattack.datasets.Dataset
            ), f"`eval_dataset` must be of type `textattack.datasets.Dataset`, but got type `{type(eval_dataset)}`."

        if training_args:
            assert isinstance(
                training_args, TrainingArgs
            ), f"`training_args` must be of type `textattack.TrainingArgs`, but got type `{type(training_args)}`."
        else:
            training_args = TrainingArgs()

        if not hasattr(model_wrapper, "model"):
            raise ValueError("Cannot detect `model` in `model_wrapper`")
        else:
            assert isinstance(
                model_wrapper.model, torch.nn.Module
            ), f"`model` in `model_wrapper` must be of type `torch.nn.Module`, but got type `{type(model_wrapper.model)}`."

        if not hasattr(model_wrapper, "tokenizer"):
            raise ValueError("Cannot detect `tokenizer` in `model_wrapper`")

        self.model_wrapper = model_wrapper
        self.task_type = task_type
        self.attack = attack
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args

        if self.training_args.Online_AT_Val is not None:
            self._setup_online_attacker()

        self._metric_name = (
            "pearson_correlation" if self.task_type == "regression" else "accuracy"
        )
        if self.task_type == "regression":
            self.loss_fct = torch.nn.MSELoss(reduction="none")
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        self._global_step = 0
        self._global_step_eval = 0

        
                
                
        

    def _setup_online_attacker(self):
        self.copy_train_dataset = copy.deepcopy(self.train_dataset)

    def _get_online_samples(self, batch):
        'Need a way to set up model on gpu for online adversarial attacks'
        dataset_to_attack = batch
        new_dataset_list = []
        for i in range(len(dataset_to_attack[0])):
            element = {'text':dataset_to_attack[0][i],'label':dataset_to_attack[1][i].tolist()}
            new_dataset_list.append(element) 
        
         
        # print ('dataset to attack',dataset_to_attack)
        dataset_to_attack = new_dataset_list
        
        new_dataset = Dataset.from_dict({"text": [sample["text"] for sample in new_dataset_list], "label": [sample["label"] for sample in new_dataset_list]})
 
        self.copy_train_dataset._dataset = new_dataset


        dataset_to_attack = self.copy_train_dataset

        # sys.exit()
        attack_args = AttackArgs(
            num_successful_examples=self.training_args.per_device_train_batch_size,
            num_examples_offset=0,
            query_budget=self.training_args.query_budget_train,
            shuffle=True,
            parallel= False, # True,#self.training_args.parallel,
            num_workers_per_device= 1, #8,#self.training_args.attack_num_workers_per_device,
            disable_stdout=True,
            silent=True,                  
        )

        attacker = Attacker(self.attack, dataset_to_attack, attack_args=attack_args)
        results = attacker.attack_dataset() 

        attack_types = collections.Counter(r.__class__.__name__ for r in results)
        total_attacks = (
            attack_types["SuccessfulAttackResult"] + attack_types["FailedAttackResult"]
        )
        success_rate = attack_types["SuccessfulAttackResult"] / total_attacks * 100
        logger.info(f"Total number of attack results: {len(results)}")
        logger.info(
            f"Attack success rate: {success_rate:.2f}% [{attack_types['SuccessfulAttackResult']} / {total_attacks}]"
        )

        # print ('results',results)
        # sys.exit()

        # adversarial_examples = [
        #     (
        #         tuple(r.perturbed_result.attacked_text._text_input.values())
        #         + ("adversarial_example",),
        #         r.perturbed_result.ground_truth_output,
        #     )
        #     for r in results
        #     if isinstance(r, (FailedAttackResult, SkippedAttackResult))
        # ]

        # results = [
        #     r
        #     for r in results
        #     if isinstance(r, (FailedAttackResult, SkippedAttackResult))
        # ] 
        adversarial_examples = [
            [
                list(r.perturbed_result.attacked_text._text_input.values())[0],
                r.perturbed_result.ground_truth_output,
                True,
        ]
            for r in results
            if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult))
        ]
 
        adv_samples = [item[0] for item in adversarial_examples]
        adv_samples_label = [item[1] for item in adversarial_examples]
        adv_samples_is_adv = [item[2] for item in adversarial_examples]
        

        return [adv_samples, torch.tensor(adv_samples_label),torch.tensor(adv_samples_is_adv)]

        # # Name for column indicating if an example is adversarial is set as "_example_type".
        # adversarial_dataset = textattack.datasets.Dataset(
        #     adversarial_examples,
        #     input_columns=self.train_dataset.input_columns + ("_example_type",),
        #     label_map=self.train_dataset.label_map,
        #     label_names=self.train_dataset.label_names,
        #     output_scale_factor=self.train_dataset.output_scale_factor,
        #     shuffle=False,
        # )

        return adversarial_dataset
        
 
        # raise NotImplementedError()
    

        
    # we concat the adv samples, we need them separated
    def _generate_adversarial_examples(self, epoch):
        """Generate adversarial examples using attacker."""
        assert (
            self.attack is not None
        ), "`attack` is `None` but attempting to generate adversarial examples."
        base_file_name = f"attack-train-{epoch}"
        log_file_name = os.path.join(self.training_args.output_dir, base_file_name)
        logger.info("Attacking model to generate new adversarial training set...")

        if isinstance(self.training_args.num_train_adv_examples, float):
            # num_train_adv_examples = math.ceil(
            #     len(self.train_dataset) * self.training_args.num_train_adv_examples
            # )

            # tmp_dataset = self.train_dataset._dataset
            # print ('type dataset',type(self.train_dataset))

            copy_train_dataset = copy.deepcopy(self.train_dataset)

            # our method to partition train dataset that is available to attack
            # if self.training_args.num_train_adv_examples == 100% we don't need to use the tain_test_split trick
            # to partition this dataset as we attempt to generate our adv samples through all of the training dataset
            # what this helps us do is avoid the situation where if lets say we want to only have 10% of the training
            # dataset as adv samples, if we have a dataset of 1000 this will be 100 samples
            # the adv generation algo will samples exacly 100 adv samples, this will however
            # likely require to perturb 15% of the dataset if the attack algorithm achives a 25% after attack accuracy
            # as the model becomes more robust e.g AAA goes from 25% to 50 % then in the following epochs if we do online training
            # we will have to perturb 20% of the dataset, this in practice may be good as we progressively open up more samples
            # for the algo to learn, however it makes it hard to benchmark as multiple runs will have different
            # dataset. our solution to this is to just show the attack algorithm 10% of the dataset at all times
            # therefore never allowing it to access extra dataset.

            if 1-self.training_args.num_train_adv_examples == 0.0:
                pass
            else:
                copy_train_dataset._dataset = copy_train_dataset._dataset.train_test_split(test_size=1-self.training_args.num_train_adv_examples , shuffle=False)['train']


            # print('len copy train',len(copy_train_dataset))
            #
            # print ('dataset after reinitialization',len(self.train_dataset))
            #
            # print ('len dataset2',len(copy_train_dataset))
            # print ('type dataset copy',type(copy_train_dataset))
            dataset_to_attack = copy_train_dataset
            # print ('dataset to attack',dataset_to_attack)
            num_train_adv_examples = math.ceil(
                len(copy_train_dataset) * 1.0
            )

        else:
            num_train_adv_examples = self.training_args.num_train_adv_examples
            dataset_to_attack = self.train_dataset

        # Use Different AttackArgs based on num_train_adv_examples value.
        # If num_train_adv_examples >= 0 , num_train_adv_examples is
        # set as number of successful examples.
        # If num_train_adv_examples == -1 , num_examples is set to -1 to
        # generate example for all of training data.

        # self.train_dataset function that only returns num_train_adv_examples as a new dataset
        # we then pass this dataset

        # print ('size_to attacks',num_train_adv_examples,dataset_to_attack)



        if num_train_adv_examples >= 0:
            #NEEDCHANGE
            if self.training_args.AttackTrain == 'BERTAttack' or self.training_args.AttackTrain == 'TextBugger'  or self.training_args.AttackTrain == 'A2T' :
                print ('cannot have parallel with bertattack, since we need 2 models interacting with one another')
                
                attack_args = AttackArgs(
                num_successful_examples=num_train_adv_examples,
                num_examples_offset=0,
                query_budget=self.training_args.query_budget_train,
                shuffle=True,
                parallel= False, # True,#self.training_args.parallel,
                num_workers_per_device= 1, #8,#self.training_args.attack_num_workers_per_device,
                disable_stdout=True,
                silent=True,
                log_to_txt=log_file_name + ".txt", 
                )
            else:
                attack_args = AttackArgs(
                    num_successful_examples=num_train_adv_examples,
                    num_examples_offset=0,
                    query_budget=self.training_args.query_budget_train,
                    shuffle=True,
                    parallel= True, # True,#self.training_args.parallel,
                    num_workers_per_device= 8, #8,#self.training_args.attack_num_workers_per_device,
                    disable_stdout=True,
                    silent=True,
                    log_to_txt=log_file_name + ".txt", 
                )
            # attack_args = AttackArgs(
            #     num_successful_examples=num_train_adv_examples,
            #     num_examples_offset=0,
            #     query_budget=self.training_args.query_budget_train,
            #     shuffle=True,
            #     parallel= False, #self.training_args.parallel,
            #     num_workers_per_device= 1, #self.training_args.attack_num_workers_per_device,
            #     disable_stdout=False,
            #     silent=False,
            #     log_to_txt=log_file_name + ".txt", # log_to_csv=log_file_name + ".csv",
            # )
        elif num_train_adv_examples == -1:
            # set num_examples when num_train_adv_examples = -1
            attack_args = AttackArgs(
                num_examples=num_train_adv_examples,
                num_examples_offset=0,
                query_budget=self.training_args.query_budget_train,
                query_budget_size=self.training_args.query_budget_train,
                shuffle=True,
                parallel=self.training_args.parallel,
                num_workers_per_device=self.training_args.attack_num_workers_per_device,
                disable_stdout=True,
                silent=True,
                log_to_txt=log_file_name + ".txt",
                # log_to_csv=log_file_name + ".csv",
            )
        else:
            assert False, "num_train_adv_examples is negative and not equal to -1."


        attacker = Attacker(self.attack, dataset_to_attack, attack_args=attack_args)
        results = attacker.attack_dataset()

        attack_types = collections.Counter(r.__class__.__name__ for r in results)
        total_attacks = (
            attack_types["SuccessfulAttackResult"] + attack_types["FailedAttackResult"]
        )
        success_rate = attack_types["SuccessfulAttackResult"] / total_attacks * 100
        logger.info(f"Total number of attack results: {len(results)}")
        logger.info(
            f"Attack success rate: {success_rate:.2f}% [{attack_types['SuccessfulAttackResult']} / {total_attacks}]"
        )
        # TODO: This will produce a bug if we need to manipulate ground truth output.

        # To Fix Issue #498 , We need to add the Non Output columns in one tuple to represent input columns
        # Since adversarial_example won't be an input to the model , we will have to remove it from the input
        # dictionary in collate_fn


        # for r in results:
        #     print ('original sample',r)
        #     print ('perturbed text',r.perturbed_result.attacked_text._text_input.values())
        #     print ('ground trouth',r.perturbed_result.ground_truth_output)
        #
        # sys.exit()

        adversarial_examples = [
            (
                tuple(r.perturbed_result.attacked_text._text_input.values())
                + ("adversarial_example",),
                r.perturbed_result.ground_truth_output,
            )
            for r in results
            if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult))
        ]

        # Name for column indicating if an example is adversarial is set as "_example_type".
        adversarial_dataset = textattack.datasets.Dataset(
            adversarial_examples,
            input_columns=self.train_dataset.input_columns + ("_example_type",),
            label_map=self.train_dataset.label_map,
            label_names=self.train_dataset.label_names,
            output_scale_factor=self.train_dataset.output_scale_factor,
            shuffle=False,
        )


        extended_results_concat = []
        extended_extra_results_concat = []

        # for i,r in enumerate(results):
        #     if isinstance(r, (SuccessfulAttackResult)):
        #         print (i,'successful')
        #         # print (i,r.perturbed_result)
        #         continue
        #     else:
        #
        #     print ('------BEGIN----------')
        #     print ('result:',i,r.perturbed_result)
        #     # print ('modified indixes',len(r.perturbed_result.attacked_text.attack_attrs['modified_indices']))
        #     number_of_mod = len(r.perturbed_result.attacked_text.attack_attrs['modified_indices'])
        #     curr_end_sample = r.perturbed_result.attacked_text
        #     while curr_end_sample.attack_attrs.get('prev_attacked_text'):
        #         print ('current sample:',curr_end_sample)
        #         # print (curr_end_sample.attack_attrs)
        #         # , 'model out:', curr_end_sample.model_output, 'ground trouth:', curr_end_sample.ground_truth_output, 'score:' , curr_end_sample.score )
        #         next_sample = curr_end_sample.attack_attrs['prev_attacked_text']
        #         # extended_results.append((tuple(next_sample._text_input.values())+ ("progressive_example",),r.perturbed_result.ground_truth_output ))
        #         curr_end_sample = next_sample
        #     print ('------END----------')

        # self.training_args.PF = True
        for i,r in enumerate(results):
            # if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult)):
            # print ('------START------')
            total_words = r.original_result.attacked_text.all_words_diff(
                    r.perturbed_result.attacked_text
                )
            num_words_changed = len(total_words)
            # print ('num changed',num_words_changed)
            # if num_words_changed < 3 or num_words_changed >= 4:
            #     continue

            number_of_mod = len(r.perturbed_result.attacked_text.attack_attrs['modified_indices'])
            # print ('indicies',r.perturbed_result.attacked_text.attack_attrs)
            # print ('perturbed res',r.perturbed_result)
            curr_end_sample = r.perturbed_result.attacked_text
            # extended_results = [(tuple(r.perturbed_result.attacked_text._text_input.values())+ ("adversarial_example",),r.perturbed_result.ground_truth_output)]
            # empty extended results because
            extended_results = []

            extended_extra_results = []


            # print ('this should be the final samples that is adv',extended_results)

            # print ('success instance?',isinstance(r, SuccessfulAttackResult))
            # print ('maxed instance?',isinstance(r,  MaximizedAttackResult))
            while curr_end_sample.attack_attrs.get('prev_attacked_text'):
                # print ('current sample:',curr_end_sample)
                next_sample = curr_end_sample.attack_attrs['prev_attacked_text']

                if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult)):
                    extended_results.append((tuple(next_sample._text_input.values())+ ("progressive_example",),r.perturbed_result.ground_truth_output ))
                else:
                    extended_extra_results.append((tuple(next_sample._text_input.values())+ ("progressive_example",),r.perturbed_result.ground_truth_output ))

                # if self.training_args.PF_Val: # append in different list. return both sep
                #     extended_results.append((tuple(next_sample._text_input.values())+ ("progressive_example",),r.perturbed_result.ground_truth_output ))
                # else:
                #     if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult)):
                #         extended_results.append((tuple(next_sample._text_input.values())+ ("progressive_example",),r.perturbed_result.ground_truth_output ))
                #
                #     else:
                #         break


                curr_end_sample = next_sample
            # extended_results.append((tuple(r.original_result.attacked_text._text_input.values())+ ("adversarial_example",),r.perturbed_result.ground_truth_output ))
            # print ('this should be all possible var, the last can be removed, since it should be the orig sample?',extended_results)
            # print ('orig sent',r.original_result.attacked_text,r.original_result)
            # remove last element, since it's the original, not perturbed sample

            # print ('current sample:',curr_end_sample)

            # print ('external before pop',extended_results)
            if len(extended_results) != 0:
                extended_results.pop()

            # print ('ext res and ext',len(extended_results),len(extended_extra_results))
            # print ('external after pop',extended_results)
            # print ('external extra',extended_extra_results)


            # if len(extended_results) == 0:
            #     pass
            # else:
            #     if self.training_args.PF_Val:
            #         if isinstance(r, SuccessfulAttackResult):
            #             extended_results.pop()
            #     else:
            #         extended_results.pop()

            # print ('ex res',extended_results)
            # print ('------END-------')

            for er in extended_results:
                extended_results_concat.append(er)

            for er2 in extended_extra_results:
                extended_extra_results_concat.append(er2)


        extended_results_concat.reverse()
        extended_extra_results_concat.reverse()
        print ('len extended_extra_results_concat',len(extended_results_concat))

        print ('len extended_extra_results_concat',len(extended_extra_results_concat))


        progress_dataset = textattack.datasets.Dataset(
            extended_results_concat,
            input_columns=self.train_dataset.input_columns + ("_example_type",),
            label_map=self.train_dataset.label_map,
            label_names=self.train_dataset.label_names,
            output_scale_factor=self.train_dataset.output_scale_factor,
            shuffle=False,
        )

        progress_extra_dataset = textattack.datasets.Dataset(
            extended_extra_results_concat,
            input_columns=self.train_dataset.input_columns + ("_example_type",),
            label_map=self.train_dataset.label_map,
            label_names=self.train_dataset.label_names,
            output_scale_factor=self.train_dataset.output_scale_factor,
            shuffle=False,
        )


        # self.train_dataset._dataset = tmp_dataset

        return adversarial_dataset,progress_dataset,progress_extra_dataset




    def _generate_adversarial_examples_evaluation(self, epoch):
        """Generate adversarial examples using attacker."""
        assert (
            self.attack is not None
        ), "`attack` is `None` but attempting to generate adversarial examples."
        base_file_name = f"attack-eval-{epoch}"
        log_file_name = os.path.join(self.training_args.output_dir, base_file_name)
        logger.info("Attacking model to generate new adversarial training set...")

        # no need to partition in this case, we just want to have the attacked eval dataset

        # if isinstance(self.training_args.num_train_adv_examples, float):
        #
        #
        #     copy_eval_dataset = copy.deepcopy(self.eval_dataset)
        #     #
        #     # copy_train_dataset._dataset = copy_eval_dataset._dataset.train_test_split(test_size=1-self.training_args.num_train_adv_examples, shuffle=False)['eval']
        #
        #
        #
        #     eval_dataset_to_attack = copy_eval_dataset
        #
        #     num_eval_adv_examples = math.ceil(
        #         len(copy_eval_dataset) * 1.0
        #     )
        # else:

        eval_dataset_to_attack =self.eval_dataset
        num_eval_adv_examples = math.ceil(
                len(self.eval_dataset) * 1.0
            )



        if num_eval_adv_examples >= 0:
            attack_args = AttackArgs(
                num_successful_examples=num_eval_adv_examples,
                num_examples_offset=0,
                query_budget=self.training_args.query_budget_train,
                shuffle=True,
                parallel=self.training_args.parallel,
                num_workers_per_device=self.training_args.attack_num_workers_per_device,
                disable_stdout=True,
                silent=True,
                log_to_txt=log_file_name + ".txt",
                # log_to_csv=log_file_name + ".csv",
            )

        else:
            assert False, "num_eval_adv_examples is negative "


        attacker = Attacker(self.attack, eval_dataset_to_attack, attack_args=attack_args)
        results = attacker.attack_dataset()

        attack_types = collections.Counter(r.__class__.__name__ for r in results)
        total_attacks = (
            attack_types["SuccessfulAttackResult"] + attack_types["FailedAttackResult"]
        )
        success_rate = attack_types["SuccessfulAttackResult"] / total_attacks * 100
        logger.info(f"Total number of attack results: {len(results)}")
        logger.info(
            f"Attack success rate: {success_rate:.2f}% [{attack_types['SuccessfulAttackResult']} / {total_attacks}]"
        )

        eval_adversarial_examples = [
            (
                tuple(r.perturbed_result.attacked_text._text_input.values())
                + ("adversarial_example",),
                r.perturbed_result.ground_truth_output,
            )
            for r in results
            if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult))
        ]

        eval_adversarial_dataset = textattack.datasets.Dataset(
            eval_adversarial_examples,
            input_columns=self.train_dataset.input_columns + ("_example_type",),
            label_map=self.train_dataset.label_map,
            label_names=self.train_dataset.label_names,
            output_scale_factor=self.train_dataset.output_scale_factor,
            shuffle=False,
        )

        return eval_adversarial_dataset


    def _print_training_args(
        self, total_training_steps, train_batch_size, num_clean_epochs
    ):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num epochs = {self.training_args.num_epochs}")
        logger.info(f"  Num clean epochs = {num_clean_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.training_args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {train_batch_size * self.training_args.gradient_accumulation_steps}"
        )
        logger.info(
            f"  Gradient accumulation steps = {self.training_args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {total_training_steps}")

    def _save_model_checkpoint(
        self, model, tokenizer, step=None, epoch=None, best=False, last=False
    ):
        # Save model checkpoint
        if step:
            dir_name = f"checkpoint-step-{step}"
        if epoch:
            dir_name = f"checkpoint-epoch-{epoch}"
        if best:
            dir_name = "best_model"
        if last:
            dir_name = "last_model"

        output_dir = os.path.join(self.training_args.output_dir, dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        if isinstance(model, (WordCNNForClassification, LSTMForClassification)):
            model.save_pretrained(output_dir)
        elif isinstance(model, transformers.PreTrainedModel):
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                state_dict,
                os.path.join(output_dir, "pytorch_model.bin"),
            )

    def _tb_log(self, log, step):
        if not hasattr(self, "_tb_writer"):
            from torch.utils.tensorboard import SummaryWriter
            method_snapshot = self.training_args.Method_Dictionary 
            self._tb_writer = SummaryWriter(self.training_args.tb_log_dir)
            # print ('training args before',self.training_args)
            self.training_args.Method_Dictionary = None# {str(key): str(value) for key, value in self.training_args.Method_Dictionary.items()}
            self._tb_writer.add_hparams(self.training_args.__dict__, {})
            self._tb_writer.flush()
            self.training_args.Method_Dictionary = method_snapshot
            # print ('training args after',self.training_args)

        for key in log:
            self._tb_writer.add_scalar(key, log[key], step)

    def _wandb_log(self, log, step):
        if not hasattr(self, "_wandb_init"):
            global wandb
            import wandb

            self._wandb_init = True
            wandb.init(
                project=self.training_args.wandb_project,
                config=self.training_args.__dict__,
            )

        wandb.log(log, step=step)

    def _csv_log(self, log, step):
        ttl,btl,lr,ce,ot,ace,mmd,mmd_progress,mmd_progress_extra,coral,flb_l  = log["train/total_loss"],log["train/loss"], log["train/learning_rate"],  log["cross_entropy"],log["optimal_transport"], log["adversarial_cross_entropy"],log['mmd'],log['mmd_progress'],log['mmd_progress_extra'],log['coral'],log['flb']
        message = f"TTL:{ttl:<9}, BTL:{btl:<9}, CE:{ce:<9}, OT:{ot:<13}, ACE:{ace:<9}, MMD:{mmd:<9}, MMD Prog:{mmd_progress:<9}, MMD Prog:{mmd_progress_extra:<9} , CORAL:{coral:<9}, FLB:{flb_l:<9}, LR:{lr:<9}\n"
        # message = f"TTL:{ttl:<3}, BTL:{btl:<3}, CE:{ce:<3}, OT:{ot:<13}, ACE:{ace:<3}, MMD:{mmd:<3}, MMD Prog:{mmd_progress:<3}, MMD Prog:{mmd_progress_extra:<3} , CORAL:{coral:<3}, FLB:{flb_l:<3}, LR:{lr:<3}\n"
        # message = f"TTL:{ttl:.5f}, BTL:{btl:.5f}, CE:{ce:.5f}, OT:{ot:.5f}, ACE:{ace:.5f}, MMD:{mmd:.5f}, MMD Prog:{mmd_progress:.5f}, MMD Prog:{mmd_progress_extra:.5f} , CORAL:{coral:.5f}, FLB:{flb_l:.5f}, LR:{lr:.5f}\n"    
        f = open(self.training_args.csv_log_dir, "a")
        f.write(message)
        f.close()
        return message

    def _csv_log_eval(self, log, step):
        ttl,btl,lr,ce,ot,ace,mmd,coral  = log["eval/total_loss"],log["eval/loss"], log["train/learning_rate"],  log["cross_entropy"],log["optimal_transport"], log["adversarial_cross_entropy"],log['mmd'],log['coral']
        message = f"TTL:{ttl:<9}, BTL:{btl:<9}, CE:{ce:<9}, OT:{ot:<13}, ACE:{ace:<9}, MMD:{mmd:<9}, CORAL:{coral:<9}, LR:{lr:<9}\n"

        f = open(self.training_args.csv_log_dir[:-4] + '_Eval.txt', "a")
        f.write(message)
        f.close()


    def get_optimizer_and_scheduler(self, model, num_training_steps):
        """Returns optimizer and scheduler to use for training. If you are
        overriding this method and do not want to use a scheduler, simply
        return :obj:`None` for scheduler.

        Args:
            model (:obj:`torch.nn.Module`):
                Model to be trained. Pass its parameters to optimizer for training.
            num_training_steps (:obj:`int`):
                Number of total training steps.
        Returns:
            Tuple of optimizer and scheduler :obj:`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]`
        """
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        if isinstance(model, transformers.PreTrainedModel):
            # Reference https://huggingface.co/transformers/training.html

            if isinstance(self.training_args.num_warmup_steps, float):
                num_warmup_steps = math.ceil(
                    self.training_args.num_warmup_steps * num_training_steps
                )
            else:
                num_warmup_steps = self.training_args.num_warmup_steps

            if self.training_args.method_test == 'FLB':
                param_optimizer = list(model.named_parameters())
                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay":  0.000001 ,
                        "amsgrad":False,
                        "maximize": False,                 
                        },
                    {
                        "params": [
                            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                        "amsgrad":False,
                        "maximize": False,    
                    },
                ]
                optimizer = transformers.optimization.AdamW(
                optimizer_grouped_parameters, lr=self.training_args.learning_rate, eps= 0.00000001
                )
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler = CosineAnnealingLR(optimizer, len(self.train_dataset) // self.training_args.per_device_train_batch_size * self.training_args.num_epochs)
            elif self.training_args.method_test == 'Embedding':
                if self.training_args.Method_Dictionary['Embedding'] == 'ASCC':
                    param_optimizer = list(model.named_parameters())

                    # this  can be passed as a hyper parameter, but i don't always want to change it between this baseline and my tests so here is a hardcoded one
                    ascc_lr = 0.000002 # self.training_args.learning_rate
                    
                    no_decay = ["bias",  "LayerNorm.weight"]
                    optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay":  0.000001 ,
                        "amsgrad":False,
                        "maximize": False,           
                        "initial_lr":ascc_lr*10,      
                        },
                    {
                        "params": [
                            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                        "amsgrad":False,
                        "maximize": False,
                        "initial_lr":ascc_lr*10,         
                    },
                    ]

                    optimizer = transformers.optimization.AdamW(
                        optimizer_grouped_parameters, lr=ascc_lr ,eps= 0.00000001
                    )
                    # from utils.ascc_utils import WarmupMultiStepLR
                    scheduler = WarmupMultiStepLR(optimizer, (40, 80), 0.1, 1.0 / 10.0, 2, 'linear')
                    # scheduler = transformers.optimization.get_linear_schedule_with_warmup(
                    #     optimizer,
                    #     num_warmup_steps=num_warmup_steps,
                    #     num_training_steps=num_training_steps,
                    # )
                elif self.training_args.Method_Dictionary['Embedding'] == 'InfoBert':
                    param_optimizer = list(model.named_parameters())
                    no_decay = ["bias", "LayerNorm.weight"]
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in param_optimizer
                                if not any(nd in n for nd in no_decay)
                            ],
                            "weight_decay":  0.000001 ,
                            "amsgrad":False,
                            "maximize": False,                 
                            },
                        {
                            "params": [
                                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": 0.0,
                            "amsgrad":False,
                            "maximize": False,    
                        },
                    ]
                    optimizer = transformers.optimization.AdamW(
                    optimizer_grouped_parameters, lr=self.training_args.learning_rate, eps= 0.00000001
                    )
                    from torch.optim.lr_scheduler import CosineAnnealingLR
                    scheduler = CosineAnnealingLR(optimizer, len(self.train_dataset) // self.training_args.per_device_train_batch_size * self.training_args.num_epochs)
                elif self.training_args.Method_Dictionary['Embedding'] == 'DSRM':
                    
                    
                    lr = 2e-05
                    weight_decay= 0.01
                    adam_epsilon = 1e-08
                    no_decay = ["bias", "LayerNorm.weight"]
                    optimizer_grouped_parameters = [
                        {
                            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                            "weight_decay":  weight_decay,
                        },
                        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
                    ]


                    optimizer = torch.optim.AdamW(
                        optimizer_grouped_parameters,
                        lr= lr,
                        eps= adam_epsilon,
                        # correct_bias=args.bias_correction
                    )

                    from torch.optim.lr_scheduler import LambdaLR

                    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
                        """ Create a schedule with a learning rate that decreases linearly after
                        linearly increasing during a warmup period.

                        From:
                            https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
                        """

                        def lr_lambda(current_step):
                            if current_step < num_warmup_steps:
                                return float(current_step) / float(max(1, num_warmup_steps))
                            return max(
                                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
                            )

                        return LambdaLR(optimizer, lr_lambda, last_epoch)
                    warmup_ratio = 0.1
                    num_training_steps = len(self.train_dataset) * self.training_args.num_epochs // self.training_args.per_device_train_batch_size
                    warmup_steps = num_training_steps *  warmup_ratio
                    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)          

            else:
                param_optimizer = list(model.named_parameters())
                no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay":  self.training_args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ] 

                optimizer = transformers.optimization.AdamW(
                    optimizer_grouped_parameters, lr=self.training_args.learning_rate 
                )
                scheduler = transformers.optimization.get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                )

            

            
 
        else:
            optimizer = torch.optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                lr=self.training_args.learning_rate,
            )
            scheduler = None
            
         
        return optimizer, scheduler

    def get_train_dataloader(self, dataset, adv_dataset, batch_size,progress_dataset=None):
        """Returns the :obj:`torch.utils.data.DataLoader` for training.

        Args:
            dataset (:class:`~textattack.datasets.Dataset`):
                Original training dataset.
            adv_dataset (:class:`~textattack.datasets.Dataset`):
                Adversarial examples generated from the original training dataset. :obj:`None` if no adversarial attack takes place.
            batch_size (:obj:`int`):
                Batch size for training.
        Returns:
            :obj:`torch.utils.data.DataLoader`
        """
        # TODO: Add pairing option where we can pair original examples with adversarial examples.
        # Helper functions for collating data
        def collate_fn(data):
            input_texts = []
            targets = []
            is_adv_sample = []

            for item in data:

                if "_example_type" in item[0].keys():

                    # Get example type value from OrderedDict and remove it

                    adv = item[0].pop("_example_type")

                    # with _example_type removed from item[0] OrderedDict
                    # all other keys should be part of input
                    _input, label = item

                    if adv == "adversarial_example":
                        is_adv_sample.append(True)
                    elif adv == "progressive_example":
                        is_adv_sample.append(False)
                    else:
                        raise ValueError(
                            "`item` has length of 3 but last element is not for marking if the item is an `adversarial example`."
                        )

                else:
                    # else `len(item)` is 2.
                    _input, label = item
                    is_adv_sample.append(False)

                if isinstance(_input, collections.OrderedDict):
                    _input = tuple(_input.values())
                else:
                    _input = tuple(_input)

                if len(_input) == 1:
                    _input = _input[0]
                input_texts.append(_input)
                targets.append(label)

            return input_texts, torch.tensor(targets), torch.tensor(is_adv_sample)

        if adv_dataset:
            dataset = torch.utils.data.ConcatDataset([dataset, adv_dataset])

        if progress_dataset:
            dataset = torch.utils.data.ConcatDataset([dataset, progress_dataset])


        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True, #NEEDCHANGE#
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last =True,
        )
        return train_dataloader

    def get_eval_dataloader(self, dataset, batch_size):
        """Returns the :obj:`torch.utils.data.DataLoader` for evaluation.

        Args:
            dataset (:class:`~textattack.datasets.Dataset`):
                Dataset to use for evaluation.
            batch_size (:obj:`int`):
                Batch size for evaluation.
        Returns:
            :obj:`torch.utils.data.DataLoader`
        """
        # Helper functions for collating data
        def collate_fn(data):
            input_texts = []
            targets = []
            for _input, label in data:
                if isinstance(_input, collections.OrderedDict):
                    _input = tuple(_input.values())
                else:
                    _input = tuple(_input)

                if len(_input) == 1:
                    _input = _input[0]
                input_texts.append(_input)
                targets.append(label)
            return input_texts, torch.tensor(targets)

        eval_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return eval_dataloader

    # def training_step(self, model, tokenizer, batch,adv_batch,progress_batch,progress_extra_batch,pos_base_batch,neg_base_batch,pos_adv_batch,neg_adv_batch):
    def training_step(self,**kwargs_train):
        # print ('kwargs_train',kwargs_train) 
        model = kwargs_train['model']
        tokenizer = kwargs_train['tokenizer']
        batch = kwargs_train['batch']
        adv_batch = kwargs_train['adv_batch']
        progress_batch = kwargs_train['progress_batch']
        progress_extra_batch = kwargs_train['progress_extra_batch'] 
        time_record = kwargs_train['time_record'] 
        # model, tokenizer, batch,adv_batch,progress_batch,progress_extra_batch,pos_base_batch,neg_base_batch,pos_adv_batch,neg_adv_batch,val_dataloader
        """Perform a single training step on a batch of inputs.

        Args:
            model (:obj:`torch.nn.Module`):
                Model to train.
            tokenizer:
                Tokenizer used to tokenize input text.
            batch (:obj:`tuple[list[str], torch.Tensor, torch.Tensor]`):
                By default, this will be a tuple of input texts, targets, and boolean tensor indicating if the sample is an adversarial example.

                .. note::
                    If you override the :meth:`get_train_dataloader` method, then shape/type of :obj:`batch` will depend on how you created your batch.

        Returns:
            :obj:`tuple[torch.Tensor, torch.Tensor, torch.Tensor]` where

            - **loss**: :obj:`torch.FloatTensor` of shape 1 containing the loss.
            - **preds**: :obj:`torch.FloatTensor` of model's prediction for the batch.
            - **targets**: :obj:`torch.Tensor` of model's targets (e.g. labels, target values).
        """

        input_texts, targets, is_adv_sample = batch
        _targets = targets
        targets = targets.to(textattack.shared.utils.device)

        if progress_batch:
            progress_input_texts, progress_targets, progress_is_progress_sample = progress_batch
            _progress_targets = progress_targets
            progress_targets = progress_targets.to(textattack.shared.utils.device)

        if progress_extra_batch:
            progress_extra_input_texts, progress_extra_targets, progress_extra_is_progress_sample =  progress_extra_batch
            _progress_extra_targets = progress_extra_targets
            progress_extra_targets = progress_extra_targets.to(textattack.shared.utils.device)

        if adv_batch:
            adv_input_texts, adv_targets, adv_is_adv_sample = adv_batch
            _adv_targets = adv_targets
            adv_targets = adv_targets.to(textattack.shared.utils.device)

            if self.task_type == 'OT_GL_CC':
                pos_base_input_texts, pos_base_targets, _ = kwargs_train['pos_base_batch']
                _pos_base_targets = pos_base_targets
                pos_base_targets = pos_base_targets.to(textattack.shared.utils.device)

                neg_base_input_texts, neg_base_targets, _ = kwargs_train['neg_base_batch']
                _neg_base_targets = neg_base_targets
                neg_base_targets = neg_base_targets.to(textattack.shared.utils.device)

                pos_adv_input_texts, pos_adv_targets, _ = kwargs_train['pos_adv_batch']
                _pos_adv_targets = pos_adv_targets
                pos_adv_targets = pos_adv_targets.to(textattack.shared.utils.device)

                neg_adv_input_texts, neg_adv_targets, _ = kwargs_train['neg_adv_batch']
                _neg_adv_targets = neg_adv_targets
                neg_adv_targets = neg_adv_targets.to(textattack.shared.utils.device)


        
        if isinstance(model, transformers.PreTrainedModel) or (
            isinstance(model, torch.nn.DataParallel)
            and isinstance(model.module, transformers.PreTrainedModel)
        ):

            # import nvidia_smi
            #
            # nvidia_smi.nvmlInit()

            input_ids = tokenizer(
                input_texts,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )

            input_ids.to(textattack.shared.utils.device)

            if self.training_args.method_test == 'Embedding':
                output = None
                logits = None
            else:
                output = model(**input_ids,return_dict=True,output_hidden_states=True)
                logits = output.logits

            
            


            if self.training_args.method_test == 'FLB':
                pooler_output = None
                hidden_states = output.hidden_states
                base_last_hidden_state = hidden_states[-1]
                base_last_hidden_state = base_last_hidden_state[::,0,::]
            elif self.training_args.method_test == 'Embedding':
                # if self.training_args.Method_Dictionary[self.training_args.method_test] == 'ASCC':
                pooler_output = None
                hidden_states = None
                base_last_hidden_state = None 

            else:
                pooler_output =output.pooler_output
                hidden_states = output.hidden_states
                base_last_hidden_state = hidden_states[-1]
                base_last_hidden_state = base_last_hidden_state[::,0,::]
            
            # logits = output.logits #[0]
            # if self.training_args.method_test == 'FLB':
            #     pooler_output = None
            # else:
            #     pooler_output =output.pooler_output 
            
            # hidden_states = output.hidden_states
            # base_last_hidden_state = hidden_states[-1]
            # base_last_hidden_state = base_last_hidden_state[::,0,::]

            # print ('size',len(output.hidden_states))
            # print ('last hidden',base_last_hidden_state)
            # print ('size',base_last_hidden_state.shape)
            # print ('size out ppooler',pooler_output)



            # nvidia_smi.nvmlInit()
            # deviceCount = nvidia_smi.nvmlDeviceGetCount()
            # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(3)
            # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            # print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(3, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

            if self.training_args.Online_AT_Val == 'TextFooler': 
                online_adv_samples = self._get_online_samples(batch)
                if len(online_adv_samples[0]) != 0:
                    
                    online_adv_input_texts, online_adv_targets, _ = online_adv_samples
                    _online_adv_targets = online_adv_targets
                    online_adv_targets = online_adv_targets.to(textattack.shared.utils.device)

                    online_adv_input_ids = tokenizer(
                        online_adv_input_texts,
                        padding="max_length",
                        return_tensors="pt",
                        truncation=True,
                    )

                    online_adv_input_ids.to(textattack.shared.utils.device)
                    online_adv_output = model(**online_adv_input_ids,return_dict=True,output_hidden_states=True)

                    online_adv_logits = online_adv_output.logits #[0]
                    
                    online_adv_pooler_output =online_adv_output.pooler_output
                    # print ('pooler out:',pooler_output[0][:5])
                    online_adv_hidden_states = online_adv_output.hidden_states
                    online_adv_last_hidden_state = online_adv_hidden_states[-1]
                    online_adv_last_hidden_state = online_adv_last_hidden_state[::,0,::]
                else:
                    online_adv_logits = None
                    online_adv_pooler_output = None
                    online_adv_last_hidden_state = None
                





                
                        



            if adv_batch:
                adv_input_ids = tokenizer(
                    adv_input_texts,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )
                adv_input_ids.to(textattack.shared.utils.device)

                adv_output = model(**adv_input_ids ,return_dict=True,output_hidden_states=True)

                adv_logits = adv_output[0]
                adv_pooler_output = adv_output.pooler_output

                adv_hidden_states = adv_output.hidden_states
                adv_last_hidden_state = adv_hidden_states[-1]
                adv_last_hidden_state = adv_last_hidden_state[::,0,::]

                # print ('size',len(adv_output.hidden_states))
                # print ('last hidden',adv_last_hidden_state)
                # print ('size',adv_last_hidden_state.shape)
                # print ('size out ppooler',adv_pooler_output) 

                # deviceCount = nvidia_smi.nvmlDeviceGetCount()
                # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(3)
                # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                # print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(3, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

                # if self.task_type == 'OT_GL_CC':
                    # pos_base_input_ids = tokenizer(
                    #     pos_base_input_texts,
                    #     padding="max_length",
                    #     return_tensors="pt",
                    #     truncation=True,
                    # )
                    # pos_base_input_ids.to(textattack.shared.utils.device)
                    # pos_base_output = model(**pos_base_input_ids ,return_dict=True)
                    # pos_base_logits = pos_base_output[0]
                    # pos_base_pooler_output = pos_base_output.pooler_output
                    #
                    #
                    # neg_base_input_ids = tokenizer(
                    #     neg_base_input_texts,
                    #     padding="max_length",
                    #     return_tensors="pt",
                    #     truncation=True,
                    # )
                    # neg_base_input_ids.to(textattack.shared.utils.device)
                    # neg_base_output = model(**neg_base_input_ids ,return_dict=True)
                    # neg_base_logits = neg_base_output[0]
                    # neg_base_pooler_output = neg_base_output.pooler_output
                    #
                    # pos_adv_input_ids = tokenizer(
                    #     pos_adv_input_texts,
                    #     padding="max_length",
                    #     return_tensors="pt",
                    #     truncation=True,
                    # )
                    # pos_adv_input_ids.to(textattack.shared.utils.device)
                    # pos_adv_output = model(**pos_adv_input_ids ,return_dict=True)
                    # pos_adv_logits = pos_adv_output[0]
                    # pos_adv_pooler_output = pos_adv_output.pooler_output
                    #
                    #
                    # neg_adv_input_ids = tokenizer(
                    #     neg_adv_input_texts,
                    #     padding="max_length",
                    #     return_tensors="pt",
                    #     truncation=True,
                    # )
                    # neg_adv_input_ids.to(textattack.shared.utils.device)
                    # neg_adv_output = model(**neg_adv_input_ids ,return_dict=True)
                    # neg_adv_logits = neg_adv_output[0]
                    # neg_adv_pooler_output = neg_adv_output.pooler_output
            if progress_batch:
                # training_step
                progress_input_ids = tokenizer(
                    progress_input_texts,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )
                progress_input_ids.to(textattack.shared.utils.device)

                progress_output = model(**progress_input_ids ,return_dict=True,output_hidden_states=True)

                progress_logits = progress_output[0]
                progress_pooler_output = progress_output.pooler_output

                progress_hidden_states = progress_output.hidden_states
                progress_last_hidden_state = progress_hidden_states[-1]
                progress_last_hidden_state = progress_last_hidden_state[::,0,::]

            if progress_extra_batch:
                # training_step
                progress_extra_input_ids = tokenizer(
                    progress_extra_input_texts,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )
                progress_extra_input_ids.to(textattack.shared.utils.device)

                progress_extra_output = model(**progress_extra_input_ids ,return_dict=True,output_hidden_states=True)

                progress_extra_logits = progress_extra_output[0]
                progress_extra_pooler_output = progress_extra_output.pooler_output

                progress_extra_hidden_states = progress_extra_output.hidden_states
                progress_extra_last_hidden_state = progress_extra_hidden_states[-1]
                progress_extra_last_hidden_state = progress_extra_last_hidden_state[::,0,::]



        else:
            input_ids = tokenizer(input_texts)
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            input_ids = input_ids.to(textattack.shared.utils.device)
            logits = model(input_ids)


        
        class L2_loss(torch.nn.Module):
            def __init__(self):
                super(L2_loss, self).__init__()

            def forward(self, source, target):
                distances = torch.norm(source - target, dim=1, p=2)
                # print ('shape distances',distances.shape)
                average_distance = distances.mean()
                # print ('avg distance',average_distance,average_distance.shape)
                return average_distance
                # return torch.dist(source, target, 2)
        ### new mmd


        class MMD_loss(torch.nn.Module):
            def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
                super(MMD_loss, self).__init__()
                self.kernel_num = kernel_num
                self.kernel_mul = kernel_mul
                self.fix_sigma = None
                self.kernel_type = kernel_type

            def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
                n_samples = int(source.size()[0]) + int(target.size()[0])
                total = torch.cat([source, target], dim=0)
                total0 = total.unsqueeze(0).expand(
                    int(total.size(0)), int(total.size(0)), int(total.size(1)))
                total1 = total.unsqueeze(1).expand(
                    int(total.size(0)), int(total.size(0)), int(total.size(1)))
                L2_distance = ((total0-total1)**2).sum(2)
                if fix_sigma:
                    bandwidth = fix_sigma
                else:
                    bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
                bandwidth /= kernel_mul ** (kernel_num // 2)
                bandwidth_list = [bandwidth * (kernel_mul**i)
                                  for i in range(kernel_num)]
                kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                              for bandwidth_temp in bandwidth_list]
                return sum(kernel_val)

            def linear_mmd2(self, f_of_X, f_of_Y):
                loss = 0.0
                delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
                loss = delta.dot(delta.T)
                return loss

            def forward(self, source, target):
                if self.kernel_type == 'linear':
                    return self.linear_mmd2(source, target)
                elif self.kernel_type == 'rbf':
                    batch_size = int(source.size()[0])
                    kernels = self.guassian_kernel(
                        source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
                    XX = torch.mean(kernels[:batch_size, :batch_size])
                    YY = torch.mean(kernels[batch_size:, batch_size:])
                    XY = torch.mean(kernels[:batch_size, batch_size:])
                    YX = torch.mean(kernels[batch_size:, :batch_size])
                    loss = torch.mean(XX + YY - XY - YX)
                    
                    return loss


        def coral(source, target):

            d = source.size(1)  # dim vector

            source_c = compute_covariance(source)
            target_c = compute_covariance(target)

            loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

            loss = loss / (4 * d * d)
            return loss


        def compute_covariance(input_data):
            """
            Compute Covariance matrix of the input data
            """
            n = input_data.size(0)  # batch_size

            # Check if using gpu or cpu
            # if input_data.is_cuda:
            #     device = torch.device('cuda')
            # else:
            #     device = torch.device('cpu')


            id_row = torch.ones(n).resize(1, n).to(device=textattack.shared.utils.device)
            sum_column = torch.mm(id_row, input_data)
            mean_column = torch.div(sum_column, n)
            term_mul_2 = torch.mm(mean_column.t(), mean_column)
            d_t_d = torch.mm(input_data.t(), input_data)
            c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

            return c



        loss = 0
        s_loss = 0
        da_loss = 0
        adv_loss= 0
        mmd_loss = 0
        mmd_progress_loss = 0
        mmd_progress_extra_loss = 0
        crl_loss = 0
        total_flb_loss = 0
        pos_adv_da_loss = 0
        neg_adv_da_loss = 0
        l2_loss = 0

        if self.task_type == "regression":
            loss = self.loss_fct(logits.squeeze(), targets.squeeze())
            preds = logits
        elif self.task_type == "optimal_transport":
            ys = targets

            # ys = adv_targets# this 100% returns the GRAUND TRUTH LABEL AS SEEN FROM ORGINAL DATASET
            # to get the adv label we have to invert them https://huggingface.co/datasets/glue/viewer/mrpc/train to see ground truth labels, print ys and adv_input_texts


            g_xs_mb = pooler_output # source (base) pooler
            f_g_xs_mb = logits # logits of source
            g_xt_mb = adv_pooler_output # target (adv) pooler
            f_g_xt_mb = adv_logits # logits of the base
            pred_xt = F.softmax(f_g_xt_mb, 1)

            # s_loss = self.loss_fct(logits, targets)
            # print ('base logits',logits)
            # print ('targets (labels base)',targets)
            # print ('source adv logits and labels',f_g_xs_mb,ys)

            # s_loss = self.loss_fct(f_g_xs_mb, ys) # loss between adv ground truth label and adv predic, they will always be wrong
            # s_loss = self.loss_fct(logits, targets)
            # print ('s_loss',s_loss)




            self.loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            s_loss = self.loss_fct(logits, targets) # used to be on target
            # s_loss = self.loss_fct(f_g_xs_mb, ys) # now is on source (adv sam)
            # print ('g_xs_mb',g_xs_mb)
            # print ('g_xt_mb',g_xt_mb)

            self.loss_fct_adv = torch.nn.CrossEntropyLoss(reduction="mean")
            adv_loss = self.loss_fct_adv(f_g_xt_mb, adv_targets)




            embed_cost = torch.cdist(g_xs_mb, g_xt_mb)**2

            #ys probably needs to be the targets and the pred_xs needs to be F.softmax(f_g_xs_mb, 1), so the preditction on the source
            ys = F.one_hot(ys, num_classes=len(model.config.id2label)).float() #num_classes=self.n_class
            t_cost = - torch.mm(ys, torch.transpose(torch.log(pred_xt), 0, 1))
            self.eta1,self.eta2  = 0.001,  0.0001
            total_cost = self.eta1 * embed_cost + self.eta2 * t_cost

            self.epsilon, self.tau = 0.1,   1.
            a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, total_cost.detach().cpu().numpy(),  self.epsilon, self.tau)

            pi = torch.from_numpy(pi).float().cuda()

            da_loss = torch.sum(pi * total_cost)
            # print ('s loss',s_loss,'da loss',da_loss)
            #NEEDCHANGE#
            # yes adv, yes ot
            # tot_loss = s_loss + adv_loss + da_loss
            # no adv, yes ot
            # tot_loss = s_loss + da_loss
            # yes adv, no ot
            tot_loss = s_loss + adv_loss
            loss = tot_loss

            preds = logits.argmax(dim=-1)

            # f = open("outputlog.txt", "a")
            # str = f'loss : {s_loss.item()}, {da_loss.item()}\n '
            # f.write(str)
            # f.close()
            adv_loss = adv_loss.detach().cpu().item()
            s_loss= s_loss.detach().cpu().item()
            da_loss =da_loss.detach().cpu().item()

        elif self.task_type == "OT_GL": # each technique MMD, FLB, ASCC should be its own class but i put everyting in same file

            if output != None: 
                ys = targets

                # ys = adv_targets# this 100% returns the GRAUND TRUTH LABEL AS SEEN FROM ORGINAL DATASET
                # to get the adv label we have to invert them https://huggingface.co/datasets/glue/viewer/mrpc/train to see ground truth labels, print ys and adv_input_texts


                g_xs_mb = pooler_output # source (base) pooler
                # g_xs_mb = base_last_hidden_state


                f_g_xs_mb = logits # logits of source


                # g_xt_mb = adv_pooler_output # target (adv) pooler
                # f_g_xt_mb = adv_logits # logits of the base
                # pred_xt = F.softmax(f_g_xt_mb, 1)

                self.loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
                s_loss = self.loss_fct(logits, targets) 

                loss +=  (self.training_args.B_Val)*s_loss
                s_loss= s_loss.detach().cpu().item() 
                
                
            else:
                print ('skip calculating base loss, we are using a baseline')
                pass

            # b_loss = (self.training_args.B_Val)*s_loss

            if self.training_args.Online_AT_Val == 'TextFooler':

                if online_adv_logits is None and online_adv_pooler_output is None and online_adv_last_hidden_state is None:
                    online_adv_loss = torch.tensor(0)
                    online_da_loss = torch.tensor(0)
                else:
                    g_xt_mb = online_adv_pooler_output # target (adv) pooler
                    # g_xt_mb = adv_last_hidden_state

                    f_g_xt_mb = online_adv_logits # logits of the base
                    pred_xt = F.softmax(f_g_xt_mb, 1)

                    self.online_loss_fct_adv = torch.nn.CrossEntropyLoss(reduction="mean")
                    online_adv_loss = self.online_loss_fct_adv(f_g_xt_mb, online_adv_targets)
                    
                    
                    # MMD
                    online_geom_loss = MMD_loss()
                    online_da_loss = online_geom_loss(g_xs_mb,g_xt_mb)
                    # loss +=  (self.training_args.MMD_Val)*mmd_loss
                    # mmd_loss = mmd_loss.detach().cpu().item()

                    #OT loss
                    # online_geom_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
                    # online_da_loss = online_geom_loss(g_xs_mb,g_xt_mb)
                    # self.training_args.FLB_Val = 1

                    loss +=  (self.training_args.AT_Val)*online_adv_loss
                    loss +=  (self.training_args.MMD_Val)*online_da_loss
                    da_loss = online_da_loss.detach().cpu().item()
                    adv_loss = online_adv_loss.detach().cpu().item()

                # loss +=  online_adv_loss
                # loss +=  da_loss
                 
                # da_loss = online_da_loss.detach().cpu().item()
                # adv_loss = online_adv_loss.detach().cpu().item()

                print ('base loss',s_loss,'adv loss',adv_loss,'ot loss',da_loss)


            # print ('self.training_args.Method_Dictionary',self.training_args.Method_Dictionary)
             
            if 'Embedding' in self.training_args.Method_Dictionary:
                if self.training_args.Method_Dictionary['Embedding'] == 'InfoBert': 
                    # assert isinstance(batch[0], torch.Tensor)
                    # batch = tuple(t.to(self.model.device) for t in batch)
                    input_texts = input_texts

                    hidden_size = model.config.hidden_size
                    from utils.info_regularizer import (CLUB, InfoNCE) 

                    def feature_ranking(grad, cl=0.5, ch=0.9):
                        n = len(grad)
                        import math
                        lower = math.ceil(n * cl)
                        upper = math.ceil(n * ch)
                        norm = torch.norm(grad, dim=1)  # [seq_len]
                        _, ind = torch.sort(norm)
                        res = []
                        for i in range(lower, upper):
                            res += ind[i].item(),
                        return res

                    
                    
                    def get_seq_len(batch):
                        # print ('batch[1] 2',batch[1])
                        lengths = torch.sum(batch[1], dim=-1)
                        return lengths.detach().cpu().numpy()

                    def _train_mi_upper_estimator( outputs, batch=None):
                        # hidden_states = outputs[1]  # need to set config.output_hidden = True
                        hidden_states = outputs.hidden_states
                        # print ('hidden states',hidden_states)
                        # sys.exit()
                        last_hidden, embedding_layer = hidden_states[-1], hidden_states[0]  # embedding layer: batch x seq_len x 768
                        embeddings = []
                        
                        lengths = get_seq_len(batch)
                        for i, length in enumerate(lengths):
                            embeddings.append(embedding_layer[i, :length])
                        embeddings = torch.cat(embeddings)  # [-1, 768]   embeddings without masks
                        return self.mi_upper_estimator.update(embedding_layer, embeddings)

                    def _get_local_robust_feature_regularizer( args, outputs, local_robust_features):
                        hidden_states = outputs.hidden_states  # need to set config.output_hidden = True
                        last_hidden, embedding_layer = hidden_states[-1], hidden_states[0]  # embedding layer: batch x seq_len x 768
                        sentence_embeddings = last_hidden[:, 0]  # batch x 768  # CLS
                        local_embeddings = []
                        global_embeddings = []
                        for i, local_robust_feature in enumerate(local_robust_features):
                            for local in local_robust_feature:
                                local_embeddings.append(embedding_layer[i, local])
                                global_embeddings.append(sentence_embeddings[i])

                        lower_bounds = []
                        from sklearn.utils import shuffle
                        local_embeddings, global_embeddings = shuffle(local_embeddings, global_embeddings, random_state=args.info_seed)
                        for i in range(0, len(local_embeddings), self.training_args.per_device_train_batch_size):
                            local_batch = torch.stack(local_embeddings[i: i + self.training_args.per_device_train_batch_size])
                            global_batch = torch.stack(global_embeddings[i: i + self.training_args.per_device_train_batch_size])
                            lower_bounds += self.mi_estimator(local_batch, global_batch),
                        return -torch.stack(lower_bounds).mean()

                    def local_robust_feature_selection(args, batch, grad):
                        """
                        :param input_ids: for visualization, print out the local robust features
                        :return: list of list of local robust feature posid, non robust feature posid
                        """
                        grads = []
                        lengths = get_seq_len(batch)
                        for i, length in enumerate(lengths):
                            grads.append(grad[i, :length])
                        indices = []
                        nonrobust_indices = []
                        for i, grad in enumerate(grads):
                            indices.append(feature_ranking(grad, args.cl, args.ch))
                            nonrobust_indices.append([x for x in range(lengths[i]) if x not in indices])
                        return indices, nonrobust_indices

                    class InfoBert_Args:
                        def __init__(self,trainer_args_inner):
                            self.adv_norm_type = 'l2'
                            self.adv_init_mag = 8e-2
                             
                            if trainer_args_inner.Dataset_attack =='AGNEWS':
                                    self.adv_steps = 7  # with MR 2 epochs using 50 we got interesting performance
                                    self.adv_learning_rate = 0.1#  4e-2
                            elif trainer_args_inner.Dataset_attack =='MR':
                                
                                if trainer_args_inner.model_name == 'BERT':
                                    self.adv_steps = 15
                                    self.adv_learning_rate = 0.1#  4e-2
                                elif trainer_args_inner.model_name == 'ROBERTA' :
                                    self.adv_steps = 25
                                    self.adv_learning_rate = 0.1#  4e-2
                            elif trainer_args_inner.Dataset_attack =='SST2':
                                 self.adv_steps = 7
                                 self.adv_learning_rate = 0.1#  4e-2
                            else:
                                # used to be this in original textdefender implementation but we found undertraining
                                # where we have no robustness gains
                                self.adv_steps = 3
                                self.adv_learning_rate =  4e-2
                            self.alpha = 5e-3
                            # self.adv_learning_rate = 0.1#  4e-2
                            self.adv_max_norm = 0# 0.25
                            self.beta = 5e-3
                            self.cl = 0.5 
                            self.ch = 0.9
                            self.info_seed = 42
                            self.max_grad_norm = 1.0

                    infobert_args = InfoBert_Args(self.training_args) 
                
                     
                    self.mi_upper_estimator = CLUB(hidden_size, hidden_size, beta=infobert_args.beta).to(textattack.shared.utils.device)
                    self.mi_estimator = InfoNCE(hidden_size, hidden_size).to(textattack.shared.utils.device)

 


                    infobert_input_ids = tokenizer(
                        input_texts,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                    )

                    #             infobert_input_ids['input_ids'] = torch.tensor([[  101,  2859,  5317,  3067,  8479,  8563,  2321,  1011,  8418, 25311,
                    #   6692,  1037,  5317,  3067,  7738,  2038,  2730,  2012,  2560,  2321,
                    #  11257,  1999,  2642,  2859,  1010,  1996,  6745,  1999,  1037,  5164,
                    #   1997, 13436,  1999,  1996,  2406,  1001,  4464,  1025,  1055, 12536,
                    #   2135,  4795,  5471,  3068,  1010,  1996,  2880,  8418, 25311,  6692,
                    #   2739,  4034,  2056,  2006,  5958,  1012,   102,     0],
                    # [  101,  1060,  1011,  4097,  1997,  2332, 10722,  2102, 22788,  2089,
                    #   7487, 15774,  1006,  9706,  1007,  9706,  1011,  1996, 22788,  1997,
                    #   2332, 10722,  5794, 15256, 23041,  2003,  2000,  2022,  1060,  1011,
                    #   4097,  2098,  1999,  2019,  3535,  2000,  9611,  1996,  6547,  1997,
                    #   2129,  1996,  9454, 22089,  2351,  2012,  2287,  2459,  1010,  5279,
                    #   1005,  1055,  2708, 18821,  2056,  4465,  1012,   102]])

                    #             infobert_input_ids['attention_mask'] = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    #  1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

                    #             infobert_input_ids['token_type_ids'] = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])                                

                    # print ('infobert_input_ids',infobert_input_ids)
                    # print ('prev targets', targets)
                    # targets = torch.tensor([0, 3]).to(textattack.shared.utils.device)
                    # print ('new targets',targets)
                    
                     
                    # print ('truncation no?',ascc_input_ids['input_ids'].shape)  

                    

                    infobert_input_tokens = infobert_input_ids['input_ids'].to(textattack.shared.utils.device) # adv_input_ids['input_ids']
                    infobert_attention_mask = infobert_input_ids['attention_mask'].to(textattack.shared.utils.device)# adv_input_ids['attention_mask']
                    
                    infobert_kargs_base = {'input_ids':infobert_input_tokens,'attention_mask': infobert_attention_mask}
                    
                    if 'token_type_ids' in infobert_input_ids:
    
                        infobert_token_type_ids  = infobert_input_ids['token_type_ids'].to(textattack.shared.utils.device)
                        infobert_kargs_base['token_type_ids'] = infobert_token_type_ids
                    else:
                        infobert_token_type_ids = torch.zeros_like(infobert_attention_mask)
                        infobert_kargs_base['token_type_ids'] = infobert_token_type_ids
                    # infobert_targets = targets
 
                    # if 'token_type_ids' in infobert_input_ids:
                    #     infobert_kargs_base['token_type_ids'] = infobert_token_type_ids
                    # else:
                    #     infobert_token_type_ids = torch.zeros_like(infobert_attention_mask)
                    #     infobert_kargs_base['token_type_ids'] = infobert_token_type_ids

                    
                    # print ('infobert_kargs_base',infobert_kargs_base)

                    infobert_kargs_base_batch = [infobert_kargs_base['input_ids'],infobert_kargs_base['attention_mask'],infobert_kargs_base['token_type_ids'],targets]

                    word_embedding_layer = model.get_input_embeddings()

                    # init input_ids and mask
                    tr_loss, upperbound_loss, lowerbound_loss = 0.0, 0.0, 0.0
                    input_ids, attention_mask, labels = infobert_kargs_base_batch[0], infobert_kargs_base_batch[1], infobert_kargs_base_batch[3]
                    embeds_init = word_embedding_layer(input_ids)

                    input_mask = attention_mask.float()
                    input_lengths = torch.sum(input_mask, 1) # B 
                    
                    
                    if infobert_args.adv_init_mag > 0:
                        if infobert_args.adv_norm_type == "l2":
                            delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                            dims = input_lengths * embeds_init.size(-1)
                            mag = infobert_args.adv_init_mag / torch.sqrt(dims)
                            delta = (delta * mag.view(-1, 1, 1)).detach()
                        elif infobert_args.adv_norm_type == "linf":
                            delta = torch.zeros_like(embeds_init).uniform_(-infobert_args.adv_init_mag,
                                                                            infobert_args.adv_init_mag) * input_mask.unsqueeze(2)
                    else:
                        delta = torch.zeros_like(embeds_init)
                    
                    for astep in range(infobert_args.adv_steps):
                        delta.requires_grad_()
                        infobert_kargs_base_batch = (embeds_init + delta, infobert_kargs_base_batch[1], infobert_kargs_base_batch[2])


                        # (1) backward
                        outputs = model(inputs_embeds = infobert_kargs_base_batch[0],attention_mask=infobert_kargs_base_batch[1],token_type_ids =infobert_kargs_base_batch[2],output_hidden_states = True )
                        # print ('outputs',outputs)
                        # print ('hidden_states',outputs.hidden_states)
                        
                        logits = outputs[0]  
                        loss_function = torch.nn.CrossEntropyLoss(reduction='none')
                        losses = loss_function(logits, labels.view(-1))
                        loss_inner = torch.mean(losses)
                        loss_inner = loss_inner / infobert_args.adv_steps

                        tr_loss += loss_inner.item()
                        print ('tr loss internal',tr_loss)

                        if self.mi_upper_estimator: 
                            
                            upper_bound = _train_mi_upper_estimator(outputs, infobert_kargs_base_batch) / infobert_args.adv_steps
                            loss_inner += upper_bound
                            upperbound_loss += upper_bound.item()

                        loss_inner.backward(retain_graph=True)

                        # (2) get gradient on delta
                        delta_grad = delta.grad.clone().detach()
                        if self.mi_estimator:
                            local_robust_features, _ = local_robust_feature_selection(infobert_args, infobert_kargs_base_batch, delta_grad)
                            lower_bound = _get_local_robust_feature_regularizer(infobert_args, outputs, local_robust_features) * \
                                        infobert_args.alpha / infobert_args.adv_steps
                            lower_bound.backward()
                            lowerbound_loss += lower_bound.item()

                        if astep == infobert_args.adv_steps - 1:  ## if no freelb, set astep = 1, adv_init=0
                            # further updates on delta
                            break

                        # (3) update and clip
                        if infobert_args.adv_norm_type == "l2":
                            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                            denorm = torch.clamp(denorm, min=1e-8)
                            delta = (delta + infobert_args.adv_learning_rate * delta_grad / denorm).detach()
                            if infobert_args.adv_max_norm > 0:
                                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                                exceed_mask = (delta_norm > infobert_args.adv_max_norm).to(embeds_init)
                                reweights = (infobert_args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                                delta = (delta * reweights).detach()
                        elif infobert_args.adv_norm_type == "linf":
                            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                            denorm = torch.clamp(denorm, min=1e-8)
                            delta = (delta + infobert_args.adv_learning_rate * delta_grad / denorm).detach()
                            if infobert_args.adv_max_norm > 0:
                                delta = torch.clamp(delta, -infobert_args.adv_max_norm, infobert_args.adv_max_norm).detach()
                        else:
                            print("Norm type {} not specified.".format(infobert_args.adv_norm_type))
                            exit()

                        embeds_init = word_embedding_layer(input_ids)
                    # clear_mask()

                    # torch.nn.utils.clip_grad_norm_(model.parameters(), infobert_args.max_grad_norm)

                    # self.optimizer.step()

                    loss_dict = {"task_loss": tr_loss}
                    if self.mi_upper_estimator:
                        loss_dict.update({"upper_bound": upperbound_loss})
                    if self.mi_estimator:
                        loss_dict.update({"lower_bound": lowerbound_loss})

                    # print ('tr_loss',tr_loss)
                    loss += tr_loss
                    s_loss = tr_loss#.detach().cpu().item() 

                    print ('loss',loss)
                    



                elif self.training_args.Method_Dictionary['Embedding'] == 'ASCC': 
                    
                    # assert isinstance(batch[0], torch.Tensor)
                    # batch = tuple(t.cuda() for t in batch)
                    # golds = batch[3]

                    input_texts = input_texts
                    



                    ascc_input_ids = tokenizer(
                        input_texts,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                    )

                    # ascc_input_ids['input_ids'] = torch.tensor([[  101, 11046,  5823,  1037, 10520,  1010,  2021,  2001,  1037,  2488,
                    #                             12504,  2070,  2671,  2089,  2031,  2042,  2439,  2004,  1996, 11046,
                    #                             2686, 18269,  5565,  1999,  1996,  6646,  5532,  2007,  1037, 20605,
                    #                             1012,  2021,  2045,  2453,  2022,  1037,  1043, 17960,  5017,  1997,
                    #                             2204,  2739,  1999,  1996,  3478,  9274,  3260,  1012,   102,     0,
                    #                                 0,     0,     0,     0,     0,     0,     0,     0,     0],
                    #                             [  101,  9409,  2000,  5672,  5766, 24190,  1011,  3027,  1006, 26665,
                    #                             1007, 26665,  1011, 10799, 24190,  1010,  3472,  1998,  2708,  1032,
                    #                             3237,  1997,  7861, 14479, 14782,  5427, 20138,  9409, 11338,  7770,
                    #                             7229,  2522,  2015,  1012,  1032,  1010,  2003,  3517,  2000,  3357,
                    #                             2091,  2306,  2847,  1010,  1037,  3780,  1032,  2988,  2006,  5958,
                    #                             1010,  8951,  2111,  2485,  2000,  1996, 10287,  1012,   102]])

                    # ascc_input_ids['attention_mask'] = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    #                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    #                                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    #                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    #                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    #                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

                    # ascc_input_ids['token_type_ids'] = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    #                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])                                

                    # print ('ascc_input_ids',ascc_input_ids)
                    # print ('prev targets', targets)
                    # targets = torch.tensor([3, 2]).to(textattack.shared.utils.device)
                    # print ('new targets',targets)
                    
                     
                    # print ('truncation no?',ascc_input_ids['input_ids'].shape)  

                    

                    ascc_input_tokens = ascc_input_ids['input_ids'].to(textattack.shared.utils.device) # adv_input_ids['input_ids']
                    ascc_attention_mask = ascc_input_ids['attention_mask'].to(textattack.shared.utils.device)# adv_input_ids['attention_mask']
                    if 'token_type_ids' in ascc_input_ids:
                        ascc_token_type_ids  = ascc_input_ids['token_type_ids'].to(textattack.shared.utils.device)
                    ascc_targets = targets

                    ascc_kargs_base = {'input_ids':ascc_input_tokens,'attention_mask': ascc_attention_mask}
                    if 'token_type_ids' in ascc_input_ids:
                        ascc_kargs_base['token_type_ids'] = ascc_token_type_ids

                    
                    # print ('ascc_kargs_base',ascc_kargs_base)
                    # if 'token_type_ids' in ascc_input_ids:
                    #     outputs_ascc = model(input_ids = ascc_kargs_base['input_ids'], attention_mask=ascc_kargs_base['attention_mask'], token_type_ids=ascc_kargs_base['token_type_ids'],return_dict=True,output_hidden_states=True)
                    # else:
                    #     outputs_ascc = model(input_ids = ascc_kargs_base['input_ids'], attention_mask=ascc_kargs_base['attention_mask'], token_type_ids=ascc_kargs_base['token_type_ids'],return_dict=True,output_hidden_states=True)
                    outputs_ascc = model(**ascc_kargs_base)
                    
                    
                    clean_logits, adv_logits = outputs_ascc.logits, outputs_ascc.KL_loss
                    logits = clean_logits
                    golds = targets
                    loss_function_ascc = torch.nn.CrossEntropyLoss(reduction='none') 
                    clean_loss_ascc = torch.mean(loss_function_ascc(clean_logits, golds))
                    # print ('clean_loss_ascc',clean_loss_ascc)
                    # sys.exit()
                    adv_loss_ascc = F.kl_div(torch.softmax(adv_logits, dim=1).log(), torch.softmax(clean_logits, dim=1), None, None, 'batchmean')
                    beta = 4
                    total_loss_ascc = clean_loss_ascc + beta * adv_loss_ascc
                    # total_loss.backward() \ in the orignal implementation we have a last backward outside the loop
                    # max_grad_norm = 1
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    # self.optimizer.step()
                    # print ("self.training_args.Method_Dictionary['Embedding_Val']",self.training_args.Method_Dictionary['Embedding_Val'])
                    total_loss_ascc = total_loss_ascc*self.training_args.Method_Dictionary['Embedding_Val']
                    loss += total_loss_ascc
                    s_loss = clean_loss_ascc.detach().cpu().item()
                    # print ('standard loss', s_loss)
                    adv_loss_ascc = adv_loss_ascc.detach().cpu().item()
                    # print ('adv_loss_ascc', adv_loss_ascc)
                    total_loss_ascc = total_loss_ascc.detach().cpu().item()
                    # print ('loss',loss) 
                    # sys.exit()
                    
                     
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                elif self.training_args.Method_Dictionary['Embedding'] == 'DSRM':
                    # get validation batch 
                    input_texts = input_texts

                    dsrm_input_ids = tokenizer(
                        input_texts,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                    )

                    # dsrm_input_ids = {}


                    # dsrm_input_ids['input_ids'] = torch.tensor([[  101,  2130,  3972, 22471,  3085,   102,     0,     0,     0,     0,
                    #                     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                    #                 [  101,  2071,  1050,  1005,  1056,  2022,  2488,  2004,  1037, 10311,
                    #                 2021,  6881,  2135,  5622,  2912,  3468, 19411, 13523,  4948,   102]])

                    # dsrm_input_ids['attention_mask'] = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    #                                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

                    # dsrm_input_ids['token_type_ids'] = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    #                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

                    # targets = torch.tensor([1, 1]).to(textattack.shared.utils.device)



                    dsrm_input_tokens = dsrm_input_ids['input_ids'].to(textattack.shared.utils.device) # adv_input_ids['input_ids']
                    dsrm_attention_mask = dsrm_input_ids['attention_mask'].to(textattack.shared.utils.device)# adv_input_ids['attention_mask']
                    if 'token_type_ids' in dsrm_input_ids:
                        dsrm_token_type_ids  = dsrm_input_ids['token_type_ids'].to(textattack.shared.utils.device)
                    dsrm_targets = targets

                    dsrm_kargs_base = {'input_ids':dsrm_input_tokens,'attention_mask': dsrm_attention_mask}
                    if 'token_type_ids' in dsrm_input_ids:
                        dsrm_kargs_base['token_type_ids'] = dsrm_token_type_ids

                    
                    


                    val_batch = next(kwargs_train['val_cycle_dataloader'])

                    dsrm_val_input_texts, dsrm_val_targets, val_is_adv_sample = val_batch
                    _val_targets = dsrm_val_targets
                    dsrm_val_targets = dsrm_val_targets.to(textattack.shared.utils.device)

                    dsrm_val_input_ids = tokenizer(
                        dsrm_val_input_texts,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                    )




                    # dsrm_val_input_ids['input_ids'] = torch.tensor([[  101,  2000,  3443,  1037,  3444,  2143,  2008,  2003, 10433,  2135,
                    #                         4569,  2000,  3422,   102,     0,     0,     0,     0,     0,     0,
                    #                             0,     0,     0,     0,     0,     0],
                    #                         [  101,  6388,  2438,  2000,  7344,  3686,  1996,  7731,  4378,  2096,
                    #                         13060, 18856, 17322,  2094,  2000, 15015, 10271,  1011,  4641,   102,
                    #                             0,     0,     0,     0,     0,     0],
                    #                         [  101,  1996,  7961,  1997,  2009,  2035,  2097,  2022,  3306,  2000,
                    #                         3087,  2025,  3653, 10521, 19155,  2000,  1996,  3185,  1005,  1055,
                    #                         12726,  1998, 13587,  8562,  1012,   102],
                    #                         [  101,  2196,  3243,  6162,  2015,  1996,  2514,  1997,  1037,  5470,
                    #                         26336,  4367,  3861,  1012,   102,     0,     0,     0,     0,     0,
                    #                             0,     0,     0,     0,     0,     0]])

                    # dsrm_val_input_ids['attention_mask'] = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #                     0, 0],
                    #                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                    #                     0, 0],
                    #                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    #                     1, 1],
                    #                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #                     0, 0]])

                    # dsrm_val_input_ids['token_type_ids'] = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #                     0, 0],
                    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #                     0, 0],
                    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #                     0, 0],
                    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #                     0, 0]])

                    # dsrm_val_targets = torch.tensor([1, 0, 0, 0]).to(textattack.shared.utils.device)







                    dsrm_val_input_tokens = dsrm_val_input_ids['input_ids'].to(textattack.shared.utils.device) # adv_input_ids['input_ids']
                    dsrm_val_attention_mask = dsrm_val_input_ids['attention_mask'].to(textattack.shared.utils.device)# adv_input_ids['attention_mask']
                    if 'token_type_ids' in dsrm_val_input_ids:
                        dsrm_val_token_type_ids  = dsrm_val_input_ids['token_type_ids'].to(textattack.shared.utils.device)
                    dsrm_val_targets = dsrm_val_targets

                    dsrm_val_kargs_base = {'input_ids':dsrm_val_input_tokens,'attention_mask': dsrm_val_attention_mask}
                    if 'token_type_ids' in dsrm_val_input_ids:
                        dsrm_val_kargs_base['token_type_ids'] = dsrm_val_token_type_ids

                    # avg_loss = training_utils_functions.ExponentialMovingAverage()
                    # print ('cycle dataloaders',kwargs_train['val_cycle_dataloader'])
                    # val_batch = next(kwargs_train['val_cycle_dataloader'])
                    # print ('adv batch',val_batch) 

                    
                    # print ('input',dsrm_kargs_base)
                    # print ('model',model)
                    # for name, param in model.named_parameters():
                    #     print(name, param.data)
                    dsrm_outputs_base = model(**dsrm_kargs_base)

                    dsrm_logits = dsrm_outputs_base.logits
                    logits = dsrm_logits

                    # print ('dsrm_outputs_base',dsrm_outputs_base)

                    self.dsrm_loss_function =  torch.nn.CrossEntropyLoss(reduction='none')
                    # print ('logits and targets',dsrm_logits,dsrm_targets.view(-1))
                    # print ('shapes',dsrm_logits.shape,dsrm_targets.shape)
                    
                    dsrm_losses = self.dsrm_loss_function(dsrm_logits,dsrm_targets.view(-1)) #, golds.view(-1))
                    
                    
                    preds = dsrm_logits.argmax(dim=-1)
                    
                    # import torch.nn.functional as F
                    # dsrm_losses = F.cross_entropy(dsrm_logits,dsrm_targets.view(-1),reduction='none')
                    # print('initial losses',dsrm_losses)
                    
                    # self.dsrm_weights = torch.nn.Linear(self.training_args.per_device_train_batch_size, 1, bias=False)

                    shifting = True
                    loss_clamp = 0.5#0.0#5 #0.1
                    dsrm_val_loss = 0
                    if shifting and torch.mean(dsrm_losses) < loss_clamp:
                        # torch.nn.init.constant_(self.dsrm_weights.weight, 1)
                        model.refresh_weights()
                         
                        with higher.innerloop_ctx(model,self.optimizer_training_step , device=textattack.shared.utils.device) as (fmodel, diffopt):
                            fmodel.train()
                            # print ('shifting!',torch.mean(dsrm_losses) ,'<', loss_clamp)

                                    
                            meta_base_output = fmodel(**dsrm_kargs_base)
                            meta_base_logits = meta_base_output.logits
                            # self.dsrm_loss_function =  torch.nn.CrossEntropyLoss(reduction='none')
                            dsrm_meta_losses = self.dsrm_loss_function(meta_base_logits,dsrm_targets.view(-1)) #, golds.view(-1))
                            # torch.nn.init.constant_(self.dsrm_weights.weight, 1)
                            # self.dsrm_weights = torch.nn.Linear(self.training_args.per_device_train_batch_size, 1, bias=False)
                            
                            dsrm_loss = fmodel.dsrm_weights(dsrm_meta_losses) / self.training_args.per_device_train_batch_size
                            
                            diffopt.step(dsrm_loss)
                            # print ('dsrm_losses',dsrm_loss) 
                            
                            meta_val_output = fmodel(**dsrm_val_kargs_base )
                            meta_val_logits = meta_val_output.logits
                            dsrm_val_losses = self.dsrm_loss_function(meta_val_logits,dsrm_val_targets.view(-1))
                            dsrm_val_loss  = torch.mean(dsrm_val_losses)
                            # print ('dsrm_val_loss base',dsrm_val_loss)

                            # print ('dsrm_val_losses',dsrm_val_loss)
                            

                            paras = fmodel.parameters(time=0)
                            for para in paras:
                                pass
                            weight_grads = torch.autograd.grad(dsrm_val_loss, para)[-1]
                            # print ('weight_grads',weight_grads) 
                        #distribution shift
                        weight_grads = 1e-8 + weight_grads / len(weight_grads)
                        # print ('components',(loss_clamp - torch.mean(dsrm_losses)),torch.mean(dsrm_losses),torch.matmul(weight_grads.unsqueeze(1), dsrm_losses))
                
                        lam = (loss_clamp - torch.mean(dsrm_losses)) / torch.matmul(weight_grads.unsqueeze(1), dsrm_losses)
                        # print ('weight_grads',weight_grads)
                        # print ('lam',lam)

                        w = torch.clamp(lam * weight_grads * self.training_args.per_device_train_batch_size + 1, -30, 30)
                        # print ('w',w)
                        da_loss = dsrm_val_loss.detach().cpu().item() 
                        
                    else:
                        # print ('not shifting',torch.mean(dsrm_losses) ,'<', loss_clamp)

                        w = torch.ones(self.training_args.per_device_train_batch_size).to(textattack.shared.utils.device)
                    
                    print ('w',w)
                    dsrm_loss_final = torch.mean(dsrm_losses * w.detach())

                    # print ('dsrm_loss_final',dsrm_loss_final)
                    loss += dsrm_loss_final
                    s_loss = dsrm_loss_final.detach().cpu().item()
                    # da_loss = dsrm_val_loss.detach().cpu().item() 
                    # print ('dsrm standard loss', dsrm_loss_final)
                    # print ('loss',loss)
                    self.avg_loss.update(s_loss)
                    mmd_loss = self.avg_loss.get_metric()[0]
                    # loss.backward()
                    # sys.exit()
                    print (f'epoch: {0:d}, ',
                                f'avg_loss[0]: {self.avg_loss.get_metric()[0]:0.4f}, avg_loss[1]: {self.avg_loss.get_metric()[1]:0.4f}, ',
                                f'loss: {s_loss:0.3f}, ',
                                f'valid loss: {dsrm_val_loss:0.3f} ',)
                    # sys.exit()
                    


            if self.training_args.FLB_Val > 0:

                word_embedding_layer = model.get_input_embeddings()
                    
                # print ('word emb layer',word_embedding_layer)

                input_texts = input_texts



                flb_input_ids = tokenizer(
                    input_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                )

                 

                flb_input_tokens = flb_input_ids['input_ids'].to(textattack.shared.utils.device) # adv_input_ids['input_ids']
                flb_attention_mask = flb_input_ids['attention_mask'].to(textattack.shared.utils.device)# adv_input_ids['attention_mask']
                if 'token_type_ids' in flb_input_ids:
                    flb_token_type_ids  = flb_input_ids['token_type_ids'].to(textattack.shared.utils.device)
                ascc_targets = targets

                
                

                def delta_initial(args, embedding: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                    if args.adv_init_mag > 0:
                        input_mask = attention_mask.to(embedding) 
                        
                        input_lengths = torch.sum(input_mask, 1) 
                        if args.adv_norm_type == 'l2':
                                
                                
                            delta = torch.zeros_like(embedding).uniform_(-1, 1) * input_mask.unsqueeze(2)
                                
                            dims = input_lengths * embedding.size(-1)
                            magnitude = args.adv_init_mag / torch.sqrt(dims)
                            delta = (delta * magnitude.view(-1, 1, 1).detach())
                        elif args.adv_norm_type == 'linf':
                            delta = torch.zeros_like(embedding).uniform_(-args.adv_init_mag,
                                                                        args.adv_init_mag) * input_mask.unsqueeze(2)
                    else:
                        delta = torch.zeros_like(embedding)
                    return delta

                
                def delta_update(args, embedding: torch.Tensor, delta: torch.Tensor, delta_grad: torch.Tensor) -> torch.Tensor:

                    if args.adv_norm_type == "l2":
                        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                        denorm = torch.clamp(denorm, min=1e-8)
                        delta = (delta + args.adv_learning_rate * delta_grad / denorm).detach()
                        if args.adv_max_norm > 0:
                            delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                            exceed_mask = (delta_norm > args.adv_max_norm).to(embedding)
                            reweights = (args.adv_max_norm / delta_norm * exceed_mask
                                        + (1 - exceed_mask)).view(-1, 1, 1)
                            delta = (delta * reweights).detach()
                    elif args.adv_norm_type == "linf":
                        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                        denorm = torch.clamp(denorm, min=1e-8)
                        delta = (delta + args.adv_learning_rate * delta_grad / denorm).detach()
                        if args.adv_max_norm > 0:
                            delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()
                    else:
                        print("Norm type {} not specified.".format(args.adv_norm_type))
                        exit()
                    return delta
                

                embedding_init = word_embedding_layer(flb_input_tokens)

                # embedding_init= torch.tensor(embedding_init)
                # embedding_init = torch.round(embedding_init * 10000) / 10000
                # embedding_init = embedding_init.to(torch.float32)

                embedding_init = torch.round(embedding_init, decimals=4  )  

                # args = {'adv_norm_type':'l2','adv_learning_rate':0.03,'adv_max_norm':0.0,'adv_init_mag':0.05}

                class FLB_Args:
                    def __init__(self,dataset):
                        self.adv_norm_type = 'l2'
                        self.adv_learning_rate = 0.03
                        self.adv_max_norm = 0.0
                        self.adv_init_mag = 0.05
                        if dataset == 'IMDB':
                            self.adv_steps = 10
                        else:
                            self.adv_steps = 30
                        self.max_grad_norm = 1.0

                flb_args = FLB_Args(dataset = self.training_args.Dataset_attack) 
                
                delta = delta_initial(flb_args, embedding_init, flb_attention_mask)


                # for distance calculation, do inferene on base samples first
                flb_kargs_base = {'inputs_embeds':embedding_init,'attention_mask': flb_attention_mask}
                if 'token_type_ids' in flb_input_ids:
                    flb_kargs_base['token_type_ids'] = flb_token_type_ids
                    
                
                
                flb_outputs_base = model(**flb_kargs_base)
                
                flb_logits_base = flb_outputs_base[0]
                flb_pooler_output = flb_outputs_base.pooler_output
                flb_pooler_output_base_g_xs_mb = flb_pooler_output 
                 


                

                
                    
                for astep in range(flb_args.adv_steps):
                    delta.requires_grad_()
                    # print  ('delta req sh',delta.shape)
                    flb_kargs = {'inputs_embeds':embedding_init + delta,'attention_mask': flb_attention_mask}
                    if 'token_type_ids' in flb_input_ids:
                        flb_kargs['token_type_ids'] = flb_token_type_ids


                        
                    
                    
                    flb_outputs = model(**flb_kargs)
                    
                    # flb_outputs = model(
                    #     inputs_embeds= delta + embedding_init,
                    #     attention_mask=flb_adv_input_ids[1],
                    #     token_type_ids=flb_adv_input_ids[2],
                    # )

                    flb_logits = flb_outputs[0]
                    flb_pooler_output_adv_g_xt_mb = flb_outputs.pooler_output
                    
                    


                    self.flb_loss_function =  torch.nn.CrossEntropyLoss(reduction='none')
                    # flb_losses = self.flb_loss_function(flb_logits,flb_targets.view(-1)) #, golds.view(-1))
                    flb_losses = self.flb_loss_function(flb_logits,targets.view(-1)) #, golds.view(-1))
                    
                    flb_loss = torch.mean(flb_losses)
                    flb_loss = flb_loss / flb_args.adv_steps
                    flb_loss = self.training_args.FLB_Val*flb_loss 
                    total_flb_loss += flb_loss
                    
                    # loss += flb_loss
                    
                    flb_loss.backward() 
                    print(f"loss: {flb_loss.item()}")
                    if astep == flb_args.adv_steps - 1:
                        break

                    # (2) get gradient on delta
                    delta_grad = delta.grad.clone().detach()

                    # print ('delta grad shpae',delta_grad.shape)

                    # (3) update and clip
                    delta = delta_update(flb_args, embedding_init, delta, delta_grad)

                    # print ('new delta', delta.shape)
                    embedding_init = word_embedding_layer(flb_input_tokens)

                    # print ('new emb int',embedding_init.shape)  
                torch.nn.utils.clip_grad_norm_(model.parameters(), flb_args.max_grad_norm)
                total_flb_loss = total_flb_loss.detach().cpu().item()
                time_record_start =  time.time() # Optimal transport and mmd for flb only done for evaluation purposes so removing their computation time
                geom_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
                da_loss = geom_loss(flb_pooler_output_base_g_xs_mb,flb_pooler_output_adv_g_xt_mb)
                mmd_function = MMD_loss()
                mmd_loss = mmd_function(flb_pooler_output_base_g_xs_mb,flb_pooler_output_adv_g_xt_mb)
                time_record_end = time.time() - time_record_start
                time_record += time_record_end
                da_loss = da_loss.detach().cpu().item()
                mmd_loss = mmd_loss.detach().cpu().item()
            # print ('da loss',da_loss) 


                

            if adv_batch:
                g_xt_mb = adv_pooler_output # target (adv) pooler
                # g_xt_mb = adv_last_hidden_state
                
                f_g_xt_mb = adv_logits # logits of the base
                pred_xt = F.softmax(f_g_xt_mb, 1)

                self.loss_fct_adv = torch.nn.CrossEntropyLoss(reduction="mean")
                adv_loss = self.loss_fct_adv(f_g_xt_mb, adv_targets)
                 


                time_record_start =  time.time()
                geom_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

                da_loss = geom_loss(g_xs_mb,g_xt_mb)
                # self.training_args.FLB_Val = 1
                time_record_end =  time.time() - time_record_start 
                if self.training_args.OT_Val == 0:
                    # if OT is 0, we compute ot loss for evaluation purposes so remove time it take to compute
                    time_record += time_record_end
                
                print ('teims',time_record_start,time_record_end,time_record )
                 
                    
                
                


                #
                if self.training_args.CRL_Val > 0:
                    # print ('input shape',g_xs_mb.shape,g_xt_mb.shape )
                    # import time
                    # t = time.time()
                    crl_loss = coral(g_xs_mb,g_xt_mb)
                    # print ('time corak',time.time() - t)
                    # print ('mmd loss',mmd_loss,mmd_loss*self.training_args.MMD_Val)
                    loss +=  (self.training_args.CRL_Val)*crl_loss
                    # print ('loss_total',mmd_loss)
                    # sys.exit()
                    crl_loss = crl_loss.detach().cpu().item()

                # print ('crl loss',crl_loss)
                # sys.exit()
                if self.training_args.Dist == 'L2':
                    l2_function = L2_loss() 
                    l2_loss = l2_function(g_xs_mb,g_xt_mb)
                    # mmd_function = MMD_loss()
                    # mmd_loss = mmd_function(g_xs_mb,g_xt_mb)

                     
                    loss +=  (self.training_args.Dist_Val)*l2_loss
                    l2_loss = l2_loss.detach().cpu().item()

                if self.training_args.MMD_Val > 0:
                    mmd_function = MMD_loss()
                    mmd_loss = mmd_function(g_xs_mb,g_xt_mb)
                    
                    loss +=  (self.training_args.MMD_Val)*mmd_loss
                    
                    mmd_loss = mmd_loss.detach().cpu().item()


                
                loss +=  (self.training_args.AT_Val)*adv_loss
                loss +=  (self.training_args.OT_Val)*da_loss
                 
                da_loss =da_loss.detach().cpu().item()
                adv_loss = adv_loss.detach().cpu().item()

                
                
                # mmd_loss = mmd_loss.detach().cpu().item()
                # crl_loss = crl_loss.detach().cpu().item()

            if progress_batch:
                g_xt_mb_progress = progress_pooler_output # target (adv) pooler
                # g_xt_mb = adv_last_hidden_state

                f_g_xt_mb_progress = progress_logits # logits of the base
                pred_xt_progress = F.softmax(f_g_xt_mb_progress, 1)

                # prev AT  loss
                # self.loss_fct_adv = torch.nn.CrossEntropyLoss(reduction="mean")
                # adv_loss = self.loss_fct_adv(f_g_xt_mb, adv_targets)

                # prev OT loss
                # geom_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

                # da_loss = geom_loss(g_xs_mb,g_xt_mb)




                #
                if self.training_args.CRL_Val > 0:
                    # print ('input shape',g_xs_mb.shape,g_xt_mb.shape )
                    # import time
                    # t = time.time()
                    crl_loss = coral(g_xs_mb,g_xt_mb)
                    # print ('time corak',time.time() - t)
                    # print ('mmd loss',mmd_loss,mmd_loss*self.training_args.MMD_Val)
                    loss +=  (self.training_args.CRL_Val)*crl_loss
                    # print ('loss_total',mmd_loss)
                    # sys.exit()
                    crl_loss = crl_loss.detach().cpu().item()

                # print ('crl loss',crl_loss)
                # sys.exit()


                if self.training_args.P_Val > 0:
                    mmd_function = MMD_loss()
                    mmd_progress_loss = mmd_function(g_xs_mb,g_xt_mb_progress)
                    # print ('1st method',mmd_loss)
                    loss +=  (self.training_args.P_Val)*mmd_progress_loss
                    # sys.exit()
                    # mmd_loss2 = MMD(g_xs_mb,g_xt_mb_progress,kernel="multiscale")
                    # print ('2nd method',mmd_loss2)
                    # loss +=  (self.training_args.MMD_Val)*mmd_loss2
                    mmd_progress_loss = mmd_progress_loss.detach().cpu().item()


                # loss +=  (self.training_args.AT_Val)*adv_loss
                # loss +=  (self.training_args.OT_Val)*da_loss
                # da_loss =da_loss.detach().cpu().item()
                # adv_loss = adv_loss.detach().cpu().item()

            if progress_extra_batch:
                g_xt_mb_progress_extra = progress_extra_pooler_output # target (adv) pooler
                # g_xt_mb = adv_last_hidden_state

                f_g_xt_mb_progress_extra = progress_extra_logits # logits of the base
                pred_xt_progress_extra = F.softmax(f_g_xt_mb_progress_extra, 1)

                # prev AT  loss
                # self.loss_fct_adv = torch.nn.CrossEntropyLoss(reduction="mean")
                # adv_loss = self.loss_fct_adv(f_g_xt_mb, adv_targets)

                # prev OT loss
                # geom_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

                # da_loss = geom_loss(g_xs_mb,g_xt_mb)




                #
                if self.training_args.CRL_Val > 0:
                    # print ('input shape',g_xs_mb.shape,g_xt_mb.shape )
                    # import time
                    # t = time.time()
                    crl_progress_extra_loss = coral(g_xs_mb,g_xt_mb_progress_extra)
                    # print ('time corak',time.time() - t)
                    # print ('mmd loss',mmd_loss,mmd_loss*self.training_args.MMD_Val)
                    loss +=  (self.training_args.PF_Val)*crl_progress_extra_loss
                    # print ('loss_total',mmd_loss)
                    # sys.exit()
                    crl_progress_extra_loss = crl_progress_extra_loss.detach().cpu().item()

                # print ('crl loss',crl_loss)
                # sys.exit()


                if self.training_args.PF_Val > 0:
                    mmd_function = MMD_loss()
                    mmd_progress_extra_loss = mmd_function(g_xs_mb,g_xt_mb_progress_extra)
                    # print ('1st method',mmd_loss)
                    loss +=  (self.training_args.PF_Val)*mmd_progress_extra_loss
                    # sys.exit()
                    # mmd_loss2 = MMD(g_xs_mb,g_xt_mb_progress_extra,kernel="multiscale")
                    # print ('2nd method',mmd_loss2)
                    # loss +=  (self.training_args.MMD_Val)*mmd_loss2
                    mmd_progress_extra_loss = mmd_progress_extra_loss.detach().cpu().item()


                # loss +=  (self.training_args.AT_Val)*adv_loss
                # loss +=  (self.training_args.OT_Val)*da_loss
                # da_loss =da_loss.detach().cpu().item()
                # adv_loss = adv_loss.detach().cpu().item()





            preds = logits.argmax(dim=-1)
            
        elif self.task_type == "OT_GL_CC":

            ys = targets

            g_xs_mb = pooler_output # source (base) pooler
            f_g_xs_mb = logits # logits of source


            self.loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            s_loss = self.loss_fct(logits, targets) # used to be on target

            loss = (self.training_args.B_Val)*s_loss


            if adv_batch:
                g_xt_mb = adv_pooler_output # target (adv) pooler
                f_g_xt_mb = adv_logits # logits of the base
                pred_xt = F.softmax(f_g_xt_mb, 1)

                self.loss_fct_adv = torch.nn.CrossEntropyLoss(reduction="mean")
                adv_loss = self.loss_fct_adv(f_g_xt_mb, adv_targets)



                geom_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

                da_loss = geom_loss(g_xs_mb,g_xt_mb)
                ###NEEDCHANGE###


                loss +=  (self.training_args.AT_Val)*adv_loss
                loss +=  (self.training_args.OT_Val)*da_loss

                da_loss =da_loss.detach().cpu().item()
                adv_loss = adv_loss.detach().cpu().item()
                s_loss= s_loss.detach().cpu().item()
                


                # deviceCount = nvidia_smi.nvmlDeviceGetCount()
                # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(3)
                # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                # print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(3, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))




                if self.task_type == "OT_GL_CC":
                    pos_base_input_ids = tokenizer(
                        pos_base_input_texts,
                        padding="max_length",
                        return_tensors="pt",
                        truncation=True,
                    )
                    pos_base_input_ids.to(textattack.shared.utils.device)
                    pos_base_output = model(**pos_base_input_ids ,return_dict=True)
                    pos_base_logits = pos_base_output[0]
                    pos_base_pooler_output = pos_base_output.pooler_output

                    # deviceCount = nvidia_smi.nvmlDeviceGetCount()
                    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(3)
                    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    # print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(3, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

                    pos_adv_input_ids = tokenizer(
                        pos_adv_input_texts,
                        padding="max_length",
                        return_tensors="pt",
                        truncation=True,
                    )
                    pos_adv_input_ids.to(textattack.shared.utils.device)
                    pos_adv_output = model(**pos_adv_input_ids ,return_dict=True)
                    pos_adv_logits = pos_adv_output[0]
                    pos_adv_pooler_output = pos_adv_output.pooler_output

                    pos_base_g_xt_mb = pos_base_pooler_output
                    pos_adv_g_xt_mb = pos_adv_pooler_output
                    pos_adv_da_loss = geom_loss(pos_base_g_xt_mb,pos_adv_g_xt_mb)
                    loss+=(self.training_args.OT_Val)*pos_adv_da_loss
                    pos_adv_da_loss =pos_adv_da_loss.detach().cpu().item()


                    # deviceCount = nvidia_smi.nvmlDeviceGetCount()
                    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(3)
                    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    # print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(3, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))



                    neg_adv_input_ids = tokenizer(
                        neg_adv_input_texts,
                        padding="max_length",
                        return_tensors="pt",
                        truncation=True,
                    )
                    neg_adv_input_ids.to(textattack.shared.utils.device)
                    neg_adv_output = model(**neg_adv_input_ids ,return_dict=True)
                    neg_adv_logits = neg_adv_output[0]
                    neg_adv_pooler_output = neg_adv_output.pooler_output


                    # deviceCount = nvidia_smi.nvmlDeviceGetCount()
                    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(3)
                    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    # print("neg adv Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(3, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

                    neg_base_input_ids = tokenizer(
                        neg_base_input_texts,
                        padding="max_length",
                        return_tensors="pt",
                        truncation=True,
                    )
                    neg_base_input_ids.to(textattack.shared.utils.device)
                    neg_base_output = model(**neg_base_input_ids ,return_dict=True)
                    neg_base_logits = neg_base_output[0]
                    neg_base_pooler_output = neg_base_output.pooler_output

                    # deviceCount = nvidia_smi.nvmlDeviceGetCount()
                    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(3)
                    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    # print(" neg baseDevice {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(3, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))


                    neg_base_g_xt_mb = neg_base_pooler_output
                    neg_adv_g_xt_mb = neg_adv_pooler_output
                    neg_adv_da_loss = geom_loss(neg_base_g_xt_mb,neg_adv_g_xt_mb)
                    loss +=(self.training_args.OT_Val)*neg_adv_da_loss
                    neg_adv_da_loss = neg_adv_da_loss.detach().cpu().item()




            preds = logits.argmax(dim=-1)


        else:
            loss = self.loss_fct(logits, targets)
            preds = logits.argmax(dim=-1)

        sample_weights = torch.ones(
            is_adv_sample.size(), device=textattack.shared.utils.device
        )
        sample_weights[is_adv_sample] *= self.training_args.alpha
        loss = loss * sample_weights
        loss = torch.mean(loss)
        preds = preds.cpu()

        # print ('s da loss',s_loss.detach().cpu().item(), da_loss.detach().cpu().item())
        # return (loss, preds, _targets,(s_loss , da_loss,adv_loss,mmd_loss,crl_loss,pos_adv_da_loss,neg_adv_da_loss )) if s_loss is not None else (loss, preds, _targets,None)
        
        training_return_dic = { 'time_record':time_record}


        return (loss, preds, _targets,(s_loss , da_loss,adv_loss,mmd_loss,l2_loss,mmd_progress_extra_loss,crl_loss,total_flb_loss,pos_adv_da_loss,neg_adv_da_loss ),training_return_dic) if s_loss is not None else (loss, preds, _targets,None)

        #
        # if s_loss and da_loss:
        #     return  loss, preds, _targets, (s_loss, da_loss)
        # else:
        #     return loss, preds, _targets

    def evaluate_step(self, model, tokenizer, batch,adv_batch):
        """Perform a single evaluation step on a batch of inputs.

        Args:
            model (:obj:`torch.nn.Module`):
                Model to train.
            tokenizer:
                Tokenizer used to tokenize input text.
            batch (:obj:`tuple[list[str], torch.Tensor]`):
                By default, this will be a tuple of input texts and target tensors.

                .. note::
                    If you override the :meth:`get_eval_dataloader` method, then shape/type of :obj:`batch` will depend on how you created your batch.

        Returns:
            :obj:`tuple[torch.Tensor, torch.Tensor]` where

            - **preds**: :obj:`torch.FloatTensor` of model's prediction for the batch.
            - **targets**: :obj:`torch.Tensor` of model's targets (e.g. labels, target values).
        """
        input_texts, targets = batch
        _targets = targets
        targets = targets.to(textattack.shared.utils.device)

        if adv_batch:
            adv_input_texts, adv_targets, adv_is_adv_sample = adv_batch
            _adv_targets = adv_targets
            adv_targets = adv_targets.to(textattack.shared.utils.device)

        if isinstance(model, transformers.PreTrainedModel):
            input_ids = tokenizer(
                input_texts,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            input_ids.to(textattack.shared.utils.device)
            # logits = model(**input_ids)[0]

            output = model(**input_ids,return_dict=True,output_hidden_states=True)


            logits = output[0]
            if 'Embedding' in self.training_args.Method_Dictionary:
                
                pooler_output = None #output.pooler_output
            else:
                pooler_output = output.pooler_output
            # print ('pooler out:',pooler_output[0][:5])
            hidden_states = output.hidden_states
            # base_last_hidden_state = hidden_states[-1]
            # base_last_hidden_state = base_last_hidden_state[::,0,::]



            if adv_batch:
                adv_input_ids = tokenizer(
                    adv_input_texts,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )
                adv_input_ids.to(textattack.shared.utils.device)

                adv_output = model(**adv_input_ids ,return_dict=True,output_hidden_states=True)

                adv_logits = adv_output[0]
                adv_pooler_output = adv_output.pooler_output

                adv_hidden_states = adv_output.hidden_states
                adv_last_hidden_state = adv_hidden_states[-1]
                adv_last_hidden_state = adv_last_hidden_state[::,0,::]

        else:
            input_ids = tokenizer(input_texts)
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            input_ids = input_ids.to(textattack.shared.utils.device)
            logits = model(input_ids)

        s_loss = 0
        da_loss = 0
        adv_loss= 0
        mmd_loss = 0
        crl_loss = 0
        pos_adv_da_loss = 0
        neg_adv_da_loss = 0

        if self.task_type == "regression":
            preds = logits
            loss = 0
        elif self.task_type == 'OT_GL':

            ys = targets

            # ys = adv_targets# this 100% returns the GRAUND TRUTH LABEL AS SEEN FROM ORGINAL DATASET
            # to get the adv label we have to invert them https://huggingface.co/datasets/glue/viewer/mrpc/train to see ground truth labels, print ys and adv_input_texts


            g_xs_mb = pooler_output # source (base) pooler
            # g_xs_mb = base_last_hidden_state


            f_g_xs_mb = logits # logits of source


            # g_xt_mb = adv_pooler_output # target (adv) pooler
            # f_g_xt_mb = adv_logits # logits of the base
            # pred_xt = F.softmax(f_g_xt_mb, 1)

            self.loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            s_loss = self.loss_fct(logits, targets) # used to be on target
            loss = (self.training_args.B_Val)*s_loss



            if adv_batch:
                g_xt_mb = adv_pooler_output # target (adv) pooler
                # g_xt_mb = adv_last_hidden_state

                f_g_xt_mb = adv_logits # logits of the base
                pred_xt = F.softmax(f_g_xt_mb, 1)

                self.loss_fct_adv = torch.nn.CrossEntropyLoss(reduction="mean")
                adv_loss = self.loss_fct_adv(f_g_xt_mb, adv_targets)



                geom_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

                da_loss = geom_loss(g_xs_mb,g_xt_mb)

                ### new mmd
                class MMD_loss(torch.nn.Module):
                    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
                        super(MMD_loss, self).__init__()
                        self.kernel_num = kernel_num
                        self.kernel_mul = kernel_mul
                        self.fix_sigma = None
                        self.kernel_type = kernel_type

                    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
                        n_samples = int(source.size()[0]) + int(target.size()[0])
                        total = torch.cat([source, target], dim=0)
                        total0 = total.unsqueeze(0).expand(
                            int(total.size(0)), int(total.size(0)), int(total.size(1)))
                        total1 = total.unsqueeze(1).expand(
                            int(total.size(0)), int(total.size(0)), int(total.size(1)))
                        L2_distance = ((total0-total1)**2).sum(2)
                        if fix_sigma:
                            bandwidth = fix_sigma
                        else:
                            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
                        bandwidth /= kernel_mul ** (kernel_num // 2)
                        bandwidth_list = [bandwidth * (kernel_mul**i)
                                          for i in range(kernel_num)]
                        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                                      for bandwidth_temp in bandwidth_list]
                        return sum(kernel_val)

                    def linear_mmd2(self, f_of_X, f_of_Y):
                        loss = 0.0
                        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
                        loss = delta.dot(delta.T)
                        return loss

                    def forward(self, source, target):
                        if self.kernel_type == 'linear':
                            return self.linear_mmd2(source, target)
                        elif self.kernel_type == 'rbf':
                            batch_size = int(source.size()[0])
                            kernels = self.guassian_kernel(
                                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
                            XX = torch.mean(kernels[:batch_size, :batch_size])
                            YY = torch.mean(kernels[batch_size:, batch_size:])
                            XY = torch.mean(kernels[:batch_size, batch_size:])
                            YX = torch.mean(kernels[batch_size:, :batch_size])
                            loss = torch.mean(XX + YY - XY - YX)
                            return loss




                def coral(source, target):

                    d = source.size(1)  # dim vector

                    source_c = compute_covariance(source)
                    target_c = compute_covariance(target)

                    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

                    loss = loss / (4 * d * d)
                    return loss


                def compute_covariance(input_data):
                    """
                    Compute Covariance matrix of the input data
                    """
                    n = input_data.size(0)  # batch_size

                    # Check if using gpu or cpu
                    # if input_data.is_cuda:
                    #     device = torch.device('cuda')
                    # else:
                    #     device = torch.device('cpu')


                    id_row = torch.ones(n).resize(1, n).to(device=textattack.shared.utils.device)
                    sum_column = torch.mm(id_row, input_data)
                    mean_column = torch.div(sum_column, n)
                    term_mul_2 = torch.mm(mean_column.t(), mean_column)
                    d_t_d = torch.mm(input_data.t(), input_data)
                    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

                    return c

                #mmd loss



                #
                # if self.training_args.CRL_Val > 0:
                    # print ('input shape',g_xs_mb.shape,g_xt_mb.shape )
                    # import time
                    # t = time.time()
                crl_loss = coral(g_xs_mb,g_xt_mb)
                loss +=  (self.training_args.CRL_Val)*crl_loss
                crl_loss = crl_loss.detach().cpu().item()

                # print ('crl loss',crl_loss)
                # sys.exit()


                # if self.training_args.MMD_Val > 0:
                mmd_function = MMD_loss()
                mmd_loss = mmd_function(g_xs_mb,g_xt_mb)
                loss +=  (self.training_args.MMD_Val)*mmd_loss
                mmd_loss = mmd_loss.detach().cpu().item()

                loss +=  (self.training_args.AT_Val)*adv_loss
                loss +=  (self.training_args.OT_Val)*da_loss
                da_loss =da_loss.detach().cpu().item()
                adv_loss = adv_loss.detach().cpu().item()
                # mmd_loss = mmd_loss.detach().cpu().item()
                # crl_loss = crl_loss.detach().cpu().item()

            preds = logits.argmax(dim=-1)
            s_loss= s_loss.detach().cpu().item()
        else:
            loss = 0


        # sample_weights = torch.ones(
        #     is_adv_sample.size(), device=textattack.shared.utils.device
        # )
        # sample_weights[is_adv_sample] *= self.training_args.alpha
        loss = loss # * sample_weights
        # print ('loss2',loss)
        loss = torch.mean(loss)
        # print ('loss3',loss)
        loss = loss.detach().cpu().item()
        preds = preds.cpu()
        # print (' da loss',s_loss.detach().cpu().item(), da_loss.detach().cpu().item())
        # print ('losses',loss, (s_loss , da_loss,adv_loss,mmd_loss,crl_loss,pos_adv_da_loss,neg_adv_da_loss ))

        return (loss, preds, _targets,(s_loss , da_loss,adv_loss,mmd_loss,crl_loss,pos_adv_da_loss,neg_adv_da_loss )) if s_loss is not None else (loss, preds, _targets,None)

        # return s_loss,preds.cpu(), _targets

    def train(self):
        """Train the model on given training dataset."""
        if not self.train_dataset:
            raise ValueError("No `train_dataset` available for training.")

        textattack.shared.utils.set_seed(self.training_args.random_seed)
        if not os.path.exists(self.training_args.output_dir):
            os.makedirs(self.training_args.output_dir)

        # Save logger writes to file
        log_txt_path = os.path.join(self.training_args.output_dir, "train_log.txt")
        fh = logging.FileHandler(log_txt_path)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.info(f"Writing logs to {log_txt_path}.")

        # Save original self.training_args to file
        args_save_path = os.path.join(
            self.training_args.output_dir, "training_args.json"
        )
        with open(args_save_path, "w", encoding="utf-8") as f:
            json.dump(self.training_args.__dict__, f)

        # writer_output_folder = os.path.join(self.training_args.output_dir, f"OT_GL_")
        #
        # writer = SummaryWriter(log_dir = writer_output_folder)

        logger.info(f"Wrote original training args and tensorboard to {args_save_path}.")

        num_gpus = torch.cuda.device_count()
        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper.model

        if self.training_args.parallel and num_gpus > 1:
            # TODO: torch.nn.parallel.DistributedDataParallel
            # Supposedly faster than DataParallel, but requires more work to setup properly.
            model = torch.nn.DataParallel(model)
            logger.info(f"Training on {num_gpus} GPUs via `torch.nn.DataParallel`.")
            train_batch_size = self.training_args.per_device_train_batch_size * num_gpus
        else:
            train_batch_size = self.training_args.per_device_train_batch_size

        if self.attack is None:
            num_clean_epochs = self.training_args.num_epochs
        else:
            num_clean_epochs = self.training_args.num_clean_epochs

        total_clean_training_steps = (
            math.ceil(
                len(self.train_dataset)
                / (train_batch_size * self.training_args.gradient_accumulation_steps)
            )
            * num_clean_epochs
        )

        # calculate total_adv_training_data_length based on type of
        # num_train_adv_examples.
        # if num_train_adv_examples is float , num_train_adv_examples is a portion of train_dataset.
        if isinstance(self.training_args.num_train_adv_examples, float):
            total_adv_training_data_length = (
                len(self.train_dataset) * self.training_args.num_train_adv_examples
            )

        # if num_train_adv_examples is int and >=0 then it is taken as value.
        elif (
            isinstance(self.training_args.num_train_adv_examples, int)
            and self.training_args.num_train_adv_examples >= 0
        ):
            total_adv_training_data_length = self.training_args.num_train_adv_examples

        # if num_train_adv_examples is = -1 , we generate all possible adv examples.
        # Max number of all possible adv examples would be equal to train_dataset.
        else:
            total_adv_training_data_length = len(self.train_dataset)

        # Based on total_adv_training_data_length calculation , find total total_adv_training_steps
        total_adv_training_steps = math.ceil(
            (len(self.train_dataset) + total_adv_training_data_length)
            / (train_batch_size * self.training_args.gradient_accumulation_steps)
        ) * (self.training_args.num_epochs - num_clean_epochs)

        total_training_steps = total_clean_training_steps + total_adv_training_steps

        optimizer, scheduler = self.get_optimizer_and_scheduler(
            model, total_training_steps
        )

        self.optimizer_training_step = optimizer

       
        
        
        self._print_training_args(
            total_training_steps, train_batch_size, num_clean_epochs
        )

        model.to(textattack.shared.utils.device)

        # Variables across epochs
        self._total_loss = 0.0
        self._current_loss = 0.0
        self._last_log_step = 0

        # Variables across epochs eval
        self._total_loss_eval = 0.0
        self._current_loss_eval = 0.0
        self._last_log_step_eval = 0

        # `best_score` is used to keep track of the best model across training.
        # Could be loss, accuracy, or other metrics.
        best_eval_score = 0.0
        best_eval_score_epoch = 0
        best_model_path = None
        epochs_since_best_eval_score = 0

        # NEEDCHANGE#
        # epoch = 1
        # if self.attack and epoch > num_clean_epochs:
        #     if (
        #         epoch - num_clean_epochs - 1
        #     ) % self.training_args.attack_epoch_interval == 0:
        #         # only generate a new adversarial training set every self.training_args.attack_period epochs after the clean epochs
        #         # adv_dataset is instance of `textattack.datasets.Dataset`
        #         model.eval()
        #         adv_dataset = self._generate_adversarial_examples(epoch)
        #         model.train()
        #         model.to(textattack.shared.utils.device)
        #     else:
        #         adv_dataset = None
        # else:
        #     logger.info(f"Running clean epoch {epoch}/{num_clean_epochs}")
        #     adv_dataset = None

        # print ('input cols',self.train_dataset.input_columns)
        # self.train_dataset2 = torch.utils.data.Subset(self.train_dataset,[3394,3395])
        # self.train_dataset3 = textattack.datasets.Dataset(self.train_dataset2, input_columns=('sentence1','sentence2'))
        # self.train_dataset4  = datasets.Dataset(self.train_dataset3 )
        #
        # self.train_dataset = textattack.datasets.HuggingFaceDataset(self.train_dataset4, None, split="train",dataset_columns=(('premise', 'hypothesis'), 'label'),label_map=self.train_dataset.label_map, label_names=self.train_dataset.label_names)

        # for i in self.train_dataset :
        #     print (i)


        if  self.task_type == 'OT_GL_CC':

            pos_label_dataset_base = copy.deepcopy(self.train_dataset)
            pos_label_dataset_base.filter_by_labels_([1])

            pos_label_dataset_base_dataloader = self.get_train_dataloader(
                pos_label_dataset_base,None,train_batch_size
            )

            pos_label_dataset_base_cycle = itertools.cycle(pos_label_dataset_base_dataloader)

            neg_label_dataset_base = copy.deepcopy(self.train_dataset)
            neg_label_dataset_base.filter_by_labels_([0])

            neg_label_dataset_base_dataloader = self.get_train_dataloader(
                neg_label_dataset_base,None,train_batch_size
            )

            neg_label_dataset_base_cycle = itertools.cycle(neg_label_dataset_base_dataloader)


        # first thing you can do is, do an evaluation step at base model, this can also be used as baseline
        # it will generate adv samples on clean model which can be later used for evaluation

        # create epoch 0 eval dataset
        # if self.attack:# and epoch > num_clean_epochs and self.training_args.Data_ratio != 0:
            # try loading the path to evaluation dataset to use

        if self.training_args.debug:
            pass
        else:
            model.eval()
            my_file = Path(f"./caches/cache_evaluation_adv_samples_AKT{self.training_args.AttackTrain}_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}.pt")
            if my_file.is_file(): #test at each epoch #  and self.training_args.attack_epoch_interval == self.training_args.num_epochs:
                pass
                # file exists so load the cache
                # adv_dataset_eval = torch.load(f'./cache_adv_samples_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}.pt')
                # print ('lengh adv dataset', len(adv_dataset))
                # if len(adv_dataset_eval) < 100:
                #     raise ValueError("EVALUATION ADVERSARIAL DATASET IS SMALLER THAN 100 SAMPLES? CHECK THE CACHE!")
            else: # need to generate evaluation file
                adv_dataset_eval = self._generate_adversarial_examples_evaluation(0)#epoch) # generate evaluation files
                # save this new file as cache
                torch.save(adv_dataset_eval,f'./caches/cache_evaluation_adv_samples_AKT{self.training_args.AttackTrain}_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}.pt' )
        
        
        
        model.train()
        model.to(textattack.shared.utils.device)

        time_to_train = 0
        start_time_train = time.time()

        time_to_generate_peturbed_dataset = 0

        for epoch in range(1, self.training_args.num_epochs + 1):

            time_epoch_ignore = 0
            
            
            logger.info("==========================================================")
            logger.info(f"Epoch {epoch}")

            # NEEDCHANGE#
            # for offline training. pass a value that if epoch >
            print ('epoch interval',epoch - num_clean_epochs - 1, self.training_args.attack_epoch_interval,(  epoch - num_clean_epochs - 1 ) % self.training_args.attack_epoch_interval)
            
            if self.attack and epoch > num_clean_epochs and self.training_args.Data_ratio != 0:
                
                if (
                    epoch - num_clean_epochs - 1
                ) % self.training_args.attack_epoch_interval == 0:
                    # only generate a new adversarial training set every self.training_args.attack_period epochs after the clean epochs
                    # adv_dataset is instance of `textattack.datasets.Dataset`
                    model.eval()

                    my_file_path = Path(f"{self.training_args.cache_path}/caches/{self.training_args.model_name}/{self.training_args.Dataset_attack}/{self.training_args.AttackTrain}/epochs{ self.training_args.num_epochs}_intervals{self.training_args.attack_epoch_interval}/")
                     
                    if not os.path.exists(my_file_path):
                        os.makedirs(my_file_path)
                        

                    my_file = Path(f"{self.training_args.cache_path}/caches/{self.training_args.model_name}/{self.training_args.Dataset_attack}/{self.training_args.AttackTrain}/epochs{ self.training_args.num_epochs}_intervals{self.training_args.attack_epoch_interval}/cache_adv_samples_M{self.training_args.model_name}_AKT{self.training_args.AttackTrain}_MR{self.training_args.max_modification_rate_adv}_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}_AKT{epoch}.pt")
                    my_prog_file = Path(f"{self.training_args.cache_path}/caches/{self.training_args.model_name}/{self.training_args.Dataset_attack}/{self.training_args.AttackTrain}/epochs{ self.training_args.num_epochs}_intervals{self.training_args.attack_epoch_interval}/cache_adv_prog_samples_M{self.training_args.model_name}_AKT{self.training_args.AttackTrain}_MR{self.training_args.max_modification_rate_adv}_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}_AKT{epoch}.pt")
                    my_prog_extra_file = Path(f"{self.training_args.cache_path}/caches/{self.training_args.model_name}/{self.training_args.Dataset_attack}/{self.training_args.AttackTrain}/epochs{ self.training_args.num_epochs}_intervals{self.training_args.attack_epoch_interval}/cache_adv_prog_extra_samples_M{self.training_args.model_name}_AKT{self.training_args.AttackTrain}_MR{self.training_args.max_modification_rate_adv}_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}_AKT{epoch}.pt")
                    
                    
                    if my_file.is_file() and my_prog_file.is_file() and my_prog_extra_file.is_file() and self.training_args.attack_epoch_interval == self.training_args.num_epochs:
                        # if self.training_args.attack_epoch_interval == self.training_args.num_epochs :
                        #     print ('only 1 epoch')
                        adv_dataset = torch.load(f'{self.training_args.cache_path}/caches/{self.training_args.model_name}/{self.training_args.Dataset_attack}/{self.training_args.AttackTrain}/epochs{ self.training_args.num_epochs}_intervals{self.training_args.attack_epoch_interval}/cache_adv_samples_M{self.training_args.model_name}_AKT{self.training_args.AttackTrain}_MR{self.training_args.max_modification_rate_adv}_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}_AKT{epoch}.pt')
                        progress_dataset = torch.load(f'{self.training_args.cache_path}/caches/{self.training_args.model_name}/{self.training_args.Dataset_attack}/{self.training_args.AttackTrain}/epochs{ self.training_args.num_epochs}_intervals{self.training_args.attack_epoch_interval}/cache_adv_prog_samples_M{self.training_args.model_name}_AKT{self.training_args.AttackTrain}_MR{self.training_args.max_modification_rate_adv}_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}_AKT{epoch}.pt')
                        progress_extra_dataset = torch.load(f'{self.training_args.cache_path}/caches/{self.training_args.model_name}/{self.training_args.Dataset_attack}/{self.training_args.AttackTrain}/epochs{ self.training_args.num_epochs}_intervals{self.training_args.attack_epoch_interval}/cache_adv_prog_extra_samples_M{self.training_args.model_name}_AKT{self.training_args.AttackTrain}_MR{self.training_args.max_modification_rate_adv}_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}_AKT{epoch}.pt' )
                            # if file exists and
                            # if self.training_args.attack_epoch_interval == self.training_args.num_epochs :
                            # loaded_object = torch.load('./saved_object.pt')
                            # else
                            #

                        if len(adv_dataset) < 5:
                            raise ValueError("ADVERSARIAL DATASET IS SMALLER THAN 100 SAMPLES? CHECK THE CACHE!")

                    else:
                        time_to_generate_peturbed_dataset = 0
                        start_time_generation = time.time()
                        adv_dataset,progress_dataset,progress_extra_dataset = self._generate_adversarial_examples(epoch)
                        time_to_generate_peturbed_dataset = (time.time() - start_time_generation)
                        print ('time_to_generate_peturbed_dataset',time_to_generate_peturbed_dataset)
                        time_to_generate_peturbed_dataset_str = f"""\n(Time Take Generate Adv Dataset Epoch {epoch}) {time_to_generate_peturbed_dataset}
                        """
                        update_p_time = f'{self.training_args.output_dir}/P.txt'

                        file_object = open(update_p_time,'a')
                        file_object.write(str(time_to_generate_peturbed_dataset_str))
                        file_object.close()


                        # if self.training_args.attack_epoch_interval == self.training_args.num_epochs
                        # save (this is because we only need to save when we do offline training)
                        # with online training we delete the file and epoch interval != num epochs so it will never load and save

                        # if self.training_args.attack_epoch_interval == self.training_args.num_epochs:
                        torch.save(adv_dataset,f'{self.training_args.cache_path}/caches/{self.training_args.model_name}/{self.training_args.Dataset_attack}/{self.training_args.AttackTrain}/epochs{ self.training_args.num_epochs}_intervals{self.training_args.attack_epoch_interval}/cache_adv_samples_M{self.training_args.model_name}_AKT{self.training_args.AttackTrain}_MR{self.training_args.max_modification_rate_adv}_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}_AKT{epoch}.pt' )
                        torch.save(progress_dataset,f'{self.training_args.cache_path}/caches/{self.training_args.model_name}/{self.training_args.Dataset_attack}/{self.training_args.AttackTrain}/epochs{ self.training_args.num_epochs}_intervals{self.training_args.attack_epoch_interval}/cache_adv_prog_samples_M{self.training_args.model_name}_AKT{self.training_args.AttackTrain}_MR{self.training_args.max_modification_rate_adv}_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}_AKT{epoch}.pt' )
                        torch.save(progress_extra_dataset,f'{self.training_args.cache_path}/caches/{self.training_args.model_name}/{self.training_args.Dataset_attack}/{self.training_args.AttackTrain}/epochs{ self.training_args.num_epochs}_intervals{self.training_args.attack_epoch_interval}/cache_adv_prog_extra_samples_M{self.training_args.model_name}_AKT{self.training_args.AttackTrain}_MR{self.training_args.max_modification_rate_adv}_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}_AKT{epoch}.pt' )
                            # loaded_object = torch.load('./saved_object.pt')


                    model.train()
                    model.to(textattack.shared.utils.device)
                    
                else:
                    if self.training_args.attack_epoch_interval == self.training_args.num_epochs :
                        adv_dataset = adv_dataset
                    else:
                        adv_dataset = None

            else:
                logger.info(f"Running clean epoch {epoch}/{num_clean_epochs}")
                adv_dataset = None


            
            


            # implement balanced dataset in training and testing
            
            val_cycle_dataloader = None
            if self.training_args.AUG_Val == 'Adv':
                train_dataloader = self.get_train_dataloader(
                    self.train_dataset, adv_dataset, train_batch_size
                )
            elif self.training_args.AUG_Val == 'AdvP':
                train_dataloader = self.get_train_dataloader(
                    self.train_dataset, adv_dataset, train_batch_size,progress_dataset
                )
            elif 'Embedding' in self.training_args.Method_Dictionary:
                if self.training_args.Method_Dictionary['Embedding'] == 'DSRM': 
                    dataset_size = len(self.train_dataset)
                    train_size = int(0.9 * dataset_size)
                    val_size = dataset_size - train_size
                    # from torch.utils.data import random_split
                    # train_dataset, val_dataset = random_split(self.train_dataset, [train_size, val_size])
                    
                    # train_dataset = self.train_dataset[:train_size]
                    # val_dataset = self.train_dataset[train_size:]
                    train_dataset, val_dataset = self.train_dataset._dataset.train_test_split(0.1, seed=42, shuffle=True).values()
            
            
                    train_dataset = textattack.datasets.HuggingFaceDataset(train_dataset, None,'train')
                    val_dataset = textattack.datasets.HuggingFaceDataset(val_dataset, None,'eval')
                    
                    
                    train_batch_size_1 = train_batch_size  # Define your batch size for the first DataLoader
                    train_batch_size_2 = train_batch_size  # Define your batch size for the second DataLoader (if you want it different)
                    train_dataloader = self.get_train_dataloader(
                        train_dataset,
                        None,
                        train_batch_size_1
                    )

                    val_dataloader = self.get_train_dataloader(
                        val_dataset,
                        None,
                        train_batch_size_2*2
                    )

                    val_cycle_dataloader = itertools.cycle(val_dataloader)

                    # if 'Embedding' in self.training_args.Method_Dictionary:
                    #     if self.training_args.Method_Dictionary['Embedding'] == 'DSRM':
                    self.avg_loss = training_utils_functions.ExponentialMovingAverage()
                    
                    


                else:
                    train_dataloader = self.get_train_dataloader(
                        self.train_dataset,None,train_batch_size
                    )
            else:
                train_dataloader = self.get_train_dataloader(
                    self.train_dataset,None,train_batch_size
                )

                # print ('tl',next(iter(train_dataloader))[0][:4])
                # sys.exit()


             




            # adv_dataset = self.train_dataset[0:int(len(self.train_dataset)*0.05)]
            # print ('input cols',self.train_dataset.input_columns)
            # adv_dataset = textattack.datasets.Dataset(adv_dataset , input_columns=[self.train_dataset.input_columns], label_map=self.train_dataset.label_map, label_names=self.train_dataset.label_names, output_scale_factor=self.train_dataset.output_scale_factor, shuffle=self.train_dataset.shuffle  )
            # print ('first sample',adv_dataset[0])
            # adv_dataset = textattack.datasets.HuggingFaceDataset(adv_dataset   )
            # print (len(adv_dataset),adv_dataset[0])

            #NEEDCHANGE# train dataset or adv dataset
            #train
            # indices = [i for i in range(int(len(self.train_dataset)*1))]
            # adv_dataset = torch.utils.data.Subset( self.train_dataset,indices)
            # adv_dataloader = self.get_train_dataloader(
            #     adv_dataset,None,train_batch_size
            # )

            # adv

            if self.training_args.Data_ratio != 0:

                adv_dataloader = self.get_train_dataloader(
                    adv_dataset,None,train_batch_size
                )

                adv_cycle_dataloader = itertools.cycle(adv_dataloader)

                progress_dataloader = self.get_train_dataloader(
                    progress_dataset,None,train_batch_size
                )

                progress_cycle_dataloader = itertools.cycle(progress_dataloader)

                progress_extra_dataloader = self.get_train_dataloader(
                    progress_extra_dataset,None,train_batch_size
                )

                progress_extra_cycle_dataloader = itertools.cycle(progress_extra_dataloader)


                if  self.task_type == 'OT_GL_CC':


                    pos_label_dataset_adv = copy.deepcopy(adv_dataset)
                    pos_label_dataset_adv.filter_by_labels_([1])


                    pos_label_dataset_adv_dataloader = self.get_train_dataloader(
                        pos_label_dataset_adv,None,train_batch_size
                    )

                    pos_label_dataset_adv_cycle = itertools.cycle(pos_label_dataset_adv_dataloader)

                    neg_label_dataset_adv = copy.deepcopy(adv_dataset)
                    neg_label_dataset_adv.filter_by_labels_([0])

                    neg_label_dataset_adv_dataloader = self.get_train_dataloader(
                        neg_label_dataset_adv,None,train_batch_size
                    )

                    neg_label_dataset_adv_cycle = itertools.cycle(neg_label_dataset_adv_dataloader)

            else:
                adv_dataloader =  None

                adv_cycle_dataloader = None

                progress_dataloader =  None

                progress_cycle_dataloader = None

                progress_extra_dataloader = None

                progress_extra_cycle_dataloader = None
                # pos_label_dataset = copy.deepcopy(adv_dataset)
                # pos_label_dataset.filter_by_labels_(1)


                # if self.training_args.task_type or self.task_type we use is OT_GL_CC (class conditional) we do this and feature extraction
                # with adv_dataset and train dataset we need to extrac 4 smaller sub datasets
                # that are class dependant. then do the get_train_dataloader

            

            model.train() # perahps we need 2 models
            # Epoch variables
            all_preds = []
            all_targets = []
            
            
            prog_bar = tqdm.tqdm(
                train_dataloader,
                desc="Iteration",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
 
            

            for step, batch in enumerate(prog_bar):

                time_batch_ignore = 0
                
                
                if self.training_args.Data_ratio != 0:

                    if self.training_args.AUG_Val:
                        adv_batch = None
                    else:
                        
                        adv_batch = next(adv_cycle_dataloader)

                    # if self.training_args.P_Val:
                    #     progress_batch = next(progress_cycle_dataloader)
                    # else:
                    progress_batch = None

                    # if self.training_args.PF_Val:
                    #     progress_extra_batch = next(progress_extra_cycle_dataloader)
                    # else:
                    progress_extra_batch = None
                 
                    # adv_batch = [next(adv_cycle_dataloader) for i in train_batch_size]
                    # for i in range(train_batch_size):
                    #     next(adv_cycle_dataloader)



                    pos_base_batch = None
                    neg_base_batch =None
                    pos_adv_batch =None
                    neg_adv_batch =None

                    if  self.task_type == 'OT_GL_CC':
                        pos_base_batch = next(pos_label_dataset_base_cycle)
                        neg_base_batch = next(neg_label_dataset_base_cycle)
                        pos_adv_batch = next(pos_label_dataset_adv_cycle)
                        neg_adv_batch = next(neg_label_dataset_adv_cycle)

                else:
                    adv_batch = None
                    progress_batch = None
                    progress_extra_batch = None
                    pos_base_batch = None
                    neg_base_batch =None
                    pos_adv_batch =None
                    neg_adv_batch =None

                # print ('shapes',len(batch[0]),len(adv_batch[0]))
                # print ('adv batch',batch[0][:4],adv_batch[0][:4])
                # print ('batch:',batch)
                # print ('adv batch:',adv_batch)
                # sys.exit()
                # high level call to model, we have to pass the batches from adv data and base data

                kwargs_train= {'model':model,
                          'tokenizer':tokenizer, 
                          'batch':batch,
                          'adv_batch':adv_batch,
                          'progress_batch':progress_batch,
                          'progress_extra_batch':progress_extra_batch,
                          'pos_base_batch':pos_base_batch,
                          'neg_base_batch':neg_base_batch,
                          'pos_adv_batch':pos_adv_batch,
                          'neg_adv_batch':neg_adv_batch,
                          'val_cycle_dataloader':val_cycle_dataloader,
                          'time_record':time_batch_ignore}

               
               


                # loss, preds, targets, extra = self.training_step(model, tokenizer, batch,adv_batch,progress_batch,progress_extra_batch,pos_base_batch,neg_base_batch,pos_adv_batch,neg_adv_batch)
                loss, preds, targets, extra,training_return_dic = self.training_step(**kwargs_train)
                
                time_to_train +=  training_return_dic['time_record'] 
                print ('time to train',time_to_train)

                if isinstance(  model, torch.nn.DataParallel):
                    loss = loss.mean()

                loss = loss / self.training_args.gradient_accumulation_steps
                print ('this loss',loss)
                print (extra)
                if loss > 0: 
                    if self.training_args.method_test =='Embedding':
                        if self.training_args.Method_Dictionary['Embedding'] == 'ASCC': 
                            loss.backward()
                            max_grad_norm = 1
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        elif self.training_args.Method_Dictionary['Embedding'] == 'InfoBert':
                            max_grad_norm = 1
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) 
                            # for name, param in model.named_parameters():
                            #     if param.grad is not None:
                            #         print('param',name, param.grad.norm())
                            
                            # for name, param in model.named_parameters(): 
                            #     if param.requires_grad and param.grad is not None and name == 'bert.encoder.layer.11.attention.output.dense.weight':
                            #         # Accessing the first weight value. Indices will be based on the dimensionality of the tensor.
                            #         first_weight_value = param.data.view(-1)[0].item()
                            #         first_weight_gradient = param.grad.view(-1)[0].item()

                            #         print(f"First weight value of {name}: {first_weight_value}")
                            #         print(f"First weight gradient of {name}: {first_weight_gradient}") 
                            # sys.exit()
                        elif self.training_args.Method_Dictionary['Embedding'] == 'DSRM':
                            loss.backward()
                            # model.zero_grad()
                        else:
                            loss.backward()
                    else:
                        loss.backward()
                            

                loss = loss.item()
                self._total_loss += loss
                self._current_loss += loss

                all_preds.append(preds)
                all_targets.append(targets)

                if (step + 1) % self.training_args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                    self._global_step += 1

                if self._global_step > 0:
                    # extra_formatted = [f'{x:.0f}' if x == 0 else f'{x:.3f}' for x in extra]
                    extra_formatted = ' '.join(f'{x:.0f}' if x == 0 else f'{x:.3f}' for x in extra)
                    description = f"Loss T{self._total_loss/self._global_step:.3f}, B{loss:.3f} E({extra_formatted})"
                    prog_bar.set_description(
                        description
                    )

                if self.training_args.method_test =='Embedding':
                    if self.training_args.Method_Dictionary['Embedding'] == 'DSRM':
                        model.zero_grad()

                # TODO: Better way to handle TB and Wandb logging

                if (self._global_step > 0) and (
                    self._global_step % self.training_args.logging_interval_step == 0
                ):

                    lr_to_log = (
                        scheduler.get_last_lr()[0]
                        if scheduler
                        else self.training_args.learning_rate
                    )
                    if self._global_step - self._last_log_step >= 1:
                        loss_to_log = round(
                            self._current_loss
                            / (self._global_step - self._last_log_step),
                            7,
                        )
                    else:
                        loss_to_log = round(self._current_loss, 7)
                    total_loss_log = round(self._total_loss/self._global_step,7)
                    cross_entropy = round(extra[0],7)
                    optimal_transport = round(extra[1],7)
                    adversarial_cross_entropy = round(extra[2],7)
                    mmd = round(extra[3],7)
                    mmd_progress = round(extra[4],7)
                    mmd_progress_extra = round(extra[5],7)
                    coral = round(extra[6],7)
                    flb_l = round(extra[7],7)
                    log = {"train/total_loss":total_loss_log,"train/loss": loss_to_log, "cross_entropy":cross_entropy,"optimal_transport":optimal_transport, "adversarial_cross_entropy":adversarial_cross_entropy,'mmd':mmd,'mmd_progress':mmd_progress,'mmd_progress_extra':mmd_progress_extra,'coral':coral,'flb':flb_l,"train/learning_rate": lr_to_log}
                    

                    # writer.add_scalar('Loss/total_train_loss',total_loss_log, step)
                    # writer.add_scalar('Loss/batch_train_loss',loss_to_log, step)
                    # writer.add_scalar('LR/lr',lr_to_log, step)
                    
                    if self.training_args.log_to_csv:
                        self._csv_log(log, self._global_step)

                    # if self._global_step > 0:
                    #     description += f"\nE{log_message}"
                    #     prog_bar.set_description(
                    #         description
                    #     )

                    if self.training_args.log_to_tb:
                        self._tb_log(log, self._global_step)

                    if self.training_args.log_to_wandb:
                        self._wandb_log(log, self._global_step)

                    self._current_loss = 0.0
                    self._last_log_step = self._global_step


                # Save model checkpoint to file.
                if self.training_args.checkpoint_interval_steps:
                    if (
                        self._global_step > 0
                        and (
                            self._global_step
                            % self.training_args.checkpoint_interval_steps
                        )
                        == 0
                    ):
                        self._save_model_checkpoint(
                            model, tokenizer, step=self._global_step
                        )

            preds = torch.cat(all_preds)
            targets = torch.cat(all_targets)
            if self._metric_name == "accuracy":
                correct_predictions = (preds == targets).sum().item()
                accuracy = correct_predictions / len(targets)
                metric_log = {"train/train_accuracy": accuracy}
                logger.info(f"Train accuracy: {accuracy*100:.2f}%")
            else:
                pearson_correlation, pearson_pvalue = scipy.stats.pearsonr(
                    preds, targets
                )
                metric_log = {
                    "train/pearson_correlation": pearson_correlation,
                    "train/pearson_pvalue": pearson_pvalue,
                }
                logger.info(f"Train Pearson correlation: {pearson_correlation:.4f}%")

            if len(targets) > 0:
                if self.training_args.log_to_tb:
                    self._tb_log(metric_log, epoch)
                if self.training_args.log_to_wandb:
                    metric_log["epoch"] = epoch
                    self._wandb_log(metric_log, self._global_step)

            # Evaluate after each epoch.
            time_record_start = time.time()
            loss_eval,eval_score, extra_eval = self.evaluate()
            time_record_end = time.time() - time_record_start
            time_epoch_ignore = time_record_end
            time_to_train += time_epoch_ignore
            print ('time epoch ignore',time_epoch_ignore)

            # self._total_loss_eval += loss_eval
            # self._current_loss_eval += loss_eval
            #
            #
            # if (self._global_step_eval > 0) and (
            #     self._global_step_eval % self.training_args.logging_interval_step == 0
            # ):
            #
            #
            #     if self._global_step_eval - self._last_log_step_eval >= 1:
            #         loss_to_log = round(
            #             self._current_loss_eval
            #             / (self._global_step - self._last_log_step_eval),
            #             7,
            #         )
            #     else:
            #         loss_to_log = round(self._current_loss_eval, 7)
            #     total_loss_log = round(self._total_loss_eval/self._global_step_eval,7)
            #
            #
            # total_loss_log = round(self._total_loss_eval/self._global_step_eval,7)
            # cross_entropy = round(extra_eval[0],7)
            # optimal_transport = round(extra_eval[1],7)
            # adversarial_cross_entropy = round(extra_eval[2],7)
            # mmd = round(extra_eval[3],7)
            # coral = round(extra_eval[4],7)
            # log_eval = {"eval/total_loss":total_loss_log,"eval/loss": loss_to_log, "cross_entropy":cross_entropy,"optimal_transport":optimal_transport, "adversarial_cross_entropy":adversarial_cross_entropy,'mmd':mmd,'coral':coral,"train/learning_rate": lr_to_log}
            #
            # if self.training_args.log_to_csv:
            #     self._csv_log_eval(log_eval, self._global_step_eval)
            # if self.training_args.log_to_tb:
            #     self._tb_log({f"eval/{self._metric_name}": eval_score}, epoch)
            # if self.training_args.log_to_wandb:
            #     self._wandb_log(
            #         {f"eval/{self._metric_name}": eval_score, "epoch": epoch},
            #         self._global_step,
            #     )
            #
            # self._current_loss_eval = 0.0
            # self._last_log_step_eval = self._global_step_eval



            if (
                self.training_args.checkpoint_interval_epochs
                and (epoch % self.training_args.checkpoint_interval_epochs) == 0
            ):
                self._save_model_checkpoint(model, tokenizer, epoch=epoch)

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_eval_score_epoch = epoch
                epochs_since_best_eval_score = 0
                self._save_model_checkpoint(model, tokenizer, best=True)
                logger.info(
                    f"Best score found. Saved model to {self.training_args.output_dir}/best_model/"
                )
            else:
                # save model every epoch

                epochs_since_best_eval_score += 1
                if self.training_args.early_stopping_epochs and (
                    epochs_since_best_eval_score
                    > self.training_args.early_stopping_epochs
                ):
                    logger.info(
                        f"Stopping early since it's been {self.training_args.early_stopping_epochs} steps since validation score increased."
                    )
                    break
        
        current_time = time.time() 
        time_to_train_final = (current_time - (start_time_train + time_to_train + time_to_generate_peturbed_dataset) )
        print ('everything',time_to_train_final,current_time,start_time_train, time_to_train, time_to_generate_peturbed_dataset)
        time_to_train_str = f"""\n(Time Take Train Epochs) {time_to_train_final} (Time ignored) {time_to_train}
        """
        update_p_time_train = f'{self.training_args.output_dir}/P.txt'

        file_object = open(update_p_time_train,'a')
        file_object.write(str(time_to_train_str))
        file_object.close()


        if self.training_args.log_to_tb:
            self._tb_writer.flush()

        # Finish training
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        if self.training_args.load_best_model_at_end:
            best_model_path = os.path.join(self.training_args.output_dir, "best_model")
            if hasattr(model, "from_pretrained"):
                model = model.__class__.from_pretrained(best_model_path)
            else:
                model = model.load_state_dict(
                    torch.load(os.path.join(best_model_path, "pytorch_model.bin"))
                )

        if self.training_args.save_last:
            self._save_model_checkpoint(model, tokenizer, last=True)

        self.model_wrapper.model = model
        self._write_readme(best_eval_score, best_eval_score_epoch, train_batch_size)

    def evaluate(self):
        """Evaluate the model on given evaluation dataset."""

        if not self.eval_dataset:
            raise ValueError("No `eval_dataset` available for training.")

        logging.info("Evaluating model on evaluation dataset.")
        model = self.model_wrapper.model
        tokenizer = self.model_wrapper.tokenizer

        model.eval()
        all_preds = []
        all_targets = []

        if isinstance(model, torch.nn.DataParallel):
            num_gpus = torch.cuda.device_count()
            eval_batch_size = self.training_args.per_device_eval_batch_size * num_gpus
        else:
            eval_batch_size = self.training_args.per_device_eval_batch_size



        if self.attack:# and epoch > num_clean_epochs and self.training_args.Data_ratio != 0:
            # try loading the path to evaluation dataset to use
            my_file = Path(f"./caches/cache_evaluation_adv_samples_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}.pt")
            if my_file.is_file(): #test at each epoch #  and self.training_args.attack_epoch_interval == self.training_args.num_epochs:
                # file exists so load the cache
                # adv_dataset_eval = torch.load(f'./cache_adv_samples_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}.pt')
                adv_dataset_eval = torch.load(f'./caches/cache_evaluation_adv_samples_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}_SS{self.training_args.Sem_Sim}_CS{self.training_args.Cos_Sim}_NC{self.training_args.No_Cand}.pt')


                # print ('lengh adv dataset', len(adv_dataset_eval))
                if len(adv_dataset_eval) < 100:
                    raise ValueError("EVALUATION ADVERSARIAL DATASET IS SMALLER THAN 100 SAMPLES? CHECK THE CACHE!")
            else:
                adv_dataset_eval = None
                warnings.warn('not loading adversarial valuation file because it dosnt exist')
            # else: # need to generate evaluation file
            #     adv_dataset_eval = self._generate_adversarial_examples_evaluation(0)#epoch) # generate evaluation files
            #     # save this new file as cache
            #     torch.save(adv_dataset,f'./cache_evaluation_adv_samples_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}.pt' )


        # if self.attack and epoch > num_clean_epochs and self.training_args.Data_ratio != 0:
        #
        #
        #
        #     my_file = Path(f"./cache_evaluation_adv_samples_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}.pt")
        #     if my_file.is_file() and self.training_args.attack_epoch_interval == self.training_args.num_epochs:
        #         # if self.training_args.attack_epoch_interval == self.training_args.num_epochs :
        #         #     print ('only 1 epoch')
        #         adv_dataset_eval = torch.load(f'./cache_adv_samples_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}.pt')
        #             # if file exists and
        #             # if self.training_args.attack_epoch_interval == self.training_args.num_epochs :
        #             # loaded_object = torch.load('./saved_object.pt')
        #             # else
        #             #
        #         print ('lengh adv dataset', len(adv_dataset))
        #         if len(adv_dataset) < 100:
        #             raise ValueError("ADVERSARIAL DATASET IS SMALLER THAN 100 SAMPLES? CHECK THE CACHE!")
        #
        #     else:
        #
        #         adv_dataset_eval = self._generate_adversarial_examples_evaluation(epoch)
        #         # if self.training_args.attack_epoch_interval == self.training_args.num_epochs
        #         # save (this is because we only need to save when we do offline training)
        #         # with online training we delete the file and epoch interval != num epochs so it will never load and save
        #         if self.training_args.attack_epoch_interval == self.training_args.num_epochs:
        #             torch.save(adv_dataset,f'./cache_adv_samples_DS{self.training_args.Dataset_attack}_DR{self.training_args.Data_ratio}.pt' )
        #             # loaded_object = torch.load('./saved_object.pt')
        #
        #
        #     model.train()
        #     model.to(textattack.shared.utils.device)

        # else:
        #     logger.info(f"Running clean epoch {epoch}/{num_clean_epochs}")
        #     adv_dataset = None



        eval_dataloader = self.get_eval_dataloader(self.eval_dataset, eval_batch_size)

        if adv_dataset_eval:
            eval_adv_dataloader = self.get_train_dataloader(
                adv_dataset_eval,None,eval_batch_size
            )

            adv_eval_cycle_dataloader = itertools.cycle(eval_adv_dataloader)


        # with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                if adv_dataset_eval:
                    eval_adv_batch = next(adv_eval_cycle_dataloader)
                else:
                    eval_adv_batch = None
                # return loss
                # loss, preds, targets, extra= self.training_step(model, tokenizer, batch,adv_batch = None,pos_base_batch=None,neg_base_batch=None,pos_adv_batch=None,neg_adv_batch=None)
                loss_eval,preds, targets, extra_eval = self.evaluate_step(model, tokenizer, batch,eval_adv_batch) # pass adv batch
                all_preds.append(preds)
                all_targets.append(targets)

                self._global_step_eval +=1


                self._total_loss_eval += loss_eval
                self._current_loss_eval += loss_eval


                if (self._global_step_eval > 0) and (
                    self._global_step_eval % self.training_args.logging_interval_step == 0
                ):


                    if self._global_step_eval - self._last_log_step_eval >= 1:
                        loss_to_log = round(
                            self._current_loss_eval
                            / (self._global_step_eval - self._last_log_step_eval),
                            7,
                        )
                    else:
                        loss_to_log = round(self._current_loss_eval, 7)
                    total_loss_log = round(self._total_loss_eval/self._global_step_eval,7)


                total_loss_log = round(self._total_loss_eval/self._global_step_eval,7)
                cross_entropy = round(extra_eval[0],7)
                optimal_transport = round(extra_eval[1],7)
                adversarial_cross_entropy = round(extra_eval[2],7)
                mmd = round(extra_eval[3],7)
                coral = round(extra_eval[4],7)
                log_eval = {"eval/total_loss":total_loss_log,"eval/loss": loss_to_log, "cross_entropy":cross_entropy,"optimal_transport":optimal_transport, "adversarial_cross_entropy":adversarial_cross_entropy,'mmd':mmd,'coral':coral,"train/learning_rate": 0}

                if self.training_args.log_to_csv:
                    self._csv_log_eval(log_eval, self._global_step_eval)


                self._current_loss_eval = 0.0
                self._last_log_step_eval = self._global_step_eval


        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        if self.task_type == "regression":
            pearson_correlation, pearson_p_value = scipy.stats.pearsonr(preds, targets)
            eval_score = pearson_correlation
        else:
            correct_predictions = (preds == targets).sum().item()
            accuracy = correct_predictions / len(targets)
            eval_score = accuracy

        if self._metric_name == "accuracy":
            logger.info(f"Eval {self._metric_name}: {eval_score*100:.2f}%")
        else:
            logger.info(f"Eval {self._metric_name}: {eval_score:.4f}%")

        return loss_eval,eval_score, extra_eval

    def _write_readme(self, best_eval_score, best_eval_score_epoch, train_batch_size):
        if isinstance(self.training_args, CommandLineTrainingArgs):
            model_name = self.training_args.model_name_or_path
        elif isinstance(self.model_wrapper.model, transformers.PreTrainedModel):
            if (
                hasattr(self.model_wrapper.model.config, "_name_or_path")
                and self.model_wrapper.model.config._name_or_path in HUGGINGFACE_MODELS
            ):
                # TODO Better way than just checking HUGGINGFACE_MODELS ?
                model_name = self.model_wrapper.model.config._name_or_path
            elif hasattr(self.model_wrapper.model.config, "model_type"):
                model_name = self.model_wrapper.model.config.model_type
            else:
                model_name = ""
        else:
            model_name = ""

        if model_name:
            model_name = f"`{model_name}`"

        if (
            isinstance(self.training_args, CommandLineTrainingArgs)
            and self.training_args.model_max_length
        ):
            model_max_length = self.training_args.model_max_length
        elif isinstance(
            self.model_wrapper.model,
            (
                transformers.PreTrainedModel,
                LSTMForClassification,
                WordCNNForClassification,
            ),
        ):
            model_max_length = self.model_wrapper.tokenizer.model_max_length
        else:
            model_max_length = None

        if model_max_length:
            model_max_length_str = f" a maximum sequence length of {model_max_length},"
        else:
            model_max_length_str = ""

        if isinstance(
            self.train_dataset, textattack.datasets.HuggingFaceDataset
        ) and hasattr(self.train_dataset, "_name"):
            dataset_name = self.train_dataset._name
            if hasattr(self.train_dataset, "_subset"):
                dataset_name += f" ({self.train_dataset._subset})"
        elif isinstance(
            self.eval_dataset, textattack.datasets.HuggingFaceDataset
        ) and hasattr(self.eval_dataset, "_name"):
            dataset_name = self.eval_dataset._name
            if hasattr(self.eval_dataset, "_subset"):
                dataset_name += f" ({self.eval_dataset._subset})"
        else:
            dataset_name = None

        if dataset_name:
            dataset_str = (
                "and the `{dataset_name}` dataset loaded using the `datasets` library"
            )
        else:
            dataset_str = ""

        loss_func = (
            "mean squared error" if self.task_type == "regression" else "cross-entropy"
        )
        metric_name = (
            "pearson correlation" if self.task_type == "regression" else "accuracy"
        )
        epoch_info = f"{best_eval_score_epoch} epoch" + (
            "s" if best_eval_score_epoch > 1 else ""
        )
        readme_text = f"""
            ## TextAttack Model Card

            This {model_name} model was fine-tuned using TextAttack{dataset_str}. The model was fine-tuned
            for {self.training_args.num_epochs} epochs with a batch size of {train_batch_size},
            {model_max_length_str} and an initial learning rate of {self.training_args.learning_rate}.
            Since this was a {self.task_type} task, the model was trained with a {loss_func} loss function.
            The best score the model achieved on this task was {best_eval_score}, as measured by the
            eval set {metric_name}, found after {epoch_info}.

            For more information, check out [TextAttack on Github](https://github.com/QData/TextAttack).

            """

        readme_save_path = os.path.join(self.training_args.output_dir, "README.md")
        with open(readme_save_path, "w", encoding="utf-8") as f:
            f.write(readme_text.strip() + "\n")
        logger.info(f"Wrote README to {readme_save_path}.")
