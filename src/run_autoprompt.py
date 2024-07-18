import pandas as pd
from ast import literal_eval
import numpy as np
import sys
sys.path.insert(0, "/home/ec2-user/SageMaker/Auto-Prompt/src")
from src.utils.functions import generate_ex_prompt_styles,\
                                generate_optimal_prompt
from src.utils.configs import path_to_prompt_ids,\
                              path_to_data, NUMBER_OF_PROMPTS


def autoprompt(fix_prompts, files):
    """
    Run automatic prompt engineering on a question and answer dataset
    Args: 
        fix_prompts(Pandas df): example question/answer combinations
            that need new instructions. 
    Returns: 
        None: upload new instructions and mapped instructions to ids. 
    """
    fix_prompts['contexts'] = fix_prompts['contexts'].apply(literal_eval)
    fix_prompts['ground_truths'] = fix_prompts['ground_truths'].apply(
        literal_eval)

    new_prompts = fix_prompts.copy()

    # run autoprompt
    prompt_1 = []
    prompt_2 = []
    prompt_3 = []
    for i in range(len(fix_prompts)):
        sample = fix_prompts.iloc[i]
        question = sample['question']
        context = '. '.join(sample['contexts'])
        answer = '. '.join(sample['ground_truths'])
        description, test_cases = generate_ex_prompt_styles(files,
                                                            question=question,
                                                            context=context,
                                                            answer=answer)
        result = generate_optimal_prompt(description, test_cases,
                                         number_of_prompts=NUMBER_OF_PROMPTS)
        prompt_1 += [result.iloc[0]['Prompt']]
        prompt_2 += [result.iloc[1]['Prompt']]
        prompt_3 += [result.iloc[2]['Prompt']]
    new_prompts['best_prompt'] = prompt_1
    new_prompts['second_best_prompt'] = prompt_2
    new_prompts['third_best_prompt'] = prompt_3
    new_prompts.to_csv(f'{path_to_data}new_prompts.csv', index=False)

    # Prompt Ids
    all_prompts = prompt_1 + prompt_2 + prompt_3
    data = {'Id': range(len(all_prompts)), 'Prompts': all_prompts}
    df_prompts = pd.DataFrame(data)
    df_prompts.to_csv(f'{path_to_data}prompt_ids.csv', index=False)


def run_autoprompts(auto, files, path_to_ragas_outputs):
    """
    Preprocess dataset and call automatic prompt engineering function. 
    Args: 
        auto(boolean): whether to run automatic prompt engineering on 
            each question/answer pair or add random instructions from 
            existing prompt ids. 
    Returns: 
        None
    """
    # Ragas import
    result = pd.read_csv(path_to_ragas_outputs)
    # Find all prompts that need auto instruct
    fix_prompts = result[(result['context_precision'] < 0.5) |
                         (result['faithfulness'] < 0.5) |
                         (result['answer_similarity'] < 0.5) |
                         (result['answer_correctness'] < 0.5)]
    if not auto:
        # Don't perform autoprompt, just randomize new instructions to
        # new samples
        instructions = pd.read_csv(path_to_prompt_ids)
        instruction_set = instructions.sample(len(fix_prompts), replace=True)
        fix_prompts['new_instruction'] = np.asarray(instruction_set['Prompts'])
        fix_prompts['instruction_id'] = np.asarray(
            instruction_set['Prompt_Ids'])
        fix_prompts.to_csv(f'{path_to_data}new_prompts.csv', index=False)
    else:
        # Perform autoprompt
        autoprompt(fix_prompts, files)