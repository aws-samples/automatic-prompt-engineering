import sys
sys.path.insert(0, "/home/ec2-user/SageMaker/Auto-Prompt")
from src.utils.configs import path_to_prompts

class PromptStyle: 
    """
    Identify prompts to an llm to generate instructions
        based on different styles. For example, generating
        step by step instructions, one with an example, 
        one in a paragraph, etc. 
    Parameters: 
        style(str): filename for the specific prompt to 
            generate an instruction
        prompt(str): the full prompt to generate an 
            instructio
    """
    def __init__(self, filename): 
        self.style = filename[:-4]
        with open(f'{path_to_prompts}/{filename}') as f:
            lines = f.readlines()
        self.prompt = ''.join(lines)
        
class Task: 
    """
    Identify the generic task in combination with the 
        style in the PromptStyle class
    Parameters: 
        all_prompts(list(PromptStyle)): all the prompt styles
            for the specific task
         prompt(str): the generic task, for example, as seen
             in ragas_test_task.txt
    """
    def __init__(self, filename): 
        self.all_prompts = []
        with open(f'{path_to_prompts}/{filename}') as f:
            lines = f.readlines()
        self.prompt = ''.join(lines)
    def add_styles(self, prompts): 
        for p in prompts: 
            self.all_prompts += [p]
            