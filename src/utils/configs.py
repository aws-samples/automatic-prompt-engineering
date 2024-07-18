# this determines how many candidate prompts to generate... the higher,
# the more expensive, but the better the results will be
NUMBER_OF_PROMPTS = 10 

# Path Variables
path_to_prompts = "../instruction_generation_templates"
path_to_data = "../data/"
path_to_prompt_ids = f"{path_to_data}prompt_ids.csv"

# K is a constant factor that determines how much ratings change
K = 32

CANDIDATE_MODEL = 'anthropic.claude-instant-v1'
CANDIDATE_MODEL_TEMPERATURE = 0.9

GENERATION_MODEL = 'anthropic.claude-instant-v1'
GENERATION_MODEL_TEMPERATURE = 0.1
GENERATION_MODEL_MAX_TOKENS = 60

N_RETRIES = 3  # number of times to retry a call to the ranking model if it fails
RANKING_MODEL = 'anthropic.claude-instant-v1'
RANKING_MODEL_TEMPERATURE = 0.5



# AutoPrompt Configs
WANDB_PROJECT_NAME = "bedrock-prompt-eng" # used if use_wandb is True, Weights &| Biases project name
WANDB_RUN_NAME = None # used if use_wandb is True, optionally set the Weights & Biases run name to identify this run

REGION = "us-east-1"

system_prompt = """
{test_case}

{description}
"""

system_gen_system_prompt = """Your job is to generate system prompts for Large
                                Language Model, given a description of the
                                use-case and some test cases.

                                The prompts you will be generating will be for
                                freeform tasks, such as generating a landing
                                page headline, an intro paragraph, solving a
                                math problem, etc.

                                In your generated prompt, you should describe
                                how the AI should behave in plain English.
                                Include what it will see, and what it's allowed
                                to output. Be creative with prompts to get the
                                best possible results. The AI knows it's an AI
                                -- you don't need to tell it this.

                                You will be graded based on the performance of
                                your prompt... but don't cheat! You cannot
                                include
                                specifics about the test cases in your prompt.
                                Any prompts with examples will be disqualified.
                                \n\nHere is the description of the use-case:
                                `{description}` Here are some test cases:
                                `{test_cases}`.

                                Most importantly, output `{number_of_prompts}`
                                system prompts as a list of objects with
                                `prompt` attribute in a json format."""


ranking_system_prompt = """Your job is to rank the quality of two outputs
generated by different prompts. The prompts are used to generate a response for
a given task.

You will be provided with the task description, the test prompt, and two
generations - one for each system prompt.

Rank the generations in order of quality. If Generation A is better, respond
with 'A'. If Generation B is better, respond with 'B'.

Remember, to be considered 'better', a generation must not just be good,
it must be noticeably superior to the other.

Also, keep in mind that you are a very harsh critic. Only rank a generation as better if it truly impresses you more than the other.

Respond with A or B, and nothing else. Be fair and unbiased in your judgement."""


