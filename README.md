# LLM Explainer 


## Overview
LLM Explainer is a project designed to provide explainable AI (XAI) through attention and gradient-based methods. This project aids researchers and practitioners in understanding the inner workings of AI models, thus enhancing transparency and trust in AI systems.

## Main Features

- **Attention Analysis**: Tools for visualizing and analyzing attention mechanisms in AI models.
- **Gradient-based Explanation**: Techniques for understanding model predictions through gradient visualization.

## Environment 
To ensure the project runs smoothly, the following environment setup is recommended:

- Python 3.8 or higher
- Necessary dependencies as listed in `requirements.txt`

```bash
python -m venv venv
source venv/bin/activate   
```

## Installation
Install the required dependencies:

```bash
pip install -e .
```

## Quick Start

### Step1 : Data Preprocess
We use a chatbot example to demonstrate how to use our toolbox. You can change the code and methods as needed. Replace `THE_MODEL_NAME_FROM_HUGGINGFACE` with the actual model name you are using from HuggingFace's model hub, or provide your locally saved model.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer for the specified model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained('THE_MODEL_NAME_FROM_HUGGINGFACE')

# Load the language model for the specified model from HuggingFace
model = AutoModelForCausalLM.from_pretrained('THE_MODEL_NAME_FROM_HUGGINGFACE')

# Define the system prompt for the chatbot
system_prompt = """\
You are a customer support chatbot. Please use the [DATA] as your current information.
Answer questions using the [DATA] provided. If the answer is not in the [DATA],
please say [Please try a different question or provide more details].
Do not include any fabricated content in your answers.\
"""

# Define the user's question and the available data
doc_question = """
What are the operating hours for online customer support and phone support?
[DATA]
Online Chat Support: Monday to Friday 8:00 AM - 8:00 PM EST
Saturday 9:00 AM - 5:00 PM EST
Phone Support: 24/7 available for Premium customers
Regular customers: Monday to Friday 9:00 AM - 6:00 PM EST\
"""

# Define the expected answer for the user's question
ans = "Our online chat support is available Monday to Friday from 8:00 AM to 8:00 PM EST and Saturday from 9:00 AM to 5:00 PM EST. Phone support is available 24/7 for Premium customers, while Regular customers can call Monday to Friday from 9:00 AM to 6:00 PM EST."

# Create a list of messages to be tokenized, including the system prompt and user's question
messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": doc_question,
    },
]

# Tokenize the chat messages using the tokenizer's chat template
tokenized_chat = tokenizer.apply_chat_template(messages)
```

### Step2 : Use Attention or Gradient Method
The `Explainer` is a context manager that provides explainable results. If you want to change the method, you can do so within the context manager. Currently, the available methods are `Attention` and `Gradient`. You can choose one or both. Below, we will show you how to use those methods.

- `method="Attention"`: Only get Attention results.
- `method="Gradient"`: Only get Gradient results.
- `method=["Attention", "Gradient"]`: Get both Attention and Gradient results.

**Note:** The toolbox cannot use the beam search strategy for inference. Please set `num_beams=1` when using the Huggingface inference method (i.e., `model.generate(**kwargs)`).

### Needed Arguments

The following arguments are required regardless of the method you choose:

- `"tokenizer"`: The tokenizer to use.
- `"output"`: The output data.
- `"input_ids"`: The tokenized input.
- `"task_type"`: The task being performed, we currently support below tasks:
  - "question_answering": Question-Answering
  - "translation": Translation
  - "summarizations": summarizations
  - "generative_text_chat": Chatbot
  - "generative_text_chat_assistant": Chatbot task with assist model in inference

Additionally, for the `chatbot` and `generative_text_chat_assistant` tasks, you will need:

- `"input_shape"`: The length of the input data.

### LLM Explainer - Attention

When choosing the `Attention` method, ensure that the model you are using includes an Attention Mechanism. If it does not, the toolbox will not produce correct results. We refer to [BertViz](https://github.com/jessevig/bertviz) and [Daam](https://github.com/castorini/daam) for their excellent work in Explainable AI.

#### Input Arguments
When using the Attention method, the following arguments should be provided as input to calculate the Attention method results:

- `"layer_names"`: The name or naming pattern of the attributes in the `model` where queries are located. Multiple names or patterns are supported (e.g., `["q_proj", "k_proj", "v_proj"]`).
- `is_input` (bool): Specifies whether the queries represent the input to the given layer. If `True`, queries are considered inputs; if `False`, they are considered outputs.
- `position` (int): The position of the query in the input/output arguments. Should be an integer. If there is only one argument, the position is 0.
- `strict` (bool): If `True`, the function will search for a layer with an exact name matching `layer_name`. If `False`, it will find layers with naming patterns matching `layer_name`.

#### Return
The return value is a dictionary. You can find the meanings of the dictionary keys below:

- `result["Prompt"]`: Decoded input tokens.
- `result["Prediction"]`: The result of the model inference.
- `result["Decode_output"]`: Decoded output tokens.
- `result["Attention"]["pair_result"]["attention"]`: Input Attention values.
- `result["Attention"]["attention_result"]`: Contains the following components:

  The output structure of the attention results is complex and divided into two parts. For each output token *i*, there are *j* blocks, *k* heads, and *w* head dimensions. The resulting dataframe will be of size *i* (number of output tokens) × *j* (number of blocks) × *k* (number of heads) × *w* (head dimensions).

  Due to the large and complex nature of this data, we first compute the median across the *w* (head dimension).

  - **Raw Data**:
    - Stored in: `result['Attention']['attention_result']['median_raw_df']`
    - DataFrame keys: *i* (output token), *j* (block), *k* (head)

  - **Group Data**:
    - Stored in: `result['Attention']['attention_result']['mean_group']`
    - DataFrame keys: *i* (output token)
    - Values represent the average attention across *j* (block) and *k* (head)

**Note:** 
1. The `generation_args` depend on your project and specific needs. In our demo, we use the inference method based on Huggingface, so we refer to the arguments from [Huggingface's text generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation). Feel free to replace these arguments with the ones you need for your project.

2. The `result['Attention']['attention_result']` currently only supports models with rotary position embeddings in Huggingface. This is used to obtain the query, key, and values when using KV cache.

```python
from explainer.attribute import ExplainerHandler
from explainer import Explainer

# Arguments for model inference
generation_args = {
    "max_new_tokens": 100,  # Maximum number of new tokens to generate
    "do_sample": False,     # Whether to use sampling; if False, use greedy decoding
    "top_k": 1,             # The number of highest probability vocabulary tokens to keep for top-k-filtering
    "top_p": 0.9,           # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation
    "return_dict_in_generate": True,  # Whether to return a dict with outputs
    "output_logits": True,  # Whether to return the logits in the output
    "num_beams": 1,         # Number of beams for beam search. 1 means no beam search.
    "input_ids": tokenized_chat  # The input token IDs
}

# Use the Explainer context manager to get explainable results
with Explainer(module=model, method="Attention") as explain:
    model.eval()  # Set the model to evaluation mode
    
    # Generate output using the model and the specified generation arguments
    output = model.generate(**generation_args)
    
    # Parameters required for the Explainer
    param = {
        "tokenizer": tokenizer,               # The tokenizer used
        "output": output,                     # The output generated by the model
        "task_type": "generative_text_chat",  # The task type (e.g., generative_text_chat)
        "input_ids": tokenized_chat.clone(),  # The tokenized chat input
        "layer_names": ["q_proj", "k_proj", "v_proj"],  # Names of the layers with queries
        "input_shape": input_shape,           # The shape of the input data
        "positions": [0, 0, 0],               # Positions of the queries in the input/output arguments
        "is_input": False,                    # Whether the queries represent the input to the layer
        "strict": False,                      # Whether to strictly match layer names
    }
        
    # Compute the explainable results using the Explainer
    result = explain.compute(**param)
```

### LLM Explainer - Gradient

When choosing the `Gradient` method, ensure that each question has a corresponding answer for calculating the model gradient.

#### Input Arguments:
When using the Gradient method, the following arguments should be provided as input to calculate the Gradient method results:

- `"label"`: The ground truth corresponding to the questions.
- `"embedding_name"`: The name of the model's embedding layer.

#### Return:
The return value is a dictionary. You can find the meanings of the dictionary keys below:

- `result["Prompt"]`: Decoded input tokens.
- `result["Prediction"]`: The result of the model inference.
- `result["Decode_output"]`: Decoded output tokens.
- `result["Gradient"]["gradient"]`: Raw gradient values.
- `result["Gradient"]["saliency_map"]`: Saliency map values.
- `result["Gradient"]["block_gradient_x_input"]`: Block Gradient multiplied by input values.

```python
from explainer.attribute import ExplainerHandler
from explainer import Explainer

# tokenized ground truth
labels = tokenizer(text=self.gpt4_ans, return_tensors="pt")["input_ids"]

# Arguments for model inference
generation_args = {
    "max_new_tokens": 100,  # Maximum number of new tokens to generate
    "do_sample": False,     # Whether to use sampling; if False, use greedy decoding
    "top_k": 1,             # The number of highest probability vocabulary tokens to keep for top-k-filtering
    "top_p": 0.9,           # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation
    "return_dict_in_generate": True,  # Whether to return a dict with outputs
    "output_logits": True,  # Whether to return the logits in the output
    "num_beams": 1,         # Number of beams for beam search. 1 means no beam search.
    "input_ids": tokenized_chat  # The input token IDs
}

# Use the Explainer context manager to get explainable results
with Explainer(module=model, method="Gradient") as explain:
    model.eval()  # Set the model to evaluation mode
    
    # Generate output using the model and the specified generation arguments
    output = model.generate(**generation_args)
    
    # Parameters required for the Explainer
    param = {
        "tokenizer": tokenizer,               # The tokenizer used
        "output": output,                     # The output generated by the model
        "task_type": "generative_text_chat",  # The task type (e.g., generative_text_chat)
        "input_ids": tokenized_chat.clone(),  # The tokenized chat input
        "input_shape": input_shape,           # The shape of the input data
        "labels": labels[0],                  # The ground truth labels for the input
    }
    
    # Compute the explainable results using the Explainer
    result = explain.compute(**param)
```

### Step3 : Display Explainable Result (Optional)
In this step, we provide an optional way to display the explainable results using color-coded text based on attention or gradient values.

```python
import pandas as pd 

def colorize_text(df, value):
    """Returns ANSI color code based on the gradient value."""
    if value <= 0.3:
        return 37  # White
    elif value <= 0.5:
        return 90  # Gray
    elif value <= 0.7:
        return 32  # Green
    elif value <= 0.9:
        return 33  # Yellow
    else:
        return 31  # Red

def display_colored_text(df):
    """Formats text with colors based on attention values."""
    text = ""
    for _, row in df.iterrows():
        color_code = colorize_text(df, row["attention"])
        text = "".join((text, f"\033[{color_code}m{row['text']}\033[0m"))
    return text

def output_visualization(result, item_name, idx):
    """Displays the colored output text based on attention values."""
    df = pd.DataFrame(
        [
            result["Prompt"],
            result["Attention"][item_name]["mean_group_df"]["tensor"][idx].tolist(),
        ]
    ).T
    df.columns = ["text", "attention"]
    print(display_colored_text(df))

def input_visualization(result, explainable_method):
    """Displays the colored input text based on attention or gradient values."""
    df = pd.DataFrame(
        [
            result["Prompt"],
            explainable_method,
        ]
    ).T
    df.columns = ["text", "attention"]
    print(display_colored_text(df))

# Input Attention Visualization
input_visualization(result, result["Attention"]["pair_result"]["attention"].tolist())

# Output Attention Visualization
for idx, word in enumerate(result["Decode_output"]):
    print(f"{idx}: {word}")
    output_visualization(result, "attention_result", idx)

# Input Gradient Visualization
input_visualization(result, result["Gradient"]["saliency_map"])
```


## More Examples
For additional examples on using the Attention and Gradient methods, please refer to the `example` folder.

## Constraint
#### Model Support:
1. Some information in `Attention` method currently only supports models with rotary position embeddings from Huggingface. This is necessary to obtain the query, key, and values when using the KV cache.
2. The toolbox does not support all models available on the Huggingface Hub. Ensure that your model has the required Attention Mechanism and rotary position embeddings for compatibility.

#### Inference Limitations:
1. The toolbox cannot use the beam search strategy for inference. Ensure `num_beams` is set to 1 when using the Huggingface inference method (i.e., `model.generate(**kwargs)`).

#### Data Requirements:
1. When using the `Gradient` method, each question must have a corresponding ground truth answer to calculate the model gradient.

#### Hardware and Performance:
1. Due to the complexity and size of the attention and gradient computations, ensure that your system has sufficient memory and processing power to handle the data and model inference.
2. Large models and datasets may require extended processing time and resources, so plan accordingly based on your hardware capabilities.

## Acknowledgments
We would like to thank the creators of [BertViz](https://github.com/jessevig/bertviz), [Daam](https://github.com/castorini/daam), [Huggingface](https://huggingface.co/), and [Captum](https://captum.ai/) for their wonderful work in the field of explainable AI and LLM. Their tools and libraries have greatly inspired and supported the development of this project.