# Tambouille

## Overview

Tambouille is a Python-based tool designed for managing and training visual retrieval models. It is built to work with ColQwen2 and ColPali models, facilitating both dataset generation and training processes for visual tasks.

## Pre-requisites

Before using Tambouille, ensure you have the following prerequisites installed on your system:
#### Poppler

To convert pdf to images we use the `pdf2image` library. This library requires `poppler` to be installed on your system. You can follow the instructions [on their website](https://poppler.freedesktop.org/). Or use the following:

__MacOS with homebrew__

```bash
brew install poppler
```

__Debian/Ubuntu__

```bash
sudo apt-get install -y poppler-utils
```

## Installation

Ensure you have Python 3.10 or higher installed on your system.

1. Clone the repository:
    ```bash
    git clone https://github.com/<your-username>/tambouille.git
    cd tambouille
    ```

2. Install the required dependencies via pip:
    ```bash
    pip install -r requirements.txt
    ```

3. For additional dependencies from Git sources, ensure Git is installed and accessible on your system.

## Usage
```python
from tambouille import Tambouille

# Initialize a Tambouille instance
tambouille = Tambouille(model_name="vidore/colqwen2-v0.1", model_type="colqwen2")

# Prepare your data
data = tambouille.prepare_data(path_to_data="path/to/data")

# Generate a dataset
dataset = tambouille.generate_dataset(
    qwen_vl_model="Qwen/Qwen2-VL-2B-Instruct", # You can any of the Qwen2VL model as long as you have the compute
    dataset=data, 
)

# Train the model
tambouille.train(dataset=dataset)
```

Your model is now trained and ready for use, while it is available in the checkpoints folder I would suggest to push it to the Hugging Face Hub for easy access and sharing. (You can use the `push_to_hub` and `repo_id` arguments in the `train` method to do so.)

[Byaldi](https://github.com/AnswerDotAI/byaldi/) is the easiest way to use it.

## Advanced Usage

### Initializing the Model

```python
from tambouille import Tambouille

tambouille = Tambouille(model_name="vidore/colqwen2-v0.1", model_type="colqwen2")
```

### Prepare your data
```python
data = tambouille.prepare_data(path_to_data="path/to/data")
```

### Generating Dataset from your data

You can generate a dataset using your own prompt and data structure by creating a `Prompt` object and passing it to the `generate_dataset` method. Can be useful to get queries more relevant to the domain or if your document is in a language other than english.

Here's an example:
```python
from tambouille.tambouille import Prompt
from pydantic import BaseModel

class QueryAnswer(BaseModel):
    query: str
    answer: str

prompt = """"Generate a query and an answer relevant to the image"""

prompt = Prompt(prompt=prompt, model=QueryAnswer)
dataset = tambouille.generate_dataset(
    qwen_vl_model="qwen-vl-model-path",
    prompt=prompt,
    dataset=data,
    img_column_name="image"
)
```

Here `dataset` is a Huggging Face Dataset object. I would strongly suggest that you look at several examples of the dataset to make sure that the queries are relevant to the image. If not, you can modify the prompt or the data structure to get better results.

Have a look at the `prompt.py` file to see the default prompt and data structure.

### Training the Model

Once you have your dataset prepared, you can proceed to train the model. Tambouille supports both local training and integration with Hugging Face Hub for sharing models.

```python
from transformers import TrainingArguments

# Define custom training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

# Specify optional parameters for logging, saving, and using WandB
tambouille.train(
    dataset="your_dataset",
    query_column_name="query",
    image_column_name="image",
    training_args=training_args,
    push_to_hub=True,
    repo_id="your-repo-id",
    local_save=True,
    wandb_logging=True,
    wandb_project="your_wandb_project"
)
```

## Contributing

Contributions to Tambouille are welcome! Fork the repository, make your changes, and submit a pull request.

## License

Tambouille is distributed under the MIT License. See the LICENSE file for more information.