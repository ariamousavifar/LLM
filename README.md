# LLM: A Lightweight Language Model for Text Generation

## Overview

The **LLM** project is a modular implementation of a transformer-based language model for text generation. This
repository provides tools to train, evaluate, and fine-tune a lightweight language model. It's designed to handle custom
datasets and is optimized for flexibility, making it a great choice for researchers and developers experimenting with
language modeling techniques.

## Features

- Transformer-based architecture
- Support for multi-head attention and position embeddings
- Configurable hyperparameters
- Training and validation pipelines
- Integrated logging with [Weights & Biases (WandB)](https://wandb.ai/)
- Model saving and loading functionality
- Custom text generation

## Requirements

This project requires Python 3.8+ and PyTorch. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Prepare your dataset as a text file (e.g., `Dataset.txt`).
2. Set the hyperparameters in the `hyperparameters` dictionary.
3. Start training:

   ```bash
   m = train_save(dataset_name="Dataset.txt", encoding="normal", parameters=hyperparameters, wandb_log=False)
   ```

   Key arguments:
    - `--dataset_name`: Path to the training dataset.
    - `--encoding`: Character encoding mode (`normal` or custom).
    - `--wandb_log`: Enable or disable logging with WandB.

### Generating Text

Once the model is trained, generate text using:

```bash
python generate.py --model_path saved_models/model0.pth --initial_text "Once upon a time" --max_new_tokens 100
```

Key arguments:

- `--model_path`: Path to the saved model file.
- `--initial_text`: Initial text to seed the generation.
- `--max_new_tokens`: Number of tokens to generate.

### Saving and Loading Models

Save a trained model automatically using `train_save()`.

```python

save_model(model, encoding, hyperparameters)
```

Load a saved model:

```python
model, params = load_model("saved_models/model0.pth")
```

## Example Workflow

```python
from utils import train_save

hyperparameters = {
    "batch_size": 64,
    "context_length": 256,
    "max_iters": 5000,
    "eval_interval": 500,
    "learning_rate": 3e-4,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "eval_iters": 200,
    "n_embed": 384,
    "num_head": 4,
    "num_layer": 6,
    "dropout": 0.2,
    "temperature": 1.0,
    "epochs": 1,
    "train_rate": 0.9,
    "vocab_size": 256,
    "steps": 500,
    "bias": False
}

train_save(dataset_name="dataset/train.txt", encoding="normal", parameters=hyperparameters, wandb_log=True)
```

## Hyperparameters

The training process is controlled by the following hyperparameters:

| Parameter        | Description                               | Default Value |
|------------------|-------------------------------------------|---------------|
| `batch_size`     | Number of examples in each training batch | 64            |
| `context_length` | Length of the input context               | 256           |
| `max_iters`      | Maximum training iterations               | 5000          |
| `learning_rate`  | Learning rate for the optimizer           | 3e-4          |
| `n_embed`        | Embedding size                            | 384           |
| `num_head`       | Number of attention heads                 | 4             |
| `num_layer`      | Number of transformer layers              | 6             |
| `dropout`        | Dropout rate                              | 0.2           |
| `epochs`         | Number of training epochs                 | 1             |

## Logging with WandB

Enable WandB logging to visualize training metrics and generated text samples. To use WandB:

1. Install WandB:
   ```bash
   pip install wandb
   ```
2. Set up your WandB account by running:
   ```bash
   wandb login
   ```
3. Pass `wandb_log=True` when calling the training function.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

Special thanks to the PyTorch and Hugging Face communities for inspiration and resources. This project is an educational
effort to explore transformer-based models for language generation.

This project is based on the Final project description of the Deep Learning course at the University of Geneva with
Prof. François Fleuret. The code has been inspired by the base code of this
project ([Colab Notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)) and
modified under fair use.

---

Feel free to contribute or report issues via the GitHub repository!
