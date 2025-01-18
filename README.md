# Shakespeare Text Generator

This is a GPT model trained on Shakespeare's text. The model has been trained to achieve a loss < 0.099999 and can generate Shakespeare-style text continuations.

## Model Details

The model is a decoder-only transformer with the following specifications:
- 151M parameters
- 12 layers
- 16 attention heads
- 1024 embedding dimension
- Trained on Shakespeare's text

## Usage

1. Enter a prompt in the text box
2. Adjust the generation parameters:
   - **Temperature**: Controls randomness (0.1-2.0)
   - **Top-k**: Number of tokens to consider at each step
   - **Max New Tokens**: Length of generated text

## Examples

Try these example prompts:
- "To be, or not to be,"
- "Friends, Romans, countrymen,"
- "Now is the winter of"

## Model Repository

The model is available at: [jatingocodeo/shakespeare-decoder](https://huggingface.co/jatingocodeo/shakespeare-decoder)
