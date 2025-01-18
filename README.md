# Shakespeare Text Generator

This is a decoder-only transformer model trained on Shakespeare's text, designed to generate text in Shakespeare's distinctive style. The model has been trained to achieve a loss < 0.099999, ensuring high-quality text generation.

## Model Architecture

The model is a GPT-style decoder-only transformer with the following specifications:
- **Total Parameters**: 151M
- **Layers**: 12 transformer blocks
- **Attention Heads**: 16 heads per block
- **Embedding Dimension**: 1024
- **Context Length**: 256 tokens
- **Vocabulary**: Character-level tokenization
- **Activation**: GELU
- **Layer Normalization**: Pre-norm configuration
- **Attention**: Flash Attention for efficient computation

## Training Details

- **Training Data**: Shakespeare's complete works
- **Training Objective**: Next character prediction
- **Optimizer**: AdamW
  - Learning Rate: 3e-4
  - Weight Decay: 0.1
  - Beta1: 0.9
  - Beta2: 0.95
- **Learning Rate Schedule**: Cosine decay with warmup
- **Gradient Clipping**: 1.0
- **Achieved Loss**: < 0.099999

## Usage

The model is deployed as a Gradio app that allows you to:
1. Enter any text prompt
2. Configure generation parameters:
   - **Temperature** (0.1-2.0): Controls randomness
     - Lower values (e.g., 0.3) for more focused, deterministic output
     - Higher values (e.g., 1.5) for more creative, diverse output
   - **Top-k** (1-100): Number of tokens to sample from
     - Lower values for more constrained text
     - Higher values for more variety
   - **Max New Tokens** (10-500): Length of generated text

## Sample Outputs

### Example 1: Famous Hamlet Soliloquy
![Sample Output 1](images/sample1.png)

In this example, the model continues the famous "To be, or not to be" soliloquy with parameters:
- Temperature: 0.8
- Top-k: 50
- Max New Tokens: 100

### Example 2: Casual Conversation in Shakespeare Style
![Sample Output 2](images/sample2.png)

Here, the model transforms a modern casual greeting into Shakespeare's style with parameters:
- Temperature: 0.1 (more deterministic)
- Top-k: 1 (most likely token)
- Max New Tokens: 100

## Example Prompts

Try these classic Shakespeare openings:
```
To be, or not to be,
Friends, Romans, countrymen,
Now is the winter of
All the world's a stage,
O Romeo, Romeo,
```

## Model Repository

The model is available at: [jatingocodeo/shakespeare-decoder](https://huggingface.co/jatingocodeo/shakespeare-decoder)

## Local Development

To run the model locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app.py
```

## Limitations

- Character-level tokenization means the model works best with standard English characters
- The context length is limited to 256 characters
- The model may occasionally generate anachronistic content
- Best results are achieved with prompts in a Shakespearean style

## Citation

If you use this model in your work, please cite:
```
@misc{shakespeare-decoder,
  author = {Jatin Garg},
  title = {Shakespeare Text Generator},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/jatingocodeo/shakespeare-decoder}
}
```
