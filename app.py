import os
import torch
import gradio as gr
from train_optimized import GPT, GPTConfig
from huggingface_hub import hf_hub_download

# Load the model
def load_model():
    # Download model files from HF Hub
    config_path = hf_hub_download(repo_id="jatingocodeo/shakespeare-decoder", filename="config.json")
    model_path = hf_hub_download(repo_id="jatingocodeo/shakespeare-decoder", filename="pytorch_model.bin")
    
    # Load config
    import json
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Initialize model with config
    config = GPTConfig(
        vocab_size=config_dict['vocab_size'],
        n_layer=config_dict['n_layer'],
        n_head=config_dict['n_head'],
        n_embd=config_dict['n_embd'],
        block_size=config_dict['block_size'],
        dropout=config_dict['dropout'],
        bias=config_dict['bias']
    )
    model = GPT(config)
    
    # Load model weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Cache the model
MODEL = load_model()

def generate_text(
    prompt, 
    max_new_tokens=100, 
    temperature=0.8, 
    top_k=50
):
    # Tokenize input
    chars = sorted(list(set(open('input.txt').read())))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Convert prompt to tensor
    x = torch.tensor(encode(prompt), dtype=torch.long)[None,...]
    
    # Generate
    with torch.no_grad():
        y = MODEL.generate(x, max_new_tokens, temperature, top_k)[0]
    
    # Decode and return
    generated_text = decode(y.tolist())
    return generated_text

# Create Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="Enter your prompt here...",
            lines=5
        ),
        gr.Slider(
            label="Max New Tokens",
            minimum=10,
            maximum=500,
            value=100,
            step=10
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=2.0,
            value=0.8,
            step=0.1
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=100,
            value=50,
            step=1
        ),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="Shakespeare GPT",
    description="""
    This is a GPT model trained on Shakespeare's text. Enter a prompt and the model will continue it in Shakespeare's style.
    
    Parameters:
    - Temperature: Higher values make the output more random, lower values make it more deterministic
    - Top-k: Number of highest probability tokens to consider at each step
    - Max New Tokens: Maximum number of tokens to generate
    """,
    examples=[
        ["To be, or not to be,", 100, 0.8, 50],
        ["Friends, Romans, countrymen,", 150, 0.7, 40],
        ["Now is the winter of", 200, 0.9, 30],
    ]
)

if __name__ == "__main__":
    demo.launch() 