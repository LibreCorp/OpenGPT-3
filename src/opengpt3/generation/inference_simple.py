import torch
from transformers import AutoTokenizer
from opengpt3.modeling.transformer import GPTModel
import torch.nn.functional as F

def simple_inference(prompt: str, model_path: str, tokenizer_path: str, max_length: int = 50, temperature: float = 1.0, nucleus_prob: float = 0.9):
    """
    Performs simple text generation using the custom GPTModel.

    Args:
        prompt (str): The initial text prompt.
        model_path (str): Path to the custom model checkpoint directory.
        tokenizer_path (str): Path to the tokenizer directory.
        max_length (int): Maximum length of the generated sequence.
        temperature (float): Softmax temperature for sampling.
        nucleus_prob (float): Top-p nucleus sampling probability.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = GPTModel.from_pretrained(model_path)
    model.eval() # Set model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output_ids = input_ids
    past = None

    # Determine how many tokens to generate
    prompt_length = input_ids.shape[1]
    n_ctx = model.hparams['n_ctx']
    
    # Ensure max_length does not exceed model's context window
    if max_length > n_ctx:
        print(f"Warning: max_length ({max_length}) exceeds model's context window ({n_ctx}). Capping at {n_ctx}.")
        max_length = n_ctx

    num_tokens_to_generate = max_length - prompt_length
    if num_tokens_to_generate <= 0:
        print("Prompt is already at or exceeds max_length. No new tokens generated.")
        print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        return

    with torch.no_grad():
        for _ in range(num_tokens_to_generate):
            if past is not None:
                current_input_ids = output_ids[:, -1:]
            else:
                current_input_ids = output_ids

            logits, past = model(current_input_ids, past=past, training=False)
            
            # Get logits for the last token
            next_token_logits = logits[-1, :]

            # Apply temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply nucleus sampling (top-p)
            if nucleus_prob < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > nucleus_prob
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Sample the next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

            # Append the sampled token to the output
            output_ids = torch.cat([output_ids, next_token], dim=1)

            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    # Example Usage:
    # Make sure your model and tokenizer paths are correct
    model_dir = "model/"
    tokenizer_dir = "model/"
    
    prompt_text = "He is a doctor. His main goal is to"
    print(f"\nPrompt: {prompt_text}")
    print("Generated Text:")
    simple_inference(
        prompt=prompt_text,
        model_path=model_dir,
        tokenizer_path=tokenizer_dir,
        max_length=100, # Total length of prompt + generated text
        temperature=0.7, # Adjust for creativity (higher = more creative)
        nucleus_prob=0.9 # Adjust for diversity (lower = more diverse)
    )

    prompt_text_2 = "The quick brown fox jumps over the lazy"
    print(f"\nPrompt: {prompt_text_2}")
    print("Generated Text:")
    simple_inference(
        prompt=prompt_text_2,
        model_path=model_dir,
        tokenizer_path=tokenizer_dir,
        max_length=80,
        temperature=0.8,
        nucleus_prob=0.95
    )
