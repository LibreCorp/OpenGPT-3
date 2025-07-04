import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from opengpt3.modeling.transformer import GPTModel

def gen(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = GPTModel.from_pretrained(args.model_path)
    model.eval()

    if args.use_gpu:
        model.to("cuda")

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]
    n_ctx = model.hparams['n_ctx']

    max_length = args.seq_len
    if max_length > n_ctx:
        print(f"Warning: seq_len ({max_length}) exceeds model's context window ({n_ctx}). Capping at {n_ctx}.")
        max_length = n_ctx

    if prompt_length >= max_length:
        print("Prompt is as long or longer than max_length. Nothing to generate.")
        print(tokenizer.decode(input_ids[0][:max_length], skip_special_tokens=True))
        return

    output_ids = input_ids
    past = None
    
    num_tokens_to_generate = max_length - prompt_length

    for _ in range(num_tokens_to_generate):
        if past is not None:
            # After the first pass, we only need to pass the last token
            current_input_ids = output_ids[:, -1:]
        else:
            # On the first pass, we pass the full prompt
            current_input_ids = output_ids

        logits, past = model(current_input_ids, past=past, training=False)
        
        # The model returns logits for all tokens in the input, we only need the last one
        next_token_logits = logits[-1, :]

        # temperature
        if args.temperature > 0:
            next_token_logits = next_token_logits / args.temperature

        # top-p
        if args.nucleus_prob < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > args.nucleus_prob
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('Inf')

        # sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

        # append
        output_ids = torch.cat([output_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id or num_tokens_to_generate <= 0:
            break

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(text)


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        "generate", help="generate text with a pretrained transformer"
    )
    parser.add_argument(
        "--tokenizer_path", required=True,
        help="HuggingFace tokenizer name or path"
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to the custom model checkpoint"
    )
    group = parser.add_argument_group("Generation options")
    group.add_argument("--prompt", required=True,
                       help="text prompt to generate from")
    group.add_argument("--seq_len", type=int, default=64,
                       help="maximum generation length")
    group.add_argument("--nucleus_prob", type=float, default=0.85,
                       help="top-p nucleus sampling probability")
    group.add_argument("--temperature", type=float, default=1.0,
                       help="softmax temperature for sampling")
    parser.add_argument("--use_gpu", action="store_true",
                        help="move model and tokens to GPU")
    parser.set_defaults(func=gen)
