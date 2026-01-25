from transformer_lens import HookedTransformer, utils
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from typing import List
from jaxtyping import Float, Int  
from torch import Tensor
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
import einops

from collections import defaultdict
import gc

from utils import *

def get_model(model_path:str):
    """   
    Model_path can be huggingface model name
    """
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = HookedTransformer.from_pretrained_no_processing(
        model_path,
        dtype=torch.bfloat16,   
        default_padding_side='left',
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def tokenize_instructions(tokenizer, instructions):
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids


def _generate_with_hooks(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    tokens: Int[Tensor, "batch_size seq_len"],
    max_tokens_generated: int = 64,
    fwd_hooks=[],
) -> List[str]:
    """
    Generate text from the model given input tokens, applying forward hooks 
    during generation.
    """
    all_tokens = torch.zeros(
        (tokens.shape[0], tokens.shape[1] + max_tokens_generated),
        dtype=torch.long,
        device=tokens.device,
    ) # Avoids repeatedly reallocating tensors during generation
    all_tokens[:, : tokens.shape[1]] = tokens
    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_tokens[:, : -max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(
                dim=-1
            )  # greedy sampling (temperature=0)
            all_tokens[:, -max_tokens_generated + i] = next_tokens
    return tokenizer.batch_decode(
        all_tokens[:, tokens.shape[1] :], skip_special_tokens=True
    )


def get_generations(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    fwd_hooks=[],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:
    generations = []
    for i in tqdm(range(0, len(instructions), batch_size)): # Avoid GPU OOM and improves throughput
        tokens = tokenize_instructions(
            tokenizer, instructions=instructions[i : i + batch_size]
        )
        generation = _generate_with_hooks(
            model,
            tokenizer,
            tokens,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)
    return generations


def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
):
    """
    activation = residual stream
    direction = refusal direction to ablate

    Uses orthogonal decomposition to remove the component of the activation
      aligned with the 'direction' vector.
    """
    if activation.device != direction.device:
        direction = direction.to(activation.device)

    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    ) # proj=(⟨a,d⟩)d # refusal component of activation (subspace set to refusal direction)
    return activation - proj # vector orthogonal to refusal subspace (no refusal component)


def get_activations(model, tokenizer, harmful_data, harmless_data, batch_size:int = 32):
    """    
    Get activations for harmful and harmless instructions.
    Args:
        model: The model to be trained.
        tokenizer: The tokenizer used for processing instructions.
        harmful_data: List of harmful instructions.
        harmless_data: List of harmless instructions.
        batch_size: The size of each training batch.
    """
    n_inst_train = min(256, len(harmful_data), len(harmless_data))
    # harmful_data => harmful_inst_train
    harmful = defaultdict(list)
    harmless = defaultdict(list)
    
    num_batches = (n_inst_train + batch_size - 1) // batch_size

    harmful_tokens = tokenize_instructions(
        tokenizer,
        instructions=harmful_data[:n_inst_train]
    )
    harmless_tokens = tokenize_instructions(
        tokenizer,
        instructions=harmless_data[:n_inst_train]
    )

    for i in tqdm(range(num_batches), desc="Training Pass"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_inst_train)
        
        harmful_batch = harmful_tokens[start_idx:end_idx]
        harmless_batch = harmless_tokens[start_idx:end_idx]
        
        harmful_logits, harmful_cache = model.run_with_cache(
            harmful_batch,
            names_filter=lambda hook_name: 'resid' in hook_name,
            device='cpu',
            reset_hooks_end=True
        )

        harmless_logits, harmless_cache = model.run_with_cache(
            harmless_batch,
            names_filter=lambda hook_name: 'resid' in hook_name,
            device='cpu',
            reset_hooks_end=True
        )

        for key in harmful_cache:
            harmful[key].append(harmful_cache[key])
            harmless[key].append(harmless_cache[key])

        del harmful_logits, harmless_logits, harmful_cache, harmless_cache  
        gc.collect()
        torch.cuda.empty_cache()

        harmful = {k: torch.cat(v) for k, v in harmful.items()}
        harmless = {k: torch.cat(v) for k, v in harmless.items()}
        return harmful, harmless
    
    
def change_weight(
    model: HookedTransformer,
    succeeded_layers: set,
    activation_scored:list,
    mode = "middle",
):
    """
    Change model weights by taking the middle of succeeded layers
    succeeded_layers: set of layer indices that succeeded
    mode: "middle" takes the middle succeeded layer's weights
    """
    if mode == "middle":
        candidate_layer = list(succeeded_layers)[len(succeeded_layers)//2]
    
    refusal_dir = activation_scored[candidate_layer]

    if refusal_dir.device != model.W_E.device:
        refusal_dir = refusal_dir.to(model.W_E.device)

    model.W_E.data = get_orthogonalized_matrix(model.W_E, refusal_dir)

    for block in tqdm(model.blocks):
        if refusal_dir.device != block.attn.W_O.device:
            refusal_dir = refusal_dir.to(block.attn.W_O.device)
        block.attn.W_O.data = get_orthogonalized_matrix(block.attn.W_O, refusal_dir)
        block.mlp.W_out.data = get_orthogonalized_matrix(block.mlp.W_out, refusal_dir)

    return model

def save_model(
        original_model: str,
        model: HookedTransformer,
        save_name:str = "abliterated_model.pt", 
        dtype: torch.dtype = torch.bfloat16
    ):
    hf_model = AutoModelForCausalLM.from_pretrained(
        original_model,
        torch_dtype=dtype,
        device='cpu',
    )
    
    lm_model = hf_model.model  
    state_dict = model.state_dict()
    lm_model.embed_tokens.weight = torch.nn.Parameter(state_dict["embed.W_E"].cpu())
    
    for l in range(model.cfg.n_layers):
        lm_model.layers[l].self_attn.o_proj.weight = torch.nn.Parameter(
            einops.rearrange(
                state_dict[f"blocks.{l}.attn.W_O"], "n h m->m (n h)", n=model.cfg.n_heads
            ).contiguous()
        )
        lm_model.layers[l].mlp.down_proj.weight = torch.nn.Parameter(
            torch.transpose(state_dict[f"blocks.{l}.mlp.W_out"], 0, 1).contiguous()
        )

    torch.save(
        hf_model.state_dict(),
        save_name
    )

    del lm_model, hf_model

