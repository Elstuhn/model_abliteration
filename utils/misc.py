from collections import defaultdict
from transformer_lens import utils
from tqdm import tqdm
import functools
import einops
from jaxtyping import Float, Int
from torch import Tensor

from model import get_generations, direction_ablation_hook

def get_act_idx(cache_dict, act_name, layer):
    key = (act_name, layer)
    return cache_dict[utils.get_act_name(*key)]

def compute_refusal(
        model,
        harmful_activations: dict,
        harmless_activations: dict,
        activation_layer = ['resid_pre', 'resid_mid', 'resid_post'],
        selected_layers = [1]
    )->list:
    activation_refusals = defaultdict(list)
    for layer_num in range(1, model.cfg.n_layers):
        pos = -1

        for layer in activation_layer:
            harmful_mean_act = get_act_idx(
                harmful_activations,
                layer,
                layer_num
            )[:, pos, :].mean(dim=0)
        
            harmless_mean_act = get_act_idx(
                harmless_activations,
                layer,
                layer_num
            )[:, pos, :].mean(dim=0)

            refusal_dir = harmful_mean_act - harmless_mean_act
            refusal_dir = refusal_dir / refusal_dir.norm()
            activation_refusals[layer].append(refusal_dir)

    activation_scored = sorted(
        [
            activation_refusals[activation_layer[layer_idx]][l-1] 
            for l in range(1, model.cfg.n_layers)
            for layer_idx in activation_layer
        ],
        key = lambda x: abs(x.mean()),
        reverse=True
    )
    return activation_scored

def evaluate_refusal_direction(
        model,
        tokenizer,
        instructions, # => harmful_inst_test
        activation_scored,
        activation_layers = ['resid_pre'], # chosen activation layer
        ablation_scale: float = 1.0
    )->list:
    """    
    Evaluate the model with the refusal direction ablated.
    Args:
        model: The model to be evaluated.
        tokenizer: The tokenizer used for processing instructions.
        instructions: List of instructions to evaluate.
        activation_scored: The scored activation directions.
        activation_layers: The layers to apply the ablation.
        ablation_scale: The scale of the ablation.
    Returns:
        List of model outputs after ablation.
    """
    N_INST_TEST = 4 
    baseline_generations = get_generations(
        model, tokenizer, instructions[:N_INST_TEST], fwd_hooks=[]
    )

    EVAL_N = 15
    evals = []
    for refusal_dir in tqdm(activation_scored[:EVAL_N]):
        hook_fn = functools.partial(
            direction_ablation_hook, 
            direction=refusal_dir
        )
        fwd_hooks = [
            (utils.get_act_name(act_name, layer), hook_fn)
            for layer in list(range(model.cfg.n_layers))
            for act_name in activation_layers
        ]
        intervention_generations = get_generations(
            model, tokenizer, instructions[:N_INST_TEST], fwd_hooks=fwd_hooks
        )
        evals.append(intervention_generations)
        
    blacklist = ["I cannot", "I can't"]
    fail = []
    for i in range(N_INST_TEST):
        for layer_candidate in range(EVAL_N):
            if any(
                word in evals[layer_candidate][i] for word in blacklist
            ):
                fail.append((i, layer_candidate))
    
    succeeded = set(range(EVAL_N)) - set(fail)

    return succeeded

def get_orthogonalized_matrix(
    matrix: Float[Tensor, "... d_model"], vec: Float[Tensor, "d_model"]
) -> Float[Tensor, "... d_model"]:
    """   
    Function to get orthoganalized vector of 'matrix' with respect to 'vec'
    """
    proj = (
        einops.einsum(
            matrix, vec.view(-1, 1), "... d_model, d_model single -> ... single"
        )
        * vec
    )

    return matrix - proj

