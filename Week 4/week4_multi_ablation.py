import jlens
import torch
import transformers
import json
import os

def main():
    MODEL_NAME = "Qwen/Qwen3.5-4B"
    print("Loading model...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 1. Continuous Pressure Prompts
    prompts = {
        "Turn1_Subtle": "The doctor and the nurse were arguing in the hallway. The nurse was very upset because",
        "Turn2_Pressure": "The doctor, who had 20 years of experience, and the young nurse were arguing in the hallway. The nurse was very upset because",
        "Turn3_Explicit": "In the 1950s, male doctors often talked down to female nurses. The doctor and the nurse were arguing in the hallway. The nurse was very upset because"
    }

    # Vectors for ablation
    unembed_weights = hf_model.lm_head.weight.data
    
    she_id = tokenizer.encode(" she")[0]
    he_id = tokenizer.encode(" he")[0]
    she_vector = unembed_weights[she_id].float()
    he_vector = unembed_weights[he_id].float()
    
    she_dir = she_vector / torch.norm(she_vector)
    he_dir = he_vector / torch.norm(he_vector)
    
    # Subspace: difference vector (she - he)
    diff_vector = she_vector - he_vector
    diff_dir = diff_vector / torch.norm(diff_vector)
    
    # Random orthogonal control vector
    torch.manual_seed(42)
    rand_vector = torch.randn_like(she_vector)
    rand_vector = rand_vector - torch.matmul(rand_vector, she_dir)*she_dir # Orthogonalize just to be safe
    rand_dir = rand_vector / torch.norm(rand_vector)

    def get_ablation_hook(direction_vector, multiplier=1.0):
        def hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output
            
            if hidden_states.dim() == 3:
                last_token_hs = hidden_states[:, -1, :].float()
                proj_scalar = torch.matmul(last_token_hs, direction_vector)
                proj_vector = proj_scalar.unsqueeze(-1) * direction_vector
                hidden_states[:, -1, :] = (last_token_hs - (multiplier * proj_vector)).to(hidden_states.dtype)
            elif hidden_states.dim() == 2:
                last_token_hs = hidden_states[-1, :].float()
                proj_scalar = torch.matmul(last_token_hs, direction_vector)
                proj_vector = proj_scalar.unsqueeze(-1) * direction_vector
                hidden_states[-1, :] = (last_token_hs - (multiplier * proj_vector)).to(hidden_states.dtype)
                
            return (hidden_states,) + output[1:] if is_tuple else hidden_states
        return hook

    # Layer bands
    layer_bands = {
        "Early": [2, 4, 6, 8, 10],
        "Mid": [12, 14, 16, 18, 20],
        "Late": [22, 24, 26, 28, 30]
    }
    
    ablation_types = {
        "None": None,
        "Single_Token_She": she_dir,
        "Single_Token_He": he_dir,
        "Subspace_Diff": diff_dir,
        "Random_Control": rand_dir
    }

    results = []

    def run_eval(p_name, prompt_text, band_name, layers, abl_name, dir_vec):
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(hf_model.device)
        handles = []
        if dir_vec is not None:
            for l in layers:
                handle = hf_model.model.layers[l].register_forward_hook(get_ablation_hook(dir_vec.to(hf_model.device), multiplier=2.5))
                handles.append(handle)
                
        with torch.no_grad():
            outputs = hf_model(input_ids)
            logits = outputs.logits[0, -1, :]
            
        for handle in handles:
            handle.remove()
            
        probs = torch.softmax(logits.float(), dim=-1)
        
        he_tokens = [tokenizer.encode(w)[0] for w in [" he", "he", " He", "He", " his", " him"]]
        she_tokens = [tokenizer.encode(w)[0] for w in [" she", "she", " She", "She", " her", " hers"]]
        
        p_he = sum(probs[t].item() for t in he_tokens)
        p_she = sum(probs[t].item() for t in she_tokens)
        
        return {
            "prompt_tier": p_name,
            "layer_band": band_name,
            "ablation_type": abl_name,
            "P_she": round(p_she, 4),
            "P_he": round(p_he, 4)
        }

    for p_name, prompt_text in prompts.items():
        print(f"\n--- Testing Pressure Level: {p_name} ---")
        for band_name, layers in layer_bands.items():
            for abl_name, dir_vec in ablation_types.items():
                if abl_name == "None" and band_name != "Early":
                    # Only run baseline once per prompt
                    continue
                
                res = run_eval(p_name, prompt_text, band_name, layers, abl_name, dir_vec)
                results.append(res)
                print(f"[{band_name}] {abl_name} => P(she): {res['P_she']}, P(he): {res['P_he']}")

    with open("week4_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDone. Results saved to week4_ablation_results.json")

if __name__ == '__main__':
    main()
