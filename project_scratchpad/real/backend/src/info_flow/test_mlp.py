import torch
from info_flow.config import Config
from info_flow.ex2_calc_mlp_as_well import ModelInformationCalculator, _get_model
from icecream import ic
if __name__ == "__main__":
    config = Config()
    model = _get_model(config.info_flow_model,config.hf_token)
    information_calculator = ModelInformationCalculator(model=model)
    prompt = 'The cat sat on the mat, and then afterword, he decided that '
    with model.trace(prompt,remote=True):
        layers_actual_output:list = list().save() # (layers,(query_d_model))
        for layer in model.model.layers:
            layers_actual_output.append(layer.output[0].save()) #(query,dm_model)
    print(f"Total dims per_layer = {layers_actual_output[0].shape[0]} x {layers_actual_output[0].shape[1]}")         
    per_layer_contribution_per_residual = information_calculator.calc(prompt).post_mlp_per_layer_contribution_per_residual  #(layer,(query,key,d_model))
    ic(layers_actual_output[0].shape)
    ic(len(layers_actual_output))
    for l in range(len(model.model.layers)): #(prompt_len(query),prompt_len(key),d_model)
        layer_contribution_per_residual = per_layer_contribution_per_residual[l]
        my_full_res = layer_contribution_per_residual.sum(dim=1)[1:]
        real_residual = layers_actual_output[l][1:]
        assert torch.is_same_size(my_full_res,real_residual)
        print("Same size")
        diff = (my_full_res-real_residual)
        # ic(diff.norm()/real_residual.norm())
        # ic(my_full_res.norm()/real_residual.norm())
        # ic(real_residual.norm(dim=-1))   # per-token norms
        
        # diff_per_tok = (my_full_res - real_residual).norm(dim=-1)
        # ref_per_tok  = real_residual.norm(dim=-1)
        # ic(diff_per_tok / ref_per_tok)             # which token spikes?
        ic((diff.abs() - config.default_rtol * real_residual.abs()).max())
        violation = diff.abs() - config.default_rtol * real_residual.abs()
        mask = violation > 0.1
        ic(mask.sum().item(), real_residual.abs()[mask].max().item() if mask.any() else None)
        try:
            assert torch.allclose(
                my_full_res,  # (query, d_model)
                real_residual,                      # (query, d_model)
                atol=config.default_atol,
                rtol=config.default_rtol,
            )
        except AssertionError:
            print(f"LAYER {l} FAILED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            print(f"layer {l} passed")
