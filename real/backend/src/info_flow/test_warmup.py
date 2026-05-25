import torch
from info_flow.config import Config
from info_flow.math_warmap_cac_normal_res import ModelInformationCalculator, _get_model

if __name__ == "__main__":
    config = Config()
    model = _get_model(config.info_flow_model,config.hf_token)
    information_calculator = ModelInformationCalculator(model=model)
    prompt = 'The cat sat on the'
    with model.trace(prompt,remote=True):
        layers_actual_output:list = list().save() # (layers,(query_d_model))
        for layer in model.model.layers:
            layers_actual_output.append(layer.self_attn.output[0][0].save()) #(query,dm_model)
    print(f"Total dims per_layer = {layers_actual_output[0].shape[0]} x {layers_actual_output[0].shape[1]}")         
    per_layer_contribution_per_residual = information_calculator.calc(prompt).per_layer_contribution_per_residual  #(layer,(query,key,d_model))
    for l in range(len(model.model.layers)): #(prompt_len(query),prompt_len(key),d_model)
        layer_contribution_per_residual = per_layer_contribution_per_residual[l]
        assert torch.is_same_size(layer_contribution_per_residual.sum(dim=1),layers_actual_output[l])
        print(f"layer {l} same size")
        diff = layer_contribution_per_residual.sum(dim=1) - layers_actual_output[l]
        print(f"Relative norm of diff to value= {torch.norm(diff)/torch.norm(layers_actual_output[l])}")
        assert torch.allclose(
            layer_contribution_per_residual.sum(dim=1),  # (query, d_model)
            layers_actual_output[l],                      # (query, d_model)
            atol=config.default_atol,
            rtol=config.default_rtol,
        )
        print(f"layer {l} passed")
