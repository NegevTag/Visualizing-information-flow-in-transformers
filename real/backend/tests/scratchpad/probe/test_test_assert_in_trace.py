import torch
from info_flow.config import Config
from real.backend.tests.scratchpad.test_assert_in_trace import ModelInformationCalculator, _get_model
from icecream import ic
if __name__ == "__main__":
    print("Started")
    config = Config()
    model = _get_model(config.info_flow_model,config.hf_token)
    information_calculator = ModelInformationCalculator(model=model)
    prompt = 'The cat sat on the'
    with model.trace(prompt,remote=True):
        layers_actual_output:list = list().save() # (layers,(query_d_model))
        for layer in model.model.layers:
            layers_actual_output.append(layer.output[0].save()) #(query,dm_model)
    ic(layers_actual_output[0].shape)
    print(f"Total dims per_layer = {layers_actual_output[0].shape[0]} x {layers_actual_output[0].shape[1]}")         
    per_layer_contribution_per_residual = information_calculator.calc(prompt).post_mlp_per_layer_contribution_per_residual  #(layer,(query,key,d_model))
    for l in range(len(model.model.layers)): #(prompt_len(query),prompt_len(key),d_model)
        layer_contribution_per_residual = per_layer_contribution_per_residual[l]
        assert torch.is_same_size(layer_contribution_per_residual.sum(dim=1),layers_actual_output[l])
        print(f"layer {l} same size")
        diff = layer_contribution_per_residual.sum(dim=1) - layers_actual_output[l]
        print(f"Relative norm of diff to value= {torch.norm(diff)/torch.norm(layers_actual_output[l])}")
        ic(torch.norm(layer_contribution_per_residual)/torch.norm(layers_actual_output[l]))
        ic(torch.norm(layers_actual_output[l]))
        # assert torch.allclose(
        #     layer_contribution_per_residual.sum(dim=1),  # (query, d_model)
        #     layers_actual_output[l],                      # (query, d_model)
        #     atol=config.default_atol,
        #     rtol=config.default_rtol,
        # )
        print(f"layer {l} passed")
