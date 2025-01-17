import commentjson as json
import tinycudann as tcnn
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config_path, input_dims, output_dims):
        super().__init__()
        with open(config_path, "r") as f:
            config = json.load(f)

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=input_dims,
            n_output_dims=output_dims,
            encoding_config=config["encoding"],
            network_config=config["network"]
        )

    def forward(self, x):
        return self.model(x)
