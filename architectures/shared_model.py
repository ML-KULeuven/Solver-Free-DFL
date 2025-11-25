from torch import nn


class SharedModel(nn.Module):
    def __init__(self, input_dim_per_coeff, output_dim_per_coeff):
        super().__init__()
        self.input_dim_per_coeff = input_dim_per_coeff
        self.output_dim_per_coeff = output_dim_per_coeff
        self.layer = nn.Linear(input_dim_per_coeff, output_dim_per_coeff)

    def forward(self, input):
        output = self.layer(input)
        return output.squeeze(-1)