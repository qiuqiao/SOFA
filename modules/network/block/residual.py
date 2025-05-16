import torch.nn as nn
import torch


class Residual(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, dim: int = -1) -> None:
        """
        Residual module for residual connections in neural networks.

        Args:
            input_dims (int): Number of input dimensions.
            output_dims (int): Number of output dimensions.
            dim (int, optional): Dimension along which to apply the residual connection. Defaults to -1.
        """
        super().__init__()
        self.dim: int = dim
        self.output_dims: int = output_dims
        self.input_dims: int = input_dims
        self.projection: nn.Module = (
            nn.Linear(input_dims, output_dims)
            if input_dims != output_dims
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual module.

        Args:
            x (torch.Tensor): Input tensor.
            out (torch.Tensor): Output tensor.

        Returns:
            torch.Tensor: Result of the residual connection.
        """
        x = torch.transpose(x, -1, self.dim)
        out = torch.transpose(out, -1, self.dim)
        if x.shape[-1] != self.input_dims:
            raise ValueError(
                f"Dimension mismatch: expected input dimension {self.input_dims}, but got {x.shape[-1]}."
            )
        if out.shape[-1] != self.output_dims:
            raise ValueError(
                f"Dimension mismatch: expected output dimension {self.output_dims}, but got {out.shape[-1]}."
            )
        return torch.transpose(out + self.projection(x), -1, self.dim)


if __name__ == "__main__":
    model = Residual(2, 3)
    x1 = torch.randn(2, 2, 2)
    x2 = torch.randn(2, 2, 3)
    y = model(x1, x2)
    print(y.shape)
