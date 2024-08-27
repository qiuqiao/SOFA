from src.torchsofa.model.network.block.conv_next import ConvNextBlock

if __name__ == "__main__":
    import torch

    model = ConvNextBlock()
    print(model)

    x = torch.randn(4, 64, 1024)
    y = model(x)
    print(y.shape)

    y.sum().backward()
