from .rms import RMSNorm

__all__ = ["RMSNorm"]


def get_normalization(name, num_dims, *args, **kwargs):
    try:
        return globals()[name](num_dims, *args, **kwargs)
    except KeyError:
        import torch.nn

        return getattr(torch.nn, name)(*args, **kwargs)


if __name__ == "__main__":
    import torch

    rms_norm = get_normalization("RMSNorm", 64)
    print(rms_norm)
    x = torch.randn(4, 64, 1024)
    y = rms_norm(x)
    print(y.shape)

    y.sum().backward()
