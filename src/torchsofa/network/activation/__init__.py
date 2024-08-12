from .glu import SwiGLU

__all__ = ["SwiGLU"]


def get_activation(name, num_dims, *args, **kwargs):
    try:
        return globals()[name](num_dims, *args, **kwargs)
    except KeyError:
        import torch.nn

        return getattr(torch.nn, name)(*args, **kwargs)


if __name__ == "__main__":
    import torch

    l_relu = get_activation("LeakyReLU", 0.01)

    x = torch.randn(10)
    print(l_relu(x))
