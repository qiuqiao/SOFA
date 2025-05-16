from modules.network.unet import UNet
import torch.nn as nn
import torch


def get_network(
    network_name: str, configs: dict, weights_path: str | None = None
) -> nn.Module:
    """
    根据网络名称和配置文件返回网络模型
    :param network_name: 网络名称
    :param configs: 配置文件
    :param weights_path: 权重路径
    :return: 网络模型
    """
    match network_name:
        case "unet":
            network = UNet(
                **configs,
            )
        case _:
            raise ValueError(f"Unknown network name: {network_name}")

    if weights_path is not None:
        # 加载预训练权重
        state_dict = torch.load(weights_path, map_location="cpu")
        network.load_state_dict(state_dict, strict=False)

    return network


if __name__ == "__main__":
    # 测试网络
    network = get_network("unet", {"n_dims": 2})
    print(network)
