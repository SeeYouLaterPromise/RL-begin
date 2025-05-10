import os

def save_training_config(config: dict, save_dir: str, filename: str = "config.txt"):
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, filename)
    with open(config_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print(f"✅ 超参数已保存至 {config_path}")
