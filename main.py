import toml
from config import ConfigManager
from DataLoader import DataLoader

class Main:
    def __init__(self):
        data_loader_config = ConfigManager.get_instance().get_all_configs("data_loader_config")
        self.data_loader = DataLoader(data_loader_config)

if __name__ == '__main__':
    # 创建 ConfigManager 单例实例
    config_manager = ConfigManager.get_instance()
    with open('config/all_config_path.toml', 'r', encoding='utf-8') as f:
        config_files = toml.load(f)
        config_manager.load_configs(config_files['config_files'])
        main = Main()