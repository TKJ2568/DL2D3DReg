import toml

from ModelManager import ModelManager
from Trainer import Trainer
from config import ConfigManager
from DataLoader import CustomDataLoader

class Main:
    def __init__(self):
        data_loader_config = ConfigManager.get_instance().get_all_configs("data_loader_config")
        self.data_loader = CustomDataLoader(data_loader_config)
        if data_loader_config['is_load_voxel']:
            print("加载体素模式中，其他模块不可用")
            return
        model_manager_config = ConfigManager.get_instance().get_all_configs("model_manager_config")
        self.model_manager = ModelManager(model_manager_config)
        # 初始化训练模块
        trainer_config = ConfigManager.get_instance().get_all_configs("trainer_config")
        self.trainer = Trainer(self.data_loader, self.model_manager, trainer_config, data_loader_config["is_cuda"])



if __name__ == '__main__':
    # 创建 ConfigManager 单例实例
    config_manager = ConfigManager.get_instance()
    with open('config/all_config_path.toml', 'r', encoding='utf-8') as f:
        config_files = toml.load(f)
        config_manager.load_configs(config_files['config_files'])
        main = Main()