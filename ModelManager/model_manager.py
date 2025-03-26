import json

from .blocks import BasicBlock
from ModelManager.network import NetworkGroup, NetworkFrame


def get_block_dict():
    return {"basic": BasicBlock}

class ModelManager:
    def __init__(self, config: dict):
        self.config = config
        self.block_dict = get_block_dict()
        self.valid_model_num = config['valid_model_num']
        self.model_group_list = []
        self.test_model_list = []

    def create_network_groups(self, **kwargs):
        general_config = self.config['general_config']
        label_transformer = kwargs.get('label_transformer')
        dataset_tuple = kwargs.get('dataset_tuple')
        for i in range(self.config['valid_model_num']):
            specific_config = self.config['model_{0}'.format(i+1)]
            block = self.block_dict[specific_config['block_type']]
            self.model_group_list.append(NetworkGroup(general_config, specific_config,
                                                      block=block,
                                                      label_transformer=label_transformer,
                                                      dataset_tuple=dataset_tuple))

    def create_test_models(self, **kwargs):
        model_info_path_list = kwargs.get('model_info_path_list')
        for model_info_path in model_info_path_list:
            with open(model_info_path, 'rb') as f:
                model_info = json.load(f)
                general_config = {"im_size": model_info['im_size'],
                                  "in_channels": model_info['in_channels'],
                                  "out_channels": model_info['out_channels']
                                  }
                specific_config = {"block": model_info['block'],
                                   "mlp": model_info['mlp']
                                   }
                block = self.block_dict[model_info['block_type']]
                net = NetworkFrame(general_config, specific_config, block=block)
                net.load_weights(model_info["model_save_path"])
                self.test_model_list.append(net)

