from .blocks import BasicBlock
from ModelManager.network import NetworkGroup


def get_block_dict():
    return {"basic": BasicBlock}

class ModelManager:
    def __init__(self, config: dict):
        self.config = config
        self.block_dict = get_block_dict()
        self.valid_model_num = config['valid_model_num']
        self.model_group_list = []

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



