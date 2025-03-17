from .Blocks import BasicBlock
from .network_frame import NetworkFrame


def get_block_dict():
    return {"basic": BasicBlock}

class ModelManager:
    def __init__(self, config: dict):
        self.block_dict = get_block_dict()
        self.valid_model_num = config['valid_model_num']
        self.model_list = []
        self.create_network_frame(config)

    def create_network_frame(self, config):
        general_config = config['general_config']
        for i in range(self.valid_model_num):
            specific_config = config['model_{0}'.format(i+1)]
            self.model_list.append(NetworkFrame(general_config, specific_config,
                                                self.block_dict[specific_config['block_type']]))


