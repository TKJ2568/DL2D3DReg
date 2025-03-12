import numpy as np

from .DRR import Projector

class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.projector = Projector(directory=config['voxel_path'])
        self.projector.im_sz = config['im_sz']
        if config['single_projection']:
            self.run_single_projection()

    def run_single_projection(self):
        init_angle = self.config['DRR']['init_pose']
        projection_type = self.config['DRR']['projection_type']
        self.projector.set_rotation(init_angle[0], init_angle[1], init_angle[2])
        self.projector.projector.set_project_mode(projection_type)

        translation_noise = self.config['DRR']['translation_noise']
        rotation_noise = self.config['DRR']['rotation_noise']
        rota_noise = np.array(translation_noise, dtype=np.float32)
        tran_noise = np.array(rotation_noise, dtype=np.float32)

        is_display = self.config['DRR']['show_single_projection']
        is_save = self.config['DRR']['save_single_projection']
        save_path = self.config['DRR']['single_projection_path']
        if is_display:
            mode = 'display'
        else:
            mode = 'not_display'
        self.projector.project(rota_noise, tran_noise, mode=mode, save=is_save, save_path=save_path)