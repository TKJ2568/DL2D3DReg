import json

import numpy as np
from tqdm import tqdm

from utils.dire_check import manage_folder
from utils.time_record import timer_decorator_with_info
from .DRR import Projector

class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        if config['is_load_voxel']:
            self.projector = Projector(directory=config['voxel_path'])
            self.projector.d_s2p = config['d_s2p']
            self.projector.im_sz = config['im_sz']
        if config['single_projection']:
            self.run_single_projection()
        if config['generate_train_data']:
            self.generate_train_data()

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

    @timer_decorator_with_info("生成训练数据")
    def generate_train_data(self):
        train_params = self.config['DRR_train']
        train_num = train_params['train_num']
        # 位姿设定
        init_pose = [0, 270, 90]
        trans_noise_range = [25, 50, 25]
        rota_noise_range = [5, 10, 5]
        pose_params = None

        if train_params['standard_pose'] == "标准正位":
            pose_params = train_params['standard_front']
        elif train_params['standard_pose'] == "标准侧位":
            pose_params = train_params['standard_side']
        if pose_params is not None:
            init_pose = pose_params['init_pose']
            trans_noise_range = pose_params['translation_noise_range']
            rota_noise_range = pose_params['rotation_noise_range']
        else:
            print("未定义的位姿参数，使用默认标准正位参数")
        # 噪声采样模式
        noise_sample_mode = train_params['noise_sample_mode']
        save_path = train_params['save_path']
        manage_folder(save_path, "images")
        self.projector.set_rotation(init_pose[0], init_pose[1], init_pose[2])
        # 字典文件保存文件标签
        im_dict = {"init_pose": {
            "rx_init": str(init_pose[0]),
            "ry_init": str(init_pose[1]),
            "rz_init": str(init_pose[2]),
        }, "total_num": train_num}
        # 使用 tqdm 创建进度条
        with tqdm(total=train_num, desc="Generating Training Data", unit="image") as pbar:
            for i in range(train_num):
                translation_noise = np.array([0, 0, 0], dtype=np.float32)
                rotation_noise = np.array([0, 0, 0], dtype=np.float32)
                if noise_sample_mode == 'uniform':
                    # 按维度均匀分布生成随机数
                    translation_noise = np.array(
                        [np.random.uniform(-r, r) for r in trans_noise_range], dtype=np.float32
                    )
                    rotation_noise = np.array(
                        [np.random.uniform(-r, r) for r in rota_noise_range], dtype=np.float32
                    )
                elif noise_sample_mode == 'normal':
                    # 按维度正态分布生成随机数（均值为 0，标准差为范围的一半）
                    translation_noise = np.array(
                        [np.random.normal(loc=0, scale=r/2) for r in trans_noise_range], dtype=np.float32
                    )
                    rotation_noise = np.array(
                        [np.random.normal(loc=0, scale=r/2) for r in rota_noise_range], dtype=np.float32
                    )
                im = self.projector.project(rotation_noise, translation_noise, mode='not_display')
                im = np.expand_dims(im, axis=0)
                # 归一±1之间
                norm_rota_noise = rotation_noise / rota_noise_range
                norm_tran_noise = translation_noise / trans_noise_range
                t_info = (str(round(float(norm_tran_noise[0]), 2)) + "_" +
                          str(round(float(norm_tran_noise[1]), 2)) + "_" +
                          str(round(float(norm_tran_noise[2]), 2)))
                im_info = ("x{0}_y{1}_z{2}_{3}.npy".
                           format(round(float(norm_rota_noise[0]), 2),
                                  round(float(norm_rota_noise[1]), 2),
                                  round(float(norm_rota_noise[2]), 2), t_info))
                im_dict[im_info] = {
                    "name": im_info,
                    "rx_noise": str(rotation_noise[0]),
                    "ry_noise": str(rotation_noise[1]),
                    "rz_noise": str(rotation_noise[2]),
                    "tx_noise": str(translation_noise[0]),
                    "ty_noise": str(translation_noise[1]),
                    "tz_noise": str(translation_noise[2]),
                }
                np.save(save_path + "/images/{0}".format(im_info), im)
                # 更新进度条
                pbar.update(1)
        # 将文件写入json
        with open(save_path + "/label.json", "w") as f:
            json.dump(im_dict, f)
        print("加载入文件完成...")