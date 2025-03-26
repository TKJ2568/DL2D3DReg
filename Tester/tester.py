import torch
from tqdm import tqdm

from DataLoader import CustomDataManager, DataLoaderWrapper
from DataLoader.DRR import Projector
from ModelManager import ModelManager
from Tester.CustomMetrics import *
from Tester.ResultManage import ResultManager


def init_projector(config):
    projector = Projector(directory=config['voxel_path'])
    projector.d_s2p = config['d_s2p']
    projector.im_sz = config['im_sz']
    return projector

def get_metric_instance_list(config, **kwargs):
    metric_instance_list = []
    if "MPD_metric" in config.keys():
        voxel_size = config["MPD_metric"]["voxel_size"]
        metric_instance_list.append(MaxPointDistanceTest(voxel_size))
    if "vm_metric" in config.keys():
        metric_instance_list.append(VoxelMSETest(**config["vm_metric"]))
    if "MSE_metric" in config.keys():
        metric_instance_list.append(ResultMSETest())
    if "Image_MSE_metric" in config.keys():
        metric_instance_list.append(ImageMSETest(projector=kwargs.get('projector')))
    if "Image_SSIM_metric" in config.keys():
        metric_instance_list.append(ImageSSIMTest(projector=kwargs.get('projector')))
    return metric_instance_list


class Tester:
    def __init__(self, data_loader: CustomDataManager, model_manager: ModelManager, config: dict, is_cuda):
        self.init_pose = data_loader.config['DRR']['init_pose']
        self.test_loader = DataLoaderWrapper(data_loader.test_dataset, 1, shuffle=False)
        self.label_transformer = data_loader.label_transformer
        self.model_manager = model_manager
        self.config = config
        self.is_cuda = is_cuda
        self.projector = init_projector(data_loader.config)
        self.projector.set_rotation(self.init_pose[0], self.init_pose[1], self.init_pose[2])
        self.metric_instance_list = get_metric_instance_list(config, projector=self.projector)
        # 初始化结果管理类
        test_name_list = [metric.name for metric in self.metric_instance_list]
        self.result_manager = ResultManager(test_name_list, **config)
        self.run_test()


    def run_test(self):
        # 初始化测试模型
        self.model_manager.create_test_models(**self.config)
        # 开始测试（外层循环：遍历模型）
        for model_name, test_model in tqdm(
                zip(self.result_manager.result_loggers.keys(), self.model_manager.test_model_list),
                desc="Testing Models",  # 进度条描述
                total=len(self.model_manager.test_model_list)  # 总模型数
        ):
            if self.is_cuda:
                test_model = test_model.cuda()

            # 内层循环：遍历 epoch
            for epoch in tqdm(
                    range(self.config['test_epoch']),
                    desc=f"Epochs ({model_name})",  # 进度条描述（带模型名）
                    leave=False  # 不保留内层进度条（避免显示混乱）
            ):
                self.test_once(model_name, test_model)
        # 可视化结果
        self.result_manager.visualize_result_and_save()

    def test_once(self, model_name, model):
        model.eval()

        images, labels = next(self.test_loader)

        with torch.no_grad():
            output = model(images)

            # 从归一化的值恢复到真实值
            labels = self.label_transformer.label2real(labels)
            output = self.label_transformer.label2real(output)

            # 转换为numpy数组
            labels_np = labels.cpu().numpy().ravel()
            output_np = output.cpu().numpy().ravel()
            for metric in self.metric_instance_list:
                res = metric(labels_np, output_np)
                # print(f"{model_name} {metric.name}: {res}")
                self.result_manager.add_one_result(model_name, metric.name, res)
            # print("Predictions:", output_np)
            # print("Ground truth:", labels_np)