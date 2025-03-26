from torch.utils.data import DataLoader

from DataLoader.Dataset import CustomDataset


class DataLoaderWrapper:
    def __init__(self, dataset: CustomDataset, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """
        初始化 DataLoaderWrapper 类。
        
        参数:
        - dataset: torch.utils.data.Dataset 的实例，表示数据集。
        - batch_size: 每个批次的大小，默认为 32。
        - shuffle: 是否在每个 epoch 开始时打乱数据，默认为 True。
        - **kwargs: 其他传递给 DataLoader 的参数（如 num_workers, pin_memory 等）。
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        """
        返回 DataLoader 的迭代器。
        """
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        """
        获取下一个批次的数据。
        """
        if self.iterator is None:
            raise RuntimeError("Iterator not initialized. Call iter() first.")
        try:
            return next(self.iterator)
        except StopIteration:
            # 如果迭代结束，重新初始化迭代器
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def reset(self):
        """
        重置迭代器。
        """
        self.iterator = iter(self.dataloader)