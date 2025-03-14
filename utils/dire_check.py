import os
import shutil

def manage_folder(path, folder_name):
    """
    检查指定路径下是否存在目标文件夹，如果存在则清空其内容，否则创建该文件夹。

    :param path: str, 文件路径
    :param folder_name: str, 目标文件夹名
    """
    # 构造完整的目标文件夹路径
    target_folder = os.path.join(path, folder_name)

    if os.path.exists(target_folder):  # 检查文件夹是否存在
        if os.path.isdir(target_folder):  # 确保目标是一个文件夹
            print(f"文件夹 '{folder_name}' 已存在，正在清空其内容...")
            # 清空文件夹内容
            for file_name in os.listdir(str(target_folder)):
                file_path = os.path.join(str(target_folder), file_name)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # 删除文件或符号链接
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # 删除子文件夹及其内容
                except Exception as e:
                    print(f"无法删除 {file_path}: {e}")
            print(f"文件夹 '{folder_name}' 的内容已清空。")
        else:
            print(f"'{target_folder}' 存在，但它不是一个文件夹。")
    else:
        # 如果文件夹不存在，则创建它
        print(f"文件夹 '{folder_name}' 不存在，正在创建...")
        os.makedirs(target_folder, exist_ok=True)
        print(f"文件夹 '{folder_name}' 已成功创建。")