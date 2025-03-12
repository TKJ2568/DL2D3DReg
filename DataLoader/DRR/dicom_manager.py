"""
dicom管理，处理以保存
"""
import os
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
import numpy as np


def _slices2voxels(slices):
    """
    因为slice的序号与z轴是反向的，所以在实际导入的时候需要逆序导入到ct_vox当中
    :param slices:
    :return:
    """
    # create 3D array
    ct_vox = np.zeros([slices[0].pixel_array.shape[0], slices[0].pixel_array.shape[1], len(slices)], dtype=np.int16)
    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img = s.pixel_array
        ct_vox[:, :, -i - 1] = img.T.astype(np.int16)
    return ct_vox


def norm_image(image, min_v=None, max_v=None, bits=None):
    image = image.astype(np.float64)
    if min_v is None:
        min_v = np.min(image)
    if max_v is None:
        max_v = np.max(image)
    if bits is None:
        return (255 * (image - min_v) / (max_v - min_v)).astype(np.uint8)
    else:
        return ((2 ** bits - 1) * (image - min_v) / (max_v - min_v)).astype(np.int)


class Manager:
    def __init__(self, pixel_status, cropped):
        self.slices = None
        self.dcm_dir = None
        self.dcm_filenames = None
        self.ct_vox = None
        self.vox_space = None
        self.status = [pixel_status, cropped]

    def open_folder(self, folder):
        """
        Inspired by https://github.com/pydicom/pydicom/blob/master/examples/image_processing/reslice.py
        :param str folder: see proj_ct_to_xray.
        :return tuple(np.ndarray, list(float), list(float), list(float)): ct_img3d, voxel_spacing, position, orientation
        """
        self.dcm_dir = folder
        # load the DICOM files
        filenames = [fn for fn in os.listdir(folder) if fn[-4:] == '.dcm']
        print('Loading {} files from {}'.format(len(filenames), folder))

        # skip files with no InstanceNumber
        slices = []
        for f_name in filenames:
            f = pydicom.read_file(os.path.join(folder, f_name), force=True)
            if hasattr(f, 'InstanceNumber'):
                slices.append(f)
                # print(f.ImagePositionPatient)
            else:
                print('File {} has no InstanceNumber'.format(f_name))

        # ensure they are in the correct order
        self.slices = sorted(slices, key=lambda slice: slice.InstanceNumber)
        # dcm的文件名，ct体素，ct体素空间对应的实际间距
        self.dcm_filenames = list(os.path.basename(s.filename) for s in self.slices)
        self.vox_space = np.array([slices[0].PixelSpacing[0], slices[0].PixelSpacing[1], slices[0].SliceThickness])
        self.ct_vox = _slices2voxels(self.slices)

    def reload_ct_vox(self):
        self.ct_vox = _slices2voxels(self.slices)

    def slices_process(self, raw2img=False, crop=False):
        """
        :param raw2img: 是否将dcm的raw_pixel坐标转换为像素坐标
        :param crop: 是否加骨窗
        :return:
        """
        # 加骨窗
        if crop:
            self.status[1] = 'cropped'
        if raw2img:
            self.status[0] = 'img'
        for i, s in enumerate(self.slices):
            img = s.pixel_array
            slope = float(s.RescaleSlope)  # 1
            intercept = float(s.RescaleIntercept)  # 可以从dcm文件中读出 -1024
            # 加骨窗
            if crop:
                if self.status[0] == 'raw_pixel':
                    img = np.clip(img, 110, 1250)
            if raw2img:
                img = img * slope + intercept
                img = norm_image(img)
            data_changed = np.array(img, dtype=np.int16)
            pd = data_changed.tobytes()
            s.PixelData = pd  # 写入dcm文件
        self.reload_ct_vox()
        print("转换完成，当前dcm状态", self.status)

    def save_dcm(self, new_dir):
        slices = []
        for i, dcm in enumerate(self.slices):
            slices.append(dcm)
        dcm_filename = zip(slices, self.dcm_filenames)
        for dcm, filename in dcm_filename:
            new_path = "{0}/{1}".format(new_dir, filename)
            # 原始图像的像素数据进行了压缩，新图像像素数据不压缩。不加上这一句会出现像素数据长度相关的报错
            dcm.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            dcm.save_as(new_path)
