# DRR投影模块说明
## 版本信息
**2024.4.22-1.0版本**

## 更新内容说明
1. 将dicom_os改为dicom_manager,实现了对dicom的复杂处理
2. 新建了get_xray_将compute_cross_voxel从ct_projector分离出来
3. 更新了加噪模型

## dicom_manager
+ 读取，处理和保存dicom
+ 对x_ray图像本身的处理

## ct_projector
+ 接收投影参数，完成投影

## get_xray
+ 帮助ct_projector完成射线穿过体素的计算
+ 提供射线累计的不同方法，包括透视投影和正交投影

## drr_generator
+ 建立dicom_manager和ct_projector之间的桥梁，提供外部接口
+ 内置加噪模型

## 噪声范围设定
### 标准正位
**位置:** 0 270 90或者0 90 270
**噪声范围：**
1. 角度：此时主光轴是CT体素的Y轴，绕Y轴的旋转为±10。绕其他轴的旋转为±5
2. 位移：沿主光轴Y轴的位移是±50，沿其他方向的位移是±25
