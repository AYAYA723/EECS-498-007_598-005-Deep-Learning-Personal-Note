# EECS 498/598: Deep Learning for Computer Vision

课程：[Deep Learning for Computer Vision](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html)
（University of Michigan · Winter 2022 · Justin Johnson）

本仓库用于存放课程编程作业的个人解答。

## 课程作业

| 作业                        | 内容                                                        | 状态      |
| --------------------------- | ----------------------------------------------------------- | --------- |
| [Assignment 1](assignment1/) | PyTorch 101 + k-Nearest Neighbor 分类器                     | ✅ 已完成 |
| [Assignment 2](assignment2/) | 线性分类器（SVM / Softmax）+ 两层神经网络 + MNIST Challenge | ⬜ 待完成 |
| [Assignment 3](assignment3/) | 全连接网络 + 卷积网络（模块化反向传播、BatchNorm、Dropout） | ⬜ 待完成 |
| [Assignment 4](assignment4/) | 目标检测（FCOS 单阶段 + Faster R-CNN 风格两阶段）           | ⬜ 待完成 |
| [Assignment 5](assignment5/) | RNN / LSTM 图像描述 + Transformer                           | ⬜ 待完成 |
| [Assignment 6](assignment6/) | VAE + GAN + 网络可视化 + 风格迁移                           | ⬜ 待完成 |

## 目录结构

每个 `assignment{n}/` 目录内包含：

- `*.ipynb` — 引导 notebook（作业说明 + 测试单元）
- `*.py` — 需要填写的代码（在 `TODO` 标记的代码块内实现）

## 环境配置

- Python 3.14
- PyTorch 2.13（CUDA 13 · NVIDIA RTX 5070）

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python.exe -m pip install jupyter matplotlib pillow scipy numpy tqdm ipykernel
```

启动 notebook：

```bash
.venv/Scripts/jupyter.exe notebook
```
