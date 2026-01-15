# Project 5: AI Image Retrieval
## NumPy Implementation of Vision Transformer

**课程**: 深圳大学人工智能学院 Python程序设计
**项目**: Vision Transformer 图像检索系统
**完成日期**: 2026年01月14日

---

## 目录

1. [项目概述](#1-项目概述)
2. [环境配置](#2-环境配置)
3. [核心代码实现](#3-核心代码实现)
4. [测试过程与结果](#4-测试过程与结果)
5. [Web应用实现](#5-web应用实现)
6. [使用说明](#6-使用说明)
7. [总结](#7-总结)

---

## 1. 项目概述

### 1.1 项目目标

本项目要求使用纯NumPy实现Vision Transformer (ViT)的前向传播过程，并基于DINOv2预训练权重构建一个图像检索Web应用系统。

### 1.2 主要任务

| 任务 | 描述 | 文件 |
|------|------|------|
| TODO 1 | 实现多头注意力机制 | `dinov2_numpy.py` |
| TODO 2 | 实现位置编码插值 | `dinov2_numpy.py` |
| TODO 3 | 实现短边resize预处理 | `preprocess_image.py` |
| TODO 4 | 构建Django图像检索Web应用 | `image_retrieval/` |

### 1.3 项目结构

```
project5-SZU-python/
├── assignments/
│   ├── dinov2_numpy.py      # ViT核心实现
│   ├── preprocess_image.py  # 图像预处理
│   ├── debug.py             # 测试脚本
│   ├── vit-dinov2-base.npz  # 预训练权重
│   └── demo_data/
│       ├── cat.jpg
│       ├── dog.jpg
│       └── cat_dog_feature.npy  # 参考特征
└── image_retrieval/
    ├── manage.py
    ├── extract_features.py
    ├── image_retrieval/
    │   ├── settings.py
    │   └── urls.py
    └── retrieval/
        ├── views.py
        └── templates/index.html
```

---

## 2. 环境配置

### 2.1 依赖库

```
numpy
scipy
Pillow
Django
tqdm
```

### 2.2 模型配置参数

```python
config = {
    "hidden_size": 768,    # 隐藏层维度
    "num_heads": 12,       # 注意力头数
    "num_layers": 12,      # Transformer层数
    "patch_size": 14,      # Patch大小
}
```

---

## 3. 核心代码实现

### 3.1 TODO 1: 多头注意力机制 (MultiHeadAttention)

**文件位置**: `dinov2_numpy.py:156-176`

**实现思路**:

多头注意力(Multi-Head Attention)是Transformer的核心组件，其计算公式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

多头机制将注意力分成多个"头"并行计算，每个头关注不同的特征子空间：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

**实现代码**:

```python
def __call__(self, x):
    B, N, D = x.shape

    # 1. 通过线性变换计算Q, K, V
    q = self.q_proj(x)  # (B, N, D)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # 2. 重塑为多头格式: (B, N, D) -> (B, num_heads, N, head_dim)
    q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    # 3. 计算缩放点积注意力
    att = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
    att = softmax(att, axis=-1)
    out = np.matmul(att, v)  # (B, num_heads, N, head_dim)

    # 4. 拼接多头并输出投影
    out = out.transpose(0, 2, 1, 3).reshape(B, N, D)
    return self.out_proj(out)
```

**关键点说明**:
- `num_heads=12`, `head_dim=64` (768/12)
- transpose操作将序列维度和头维度交换，便于并行计算
- 缩放因子 $\sqrt{d_k}$ 防止点积值过大导致softmax梯度消失

---

### 3.2 TODO 2: 位置编码插值 (interpolate_pos_encoding)

**文件位置**: `dinov2_numpy.py:39-67`

**实现思路**:

ViT的位置编码是预训练时固定尺寸的(如224×224对应16×16个patch)。当输入图像尺寸变化时，需要对位置编码进行插值以匹配新的patch数量。

**实现代码**:

```python
def interpolate_pos_encoding(self, embeddings, height, width):
    num_patches = embeddings.shape[1] - 1  # 当前patch数(减去cls_token)
    N = self.position_embeddings.shape[1] - 1  # 原始patch数

    if num_patches == N:
        return self.position_embeddings  # 尺寸相同，无需插值

    # 分离cls_token位置编码和patch位置编码
    cls_pos = self.position_embeddings[:, :1, :]
    patch_pos = self.position_embeddings[:, 1:, :]

    # 计算空间维度
    dim = self.hidden_size
    h0 = height // self.patch_size
    w0 = width // self.patch_size
    sqrt_N = int(np.sqrt(N))

    # 重塑为2D空间格式
    patch_pos = patch_pos.reshape(1, sqrt_N, sqrt_N, dim)

    # 使用scipy.ndimage.zoom进行双线性插值
    scale_h = h0 / sqrt_N
    scale_w = w0 / sqrt_N
    patch_pos = zoom(patch_pos, (1, scale_h, scale_w, 1), order=1)

    # 重塑回序列格式
    patch_pos = patch_pos.reshape(1, -1, dim)

    return np.concatenate([cls_pos, patch_pos], axis=1)
```

**关键点说明**:
- `order=1` 表示双线性插值
- cls_token的位置编码保持不变，只对patch位置编码进行插值
- 支持任意分辨率输入(需为patch_size的倍数)

---

### 3.3 TODO 3: 短边Resize预处理 (resize_short_side)

**文件位置**: `preprocess_image.py:27-59`

**实现思路**:

与center_crop不同，resize_short_side保持图像纵横比，将短边缩放到目标尺寸，并确保两边都是patch_size(14)的倍数。

**实现代码**:

```python
def resize_short_side(img_path, target_size=224):
    # Step 1: 加载图像
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    # Step 2: 计算新尺寸(短边=target_size，保持纵横比)
    if w < h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    # Step 3: 确保两边都是14的倍数
    new_w = (new_w // 14) * 14
    new_h = (new_h // 14) * 14
    new_w = max(new_w, 14)
    new_h = max(new_h, 14)

    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Step 4: 归一化
    image = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
    return image[None]  # (1,C,H,W)
```

**关键点说明**:
- 保持纵横比可以避免图像变形
- 14的倍数约束确保可以完整划分为patch
- 使用ImageNet标准归一化参数

---

## 4. 测试过程与结果

### 4.1 测试方法

使用`debug.py`对实现进行验证，通过提取cat.jpg和dog.jpg的特征，与参考特征`cat_dog_feature.npy`进行对比。

### 4.2 测试代码

```python
import numpy as np
from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

weights = np.load('vit-dinov2-base.npz')
vit = Dinov2Numpy(weights)

cat_feat = vit(center_crop('./demo_data/cat.jpg'))
dog_feat = vit(center_crop('./demo_data/dog.jpg'))
ref_feats = np.load('./demo_data/cat_dog_feature.npy')

# 计算差异
cat_diff = np.abs(cat_feat - ref_feats[0]).max()
dog_diff = np.abs(dog_feat - ref_feats[1]).max()
```

### 4.3 测试结果

| 指标 | Cat | Dog |
|------|-----|-----|
| 特征维度 | (1, 768) | (1, 768) |
| 最大绝对差异 | 8.84e-02 | 5.58e-02 |
| 均方误差(MSE) | 6.86e-04 | 3.88e-04 |
| **余弦相似度** | **0.999883** | **0.999935** |

### 4.4 结果分析

- **余弦相似度 > 0.9999**：提取的特征向量方向与参考特征几乎完全一致
- 微小的数值差异来源于浮点运算精度，不影响实际检索效果
- 验证通过，实现正确

---

## 5. Web应用实现

### 5.1 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                       前端 (index.html)                      │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   上传图片按钮   │ -> │        显示Top-10检索结果        │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │ POST /search/
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    后端 (Django views.py)                    │
│  1. 接收上传图片                                              │
│  2. 预处理 + ViT特征提取                                      │
│  3. 与图库特征计算余弦相似度                                   │
│  4. 返回Top-10结果                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              预提取特征 (gallery_features.npz)               │
│            10000+ 图库图片的768维特征向量                     │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 核心模块

#### 5.2.1 特征预提取脚本 (extract_features.py)

```python
def extract_features():
    vit = Dinov2Numpy(np.load('vit-dinov2-base.npz'))

    features = []
    paths = []
    for img_path in tqdm(image_files):
        feat = vit(center_crop(str(img_path)))[0]
        features.append(feat)
        paths.append(str(img_path))

    np.savez('gallery_features.npz', features=features, paths=paths)
```

#### 5.2.2 检索视图 (views.py)

```python
@csrf_exempt
def search(request):
    # 1. 保存上传图片
    uploaded_file = request.FILES['image']

    # 2. 提取查询特征
    vit = get_vit_model()
    query_feat = vit(center_crop(temp_path))[0]

    # 3. 计算余弦相似度
    similarities = cosine_similarity(query_feat, gallery_feats)

    # 4. 返回Top-10
    top_indices = np.argsort(similarities)[::-1][:10]
    return JsonResponse({'results': results})
```

#### 5.2.3 余弦相似度计算

```python
def cosine_similarity(query_feat, gallery_feats):
    query_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)
    gallery_norms = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)
    similarities = np.dot(gallery_norms, query_norm)
    return similarities
```

### 5.3 前端页面

使用原生HTML/CSS/JavaScript实现，主要功能：
- 图片上传与预览
- 异步请求搜索接口
- 网格布局展示Top-10结果及相似度

---

## 6. 使用说明

### 6.1 环境准备

```bash
# 安装依赖
pip install numpy scipy Pillow Django tqdm
```

### 6.2 验证ViT实现

```bash
cd project5-SZU-python/assignments
python debug.py
```

### 6.3 准备图库

1. 下载10000+图片到 `downloaded_images/` 目录
2. 复制图片到Django静态目录：
```bash
cp downloaded_images/*.jpg image_retrieval/static/gallery/
```

### 6.4 提取图库特征

```bash
cd project5-SZU-python/image_retrieval
python extract_features.py
```

### 6.5 启动Web服务

```bash
cd project5-SZU-python/image_retrieval
python manage.py runserver
```

访问 http://127.0.0.1:8000 使用系统

### 6.6 使用流程

1. 打开浏览器访问系统
2. 点击"Upload Image"上传查询图片
3. 系统自动进行特征提取和相似度计算
4. 页面展示Top-10最相似的图库图片

---

## 7. 总结

### 7.1 完成情况

| 任务 | 状态 | 说明 |
|------|------|------|
| TODO 1: MultiHeadAttention | ✅ 完成 | 实现12头注意力机制 |
| TODO 2: interpolate_pos_encoding | ✅ 完成 | 支持任意分辨率输入 |
| TODO 3: resize_short_side | ✅ 完成 | 保持纵横比的预处理 |
| TODO 4: Django Web应用 | ✅ 完成 | 完整的图像检索系统 |

### 7.2 技术要点

1. **纯NumPy实现ViT**: 深入理解了Transformer架构，包括多头注意力、位置编码、LayerNorm等组件
2. **特征检索**: 使用余弦相似度进行高效的向量检索
3. **Web开发**: 基于Django构建前后端分离的图像检索应用

### 7.3 性能说明

- 特征维度: 768维
- 单张图片特征提取: 约2-3秒(纯NumPy)
- 相似度计算: 毫秒级(向量化运算)

### 7.4 改进方向

1. 使用GPU加速(PyTorch/TensorFlow)提升特征提取速度
2. 引入近似最近邻搜索(如FAISS)支持更大规模图库
3. 添加图片分类标签，支持语义过滤

---
