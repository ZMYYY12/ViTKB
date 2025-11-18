## 代码使用简介
# ViTKB-for-Soybean-Disease-Identification

> 官方PyTorch实现 | 论文处于投刊阶段，标题：《ViTKB: A New Deep Learning model for Soybean Leaf Disease Identification》

> 提出ViTKB模型，基于PyTorch框架实现三类大豆常见病害的高精度识别，助力农业病害智能化诊断。  

## 1. 研究背景与模型定位  

大豆作为重要豆科作物，其叶片病害（如灰斑病、黄斑虫、花叶病）易导致产量下降，传统人工检测存在效率低、依赖经验的问题。  

本文提出**ViTKB**，通过灵活的非线性函数逼近与双向跨尺度注意力机制，解决大豆病害“多类别区分难、复杂背景干扰大”的问题。模型基于PyTorch框架实现，采用Vision Transformer基础架构，深度集成KAN符号网络与BiFormer双向路由注意力，实现多尺度特征的高效建模，在三类大豆病害数据集上实现优异的分类性能，为农业病害自动化诊断提供高效解决方案。  


## 2. ViTKB核心创新点  

 
1.**双重创新机制协同**：
- **KAN非线性前馈网络**：基于Kolmogorov-Arnold表示理论，通过可学习的样条函数替代传统MLP，实现更灵活的非线性特征变换；
- **BiFormer稀疏注意力**：采用双向路由注意力机制，动态计算token间的相关性，实现高效的全局-局部特征交互；

2. **分层特征学习架构**：
- **多层Transformer Block堆叠**：每层包含LayerNorm、注意力机制和前馈网络，逐步抽象语义特征；
- **Patch Embedding预处理**：将224×224图像分割为16×16图像块，通过线性投影得到序列化token表示；
- **分层特征融合机制**：通过残差连接和层归一化，实现低层细节特征与高层语义特征的跨层融合。

3. **多尺度训练优化策略**：
- **渐进式学习率调度**：采用CosineAnnealingWarmRestarts策略，在训练过程中动态调整学习率；
- **分层参数优化**：为KAN层设置独立学习率（1e-4），避免预训练权重被过度破坏；
- **类别平衡采样**：针对三类任务的细粒度差异，在数据加载阶段实现类别均衡。



## 3. 实验数据集：三类大豆病害数据集  

### 3.1 数据集概况  

本研究基于**三类大豆病害识别数据集**，包含三种常见大豆叶片状态，数据集需自行下载后使用：  

| 数据集名称 | 包含类别 | 图像总数 | 图像分辨率 | 数据分布（训练:验证:测试） |
|------------|-------------------------|----------|------------|-----------------------|
| 三类大豆数据集 | 灰斑病（Grey spot）、黄斑病（Macular）、花叶病（Mosaic）、 | 6,000+ | 统一resize至224×224（适配模型输入） | 3:1:1 |  


### 3.2 数据集结构  
 
2. **文件夹组织**（下载后解压至项目根目录，结构如下）：  
```  
datasets/
├──train/ 
│   ├── Grey spot/       # 大豆灰斑病叶片图像  
│   ├── Macular/        # 大豆黄斑病叶片图像  
│   └──  mosaic/         # 大豆花叶病叶片图像
├──test/
│   ├── Grey spot/       # 大豆灰斑病叶片图像  
│   ├── Macular/        # 大豆黄斑病叶片图像  
│   └──  mosaic/         # 大豆花叶病叶片图像
            
```  


## 4. 实验环境配置  

### 4.1 依赖安装  

推荐使用Anaconda创建虚拟环境，确保依赖版本匹配（PyTorch框架核心依赖）：  

```bash  
# 1. 创建并激活虚拟环境  
conda create -n ViTKB-torch python=3.9
conda activate ViTKB-torch  

# 2. 安装PyTorch（支持GPU/CPU，示例为CPU版本）  
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu

# 3. 安装其他依赖库   
pip install numpy==1.24.4 matplotlib==3.5.2 opencv-python==4.11.0.86
pip install pandas==1.4.4 pillow==11.1.0 scikit-learn==1.0.2
pip install tqdm==4.67.1 tensorboard==2.11.2

```  

## 5 代码使用说明  

### 5.1 模型训练  

运行`train.py`脚本启动训练，支持通过参数调整训练配置，示例命令：  

```bash  
#启用KAN和BiFormer
python train.py \
  --use-biformer \
  --use-kan \
  --num_classes 4 \
  --data-path ./datasets \  # 数据集根目录
  --batch-size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --kan-lr 1e-4 \
  --device cpu
#冻结训练
python train.py \
  --use-biformer \
  --use-kan \
  --freeze-kan \
  --freeze-biformer \
  --num_classes 4 \
  --data-path ./datasets \  # 数据集根目录
  --batch-size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --device cpu
```  


#### 关键参数说明：  

| 参数名 | 含义 | 默认值 |
|-----------------|---------------------------------------|-----------------|
| `--data_dir` | 数据集根目录路径 | `./datasets` |
| `--epochs` | 训练轮数 | 50 |
| `--batch_size` | 批次大小（根据显存调整，8/16/32） |8 |
| `--lr` | 初始学习率 | 1e-4 |
| `--save_dir` | 训练权重保存目录（.pth格式） | `./weights` |
| `--device` | 训练设备（`GPU`或`CPU`） | `CPU` |  



### 5.2 模型预测  

使用训练好的权重进行单张大豆叶片图像预测，运行`predict.py`脚本，示例命令：  

```bash  
python predict.py\
--image-path ./examples/datasets/test/Grey spot/Grey spot_16.jpg\  # 输入图像路径
--weight_path ./weights/best_model.pth \  # 预训练权重路径
--device CPU  
```  


#### 预测输出示例：  

```  
输入图像路径： ./examples/datasets/test/Grey spot/Grey spot_16.jpg\  
预测类别：灰斑病（Grey spot）  
置信度：0.9742  
```  


## 6. 项目文件结构  

```  
ViTKB/  
├── datasets/            # 三类大豆病害数据集   
├── vit_model.py         #整体模型实现
├──ViT_base_model.py     #原ViT模型实现
├── biFormer.py          #BiFormer注意力机制实现  
├── kan.py              # KAN分裂注意力机制实现  
├── my_dataset.py        #自定义数据加载器 
├── class_indices.json      #数据集的类别及类别数
├── utils.py               #训练工具函数集   
├── train.py              # 模型训练脚本（PyTorch版）  
├── predict.py              # 模型预测脚本（PyTorch版）
├── vit_base_patch16_224_in21k.pth            # 模型预训练权重    
└── README.md             # 项目说明文档（本文档）  
```  


## 7. 已知问题与注意事项  

1. **框架适配**：本项目基于 PyTorch 2.0.1+ 和 Vision Transformer 架构；
2.**输入尺寸**：模型固定输入为 224×224×3，训练和预测时会自动进行 resize 和中心裁剪；
3.**模块配置**：支持通过命令行参数灵活启用 BiFormer 注意力和 KAN 激活函数；
4.**数据集扩展**：如需新增类别，需补充图像数据并修改 train.py 中的 --num_classes 参数；
5.**训练策略**：支持分层冻结、迁移学习和自定义学习率调度。

## 8. 引用与联系方式  

### 8.1 引用方式  

论文处于投刊阶段，正式发表后将更新BibTeX引用格式，当前可临时引用：  

```bibtex  
@article{ViTKB_Soybean_disease,  
title={ViTKB: A New Deep Learning model for Soybean Leaf Disease Identification},  
  author={[作者姓名，待发表时补充]},  
  journal={[期刊名称，待录用后补充]},  
  year={2025},  
  note={Manuscript submitted for publication}  
}  
```  


### 8.2 联系方式  

若遇到代码运行问题或学术交流需求，请联系：  
- 邮箱：zhumengyuanhuuc@yeah.net 
- GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。
