# 大论文第三章：静态语义重要性排序导向的语义通信跨层优化：瞬时语义成功概率

## 1. 简介
本代码是学位论文第三章的支撑代码。
主要实现了：
- 基于香农公式的联合离散-连续动作优化环境（OptimizationEnv）
- Multi-Pass P-DQN 强化学习算法
- 系统成功传输概率 p_succ 的最大化

## 2. 环境配置
- 编程语言：Python 3.8+
- 核心依赖库：PyTorch 1.12.1, NumPy 1.23.5, Gym 0.21.0, Matplotlib 3.6.2
- 见根目录 requirements.txt

## 3. 文件结构说明
- `src/main_pdqn_optimization.py`：核心算法入口文件，包含环境、网络、Agent和训练脚本
- `requirements.txt`：Python依赖列表

## 4. 如何运行
1. 安装依赖：`pip install -r requirements.txt`
2. 运行训练：`python src/main_pdqn_optimization.py`

