# Mini Pi Plus based on BeyondMimic 

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/download.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab/v2.2.0/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)
![gif](https://github.com/Daily-study-HT/bydmimic_publish/blob/main/gif/6363667a0f27da450e1059a30c2b274b.gif)
## 介绍

此项目是基于 [Beyondmimic](https://github.com/HybridRobotics/whole_body_tracking) 进行修改建立的。原项目是由作者qiayuanl提出的一个多功能的人形机器人控制框架，它能够在实际部署中提供高度动态的运动跟踪和最先进的运动质量，并通过基于引导扩散的控制器提供可控的测试时间控制。

此 repo 涵盖了运动追踪训练，提供了高擎机电机器人的asset，并针对高擎机电机器人进行了配置参数的调优。**您能够
在所提供的motion文件夹下的数据集中训练可模拟到现实的运动，而无需调整任何参数**。

## 安装

按照[安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Sim v5.0.0 、 Isaac Lab v2.2.0 和 rsl_rl 2.3.1。我们建议使用 conda 安装，因为它可以简化从终端调用 Python 脚本的操作（备注:Isaac Sim v4.5,Isaac Lab v2.1.0环境也可以）。

- 将此代码库与 Isaac Lab 安装目录分开克隆（即，克隆到`IsaacLab`目录之外）：

```bash
git clone https://github.com/HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic
```

- 使用安装了 Isaac Lab 的 Conda 虚拟环境安装本项目，进入项目目录后，执行

```bash
python -m pip install -e source/whole_body_tracking
```

## 动作跟踪

### Motion Preprocessing & Registry Setup

为了便于检查数据内容，我们在`source/motion`中提供了csv版本的重定向数据文件，在进行训练前需要转换为npz格式。

```bash
# 格式转换
python scripts/csv_to_npz.py --robot pi_plus --input_file source/motion/hightorque/pi_plus/csv/pi_plus_right_big_kick_cut.csv --input_fps 30 --output_name source/motion/hightorque/pi_plus/npz/{motion_name}
#若不想加载图形化界面则添加参数 --headless

# 数据播放
python scripts/replay_npz.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz 
```

### 模型训练

通过以下命令训练策略：

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-PI-Plus-Wo-v0 --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz --headless --log_project_name pi_plus_beyondmimic
#若不想使用wandb，可以去掉 --logger wandb
#继续训练 --resume {load_run_name}
```
### 模型导出

通过以下命令play训练好的策略：
```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-PI-Plus-Wo-v0 --checkpoint {logs_path_to}/model_xxx.pt --num_envs=1 --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz
```
![if](https://github.com/Daily-study-HT/bydmimic_publish/blob/main/gif/e7faf89fbdbf87cf909bbf81ceeb1a7f.gif)

### 模型评估

通过以下命令在mujoco中验证策略：

```bash
python scripts/sim2sim.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz --xml_path source/whole_body_tracking/whole_body_tracking/assets/hightorque/pi_plus/mjcf/pi_20dof.xml --policy_path {logs_path_to}/exported/{model_xxx}.onnx --save_json --loop
#使用--loop可以循环播放策略
```
![f](https://github.com/Daily-study-HT/bydmimic_publish/blob/main/gif/de78f3ab232911f9a93e936cb5463164.gif)
## 代码结构

以下是此项目的代码结构概述：

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  此目录包含用于定义 BeyondMimic 的 MDP 的原子函数。以下是这些函数的细分：
    - **`commands.py`**
      命令库用于根据参考运动、当前机器人状态和误差计算相关变量。
      计算包括位姿和速度误差计算、初始状态随机化和自适应采样。

    - **`rewards.py`**
      实现 DeepMimic 奖励函数和平滑项。

    - **`events.py`**
      实现域随机化项。

    - **`observations.py`**
      实现运动跟踪和数据收集的观测项。

    - **`terminations.py`**
      实现提前终止和超时。

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  包含跟踪任务的环境（MDP）超参数配置。

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  包含跟踪任务的 PPO 超参数。

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  包含机器人特定设置，包括骨架参数、关节刚度/阻尼计算和动作比例计算。

- **`scripts`**
  包括用于预处理运动数据、训练策略和评估训练策略的实用脚本。

该结构旨在确保开发人员扩展项目的模块化和易于寻找。


## 原项目地址
[whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
