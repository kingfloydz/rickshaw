# mjlab 黄包车牵引力训练

这是一个基于 mjlab 的平地黄包车速度跟踪任务。策略输出左右牵引点在车体坐标系下的三维力，动作维度为 6；每侧实际力的二范数限制为 `50 N`，并在车体质心处写入等效力矩。

任务 ID：`Mjlab-Rickshaw-Force-Flat`

## 观测

Actor 为 25 维，顺序为：

1. 目标前进速度、目标偏航角速度（2）
2. 车体坐标系线速度（3）
3. 车体坐标系角速度（3）
4. 投影重力（3）
5. 左右牵引点到地面的法向高度（2）
6. 上一时刻左右实际牵引力（6）
7. 上上时刻左右实际牵引力（6）

Critic 在 Actor 观测后追加左右轮角速度、地面坡度和左右轮接触状态，共 31 维。接触状态使用最近 3 个物理子步的历史；当前平地任务的坡度为零。

## 运行

安装 mjlab 和本任务包：

```bash
uv sync
```

训练：

```bash
uv run train Mjlab-Rickshaw-Force-Flat --env.scene.num-envs 4096
```

查看随机动作：

```bash
uv run play Mjlab-Rickshaw-Force-Flat --agent random
```

## 资产说明

`asset_zoo/rickshaw/xmls/rickshaw.xml` 根据目录中的 `rickshaw.urdf` 建立 MJCF。轮子碰撞体使用 URDF 中的圆柱参数；车体保留 STL 视觉网格，车轮负责与水平 plane 的接触。初始化姿态由轮子半径、牵引点局部坐标和目标高度解析求得，使轮胎落地且两个牵引点高度为 `0.75 m`。


uv run play Mjlab-Rickshaw-Force-Flat \
  --checkpoint-file /inspire/hdd/project/leverage-robot/ky26212/rickshaw/logs/rsl_rl/rickshaw_force_velocity/2026-07-24_06-15-22/model_200.pt \
  --num-envs 19 \
  --device cuda:0 \
  --viewer viser



git pull origin main

git add .

git commit -m "code"

git push origin main



cd /inspire/hdd/project/leverage-robot/ky26212/rickshaw

uv run tensorboard \
  --logdir logs/rsl_rl/rickshaw_force_velocity \
  --host 0.0.0.0 \
  --port 6006


git pull --ff-only



export PATH="$HOME/.local/bin:$PATH"
