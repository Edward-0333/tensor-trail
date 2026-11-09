# Python CPU 环境

项目的 Python 输入采集工具使用独立 Conda 环境，位于项目根目录的 `.conda/tensor-trail-cpu/`。

核心版本与本机已验证的 `ndts_plan_win` CPU 环境一致：Python 3.11.18、PyTorch 2.11.0+cpu、NumPy 1.26.4。`requirements-cpu.txt` 锁定 PyTorch、NumPy 及其核心依赖。源环境没有额外的 Conda 环境变量需要迁移；激活时由 Conda 管理解释器路径。环境目录已加入 Git 忽略规则。

在项目根目录的 Anaconda Prompt 中激活：

```bat
conda activate .\.conda\tensor-trail-cpu
python -m unittest discover -s python -p test_tensor_trail_runtime.py
```

也可以不切换当前终端环境，直接运行测试：

```bat
conda run --no-capture-output -p .\.conda\tensor-trail-cpu python -m unittest discover -s python -p test_tensor_trail_runtime.py
```

其他电脑可在项目根目录重建相同的核心环境：

```bat
conda create -p .\.conda\tensor-trail-cpu python=3.11.18 pip -y
conda run -p .\.conda\tensor-trail-cpu python -m pip install -r python\requirements-cpu.txt
```

依赖清单指定 [PyTorch 官方 CPU 安装源](https://download.pytorch.org/whl/cpu/)，固定 `torch==2.11.0+cpu`。网页前端继续使用 `app/start.cmd` 或 npm 启动；本环境供 `tensor_trail_runtime.py` 和真实训练输入采集使用。操作流程见 [训练输入采集](../docs/训练输入采集.md)。
