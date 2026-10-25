# Criteo 项目运行说明

项目使用 Criteo Uplift Prediction Dataset v2.1，分析公开样本中的组间差异、统计区间和投放排序表现。公开数据经过非均匀抽样，结果按样本内描述性比较解释。完整方法见英文 [README](README.md)，主要结果见 [CONCLUSION](CONCLUSION.md)。

## 准备环境

解压 `criteo-ab-uplift-final.zip`，保留完整项目目录。在包含 `run.sh`、`src/` 和 `sql/` 的目录中运行程序；单独下载说明文件无法运行分析。

安装 Python 3.12。Apple Silicon Mac 使用 ARM64 或 universal2 版本，无需 GPU。首次运行需要联网下载依赖和数据。

## 运行

在终端中进入项目目录，执行：

```bash
bash run.sh
```

Mac 也可双击 `Run_Project.command`。程序会建立独立环境、安装依赖、下载并校验数据，完成统计分析、模型训练和评估，随后生成图表、执行 notebooks，并更新 README 和结论。已完成的数据处理和模型计算会从缓存读取。

如需指定 Python 路径：

```bash
PYTHON_BIN=/path/to/python3.12 bash run.sh
```

Linux 使用相同入口；在 HPC 上，应在调度器分配的计算节点中运行。

## 结果文件

- `README.md`：研究方法、主要结果和参考文献。
- `CONCLUSION.md`：结果摘要。
- `notebooks/`：五份含已执行输出的分析说明，读取保存的结果。
- `results/`：数值结果、图表和运行记录。

交付包包含上述结果。重新运行会生成新的结果目录，并保留已有结果。

## 提交到 GitHub

提交代码、`sql/`、`notebooks/`、`results/`、说明文件和许可文件。`.gitignore` 已排除原始数据、模型缓存、虚拟环境和临时文件；通过网页上传时也需排除 `data/` 和 `.venv/`。

数据通过下载脚本获取，遵循 **CC BY-NC-SA 4.0**，不随仓库分发。项目代码使用 **MIT** 许可。
