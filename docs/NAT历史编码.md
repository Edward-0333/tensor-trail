# PLUTO 历史编码器与第 80–82 行

本次画布覆盖 `AgentEncoder.forward()` 的以下调用，到 `x_agent_tmp` 为止：

```python
x_agent_tmp = self.history_encoder(
    agent_feature[valid_agent_mask].permute(0, 2, 1).contiguous()
)
```

`agent_encoder` 画布只显示数据准备与 `history_encoder` 调用。双击 `history_encoder` 进入 `NATSequenceEncoder.forward()`，再双击各级 `levels[i]`、`blocks[i]`、`mlp`、`embed` 或 `downsample` 继续查看其内部。层级对应本地 `pluto/src/models/pluto/layers/embedding.py` 的模块边界。

## 输入准备

| 节点 | 输入 | 输出 |
| --- | --- | --- |
| 布尔索引 `x[mask]` | `agent_feature [B×A,20,9]`、`valid_agent_mask [B×A]` | `[N_valid,20,9]` |
| Permute | `[N_valid,20,9]` | `[N_valid,9,20]` |
| Contiguous | `[N_valid,9,20]` | 同形状、同 dtype |
| history_encoder | `[N_valid,9,20]` | `[N_valid,128]` |

`N_valid` 是 `count_nonzero(valid_agent_mask)`，不是 `B×A`，也不是固定常量。项目输入记录目前只有 shape 与 dtype，没有掩码数值，因此保留这个符号。布尔索引节点检查一维 bool 掩码与输入第 0 轴相匹配；现已扩展为匹配前若干轴的多维 bool 掩码；本处仍使用一维掩码。

## NATSequenceEncoder 内部

当前生成器覆盖 PLUTO 使用的三级结构与 `out_indices=[0,1,2]`，默认参数为输入通道 9、首层宽度 32、深度 `[2,2,2]`、头数 `[2,4,8]`、窗口 `[3,3,5]`、MLP 扩展倍数 3。节点参数可调整各级深度、宽度、头数和丢弃概率；参数修改会重新生成内部图。工程超参数用于本次创建时的初始化快照，之后可以在节点属性中修改；没有把局部参数副本用于代替张量数据连线。

| 位置 | 形状（默认历史长度） |
| --- | --- |
| ConvTokenizer：Conv1d + Permute | `[N_valid,20,32]` |
| level 0 的 xo / 下一层 x | `[N_valid,20,32]` / `[N_valid,10,64]` |
| level 1 的 xo / 下一层 x | `[N_valid,10,64]` / `[N_valid,5,128]` |
| level 2 的两个返回值 | 均为 `[N_valid,5,128]` |
| 三路 norm + Permute + Contiguous + lateral Conv1d | `[N_valid,128,20/10/5]` |
| 深层插值到中层并相加，再插值到浅层并相加 | `[N_valid,128,20]` |
| fpn_conv | `[N_valid,128,20]` |
| `out[:, :, -1]` | `[N_valid,128]` |

每个 NATBlock 的 `xo` 是下采样前结果，`next` 是下一层的输入。最后一级不下采样，因此返回同一结果两次。ConvDownsampler 保留 kernel=3、stride=2、padding=1、bias=False 和后续 LayerNorm。内部的 NATConv1d 使用已有 Conv1d 形状规则，并额外显式记录 bias。

每个 NATLayer 保留两次残差相加：

```python
x = shortcut + drop_path(attn(norm1(x)))
x = x + drop_path(mlp(norm2(x)))
```

六个 NATLayer 的 DropPath 按全编码器线性递增，默认分别为 `0, 0.04, 0.08, 0.12, 0.16, 0.2`。同一层两次调用共用相同概率，但表示两次采样，不能将两次调用当作同一次计算结果。MLP 保留 Linear → GELU → Dropout → Linear → Dropout。

## 函数代码块

计算库增加“函数代码块”，当前提供三个具有固定输入输出规则的模板：

- `NeighborhoodAttention1D`：检查特征维度、头数、邻域大小和概率，保留 `[N,L,C]`；右侧展示 NATTEN 调用及计算过程说明。它包含可训练投影和相对位置偏置，不等同于全局多头注意力。参照本地 NATTEN 0.14.6，短序列会补齐到窗口长度后裁回。
- `DropPath`：记录训练/推理语义与概率，保持 shape。
- `Interpolate1D`：输入 `in` 和长度参考 `like`，按 `like.shape[-1] / in.shape[-1]`、`mode="linear"`、`align_corners=False` 插值，只采用 `like` 的末轴长度。

这些代码在右侧“函数代码”区域阅读，参数可编辑，代码模板由实现生成。网页不运行 Python/CUDA，也不支持任意 Python 文本的自动形状推导。Tensor Trail 仍是结构与形状工作台；没有把函数块当成可直接训练的模型导出。

## 验证

使用本地 PLUTO 与 NATTEN，在 CPU、eval、随机权重、batch=3 下分别运行长度为 20、21、7 的真实输入，记录 tokenizer、各 level、侧向卷积、FPN 与最终输出形状，与画布推导一致。三个测试输出均为 `[3,128]`；这只是前向验证，没有训练或生成 cache。自动测试还覆盖错误掩码、无效配置、残差连接、FPN 融合顺序、短序列和序列化重开。

## 第 83–85 行：恢复完整车辆维度

在 `agent_encoder` 中继续接入：

```python
x_agent = torch.zeros(bs * A, self.dim, device=position.device)
x_agent[valid_agent_mask] = x_agent_tmp
x_agent = x_agent.view(bs, A, self.dim)
```

新增“零张量 Zeros”和“布尔索引赋值”，画布分别显示 `zeros()`、`x[mask] = v`，代码和参数在右侧查看。View 复用现有模块。当前输入 B=2、A=49、dim=128，对应 `[98,128] → [98,128] → [2,49,128]`。

- `Zeros` 通过尺寸接口接收 `bs × A` 与 `dim`，通过 `device_like` 接收原 `position`。该连线只表示设备依赖，不复制其形状或 dtype；当前默认 dtype 记录为 float32。画布不分配实际内存。
- `BooleanAssign` 接收原 `valid_agent_mask` 和 `x_agent_tmp`，按 True 行回填；未选中行保持零。当前支持多维前缀 bool 掩码，默认不广播；可在节点中显式开启广播。此处继续采用一维掩码、不广播的逐行赋值，校验掩码长度、尾轴及 dtype。`N_valid` 的实际计数依赖运行时数据。
- `bs`、`A` 和 `dim` 使用局部参数关联副本；张量从原节点连线。三个 `x_agent` 展示分别表示初始化、回填、恢复形状后的不同数据流版本，不设置值关联。

使用 PyTorch 验证了部分有效、全部有效、全部无效三类掩码，输出位置、未选中行零值，以及回填值的反向梯度均符合原代码。

`AgentEncoder` 第 87–94 行的 ego/type embedding 分支与返回值现已补齐，详见 [AgentEncoder 完整流程](AgentEncoder完整流程.md)。历史编码器部分的结构保持不变。
