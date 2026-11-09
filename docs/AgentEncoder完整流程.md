# PLUTO AgentEncoder 的完整画布

`pluto.tensor-trail.json` 已覆盖 `AgentEncoder.forward()` 第 55–94 行。主模型中的 `agent_encoder` 现在具有 `out` 接口，并连接到主画布的 `x_agent`，当前记录的输出形状为 `[2,49,128]`。PLUTO 主模型剩余部分尚未绘制，因此主模型没有最终 Output 的提示仍然保留。

## 第 87–94 行

```python
if not self.use_ego_history:
    ego_feature = data["current_state"][:, : self.state_channel]
    x_ego = self.ego_state_emb(ego_feature)
    x_agent[:, 0] = x_ego

x_type = self.type_emb(category)
return x_agent + x_type
```

在 AgentEncoder 画布中，“if not self.use_ego_history”是条件代码区域，不代表 PLUTO 中新增了 nn.Module。它接收原 `data` 和恢复形状后的 `x_agent`；双击进入查看字段读取、切片、`ego_state_emb` 调用以及整数索引赋值。当前 use_ego_history=False，条件为真，执行替换。将此节点的 use_ego_history 设为 True 会重建成直通路径，内部不再创建或计算当前状态编码器。

当前 state_attn_encoder=True、state_channel=6、dim=128、state_dropout=0.75；训练模式记录为 True。节点参数是创建时的配置快照，修改项目超参数不会自动重建已有子图。应在条件节点中修改相应参数，以重建整个分支；训练/推理切换也在此处设置。这样不会把配置变化误表示为已经重新采集过模型。

形状依次为：

- `data["current_state"] [2,7] → 切片 [2,6] → ego_state_emb [2,128]`。
- `x_agent[:,0] = x_ego` 覆盖第 0 个 agent，输出仍为 `[2,49,128]`，其他 agent 保持原值。
- `type_emb` 为 `nn.Embedding(4,128)`，将 `category [2,49]` 的整数索引转换为 `[2,49,128]`。
- 两路相加后返回 `[2,49,128]`。历史分支中无效 agent 曾置零，但加上类别嵌入后，这些位置不保证为零。

没有新增张量关联副本。张量从原节点连线；条件区域通过模块输入接收上一级的同一数据。

## StateAttentionEncoder

双击条件区域中的 `ego_state_emb`：

1. 六个独立的 `Linear(1,128)` 分别编码六个状态标量，各路权重不共享。
2. Stack 得到 `[B,6,128]`，加上按 batch 重复的可训练 `pos_embed [1,6,128]`。
3. 可训练 `query [1,1,128]` 按 batch 重复，得到 `[B,1,128]`。
4. “状态遮挡 + attn”函数块执行原代码的状态遮挡和 `nn.MultiheadAttention` 调用，输出 `[B,1,128]`。
5. `x_state[:,0]` 返回 `[B,128]`。

`pos_embed` 和 `query` 使用可训练参数节点，明确区别于超参数或常量。当前正态初始化 std=0.02，画布记录结构与类型，不生成或冒充训练权重数值。

训练且 state_dropout>0 时，前三个状态 token 的 mask 始终为 False，其余 token 按概率遮挡；eval 或概率为零时，key_padding_mask=None。注意力使用 4 个头、batch_first=True，Q/K/V 与输出投影均可训练。函数块右侧显示代码；网页不执行 Python 或随机采样。

state_attn_encoder=False 时，条件区域改用原代码的 `Linear(bias=False) → BatchNorm1d → ReLU → Linear(bias=True)`，没有将该配置替换成注意力。

## 验证

- 自动测试覆盖分支跳过、MLP 替代分支、Embedding、整数赋值、独立 Linear、参数重复、注意力形状、概率边界和序列化。
- 使用本地真实 PLUTO 在 CPU 上核对完整 AgentEncoder 的训练/推理、使用/不使用自车历史、注意力/MLP 配置，输出与按历史编码、ego 替换、类型嵌入重建的结果一致。
- 核对真实 StateAttentionEncoder 的训练掩码、eval 的 None 掩码，以及 pos_embed、query 和状态输入的反向梯度。未启动训练任务。
- 浏览器验证子画布进入、代码详情、无新增节点重叠和保存重开。
