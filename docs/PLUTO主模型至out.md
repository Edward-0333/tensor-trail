# PLUTO 主模型：场景编码至 out 字典

本次画布对应 `pluto_model.py` 第 148–176 行。入口沿用此前的三个编码器结果，结束于 `out = {...}`；后续 hidden/ref_free 分支、推理后处理和 `return out` 尚未绘制。

## 主画布

1. 拼接 agent、map、static 特征，得到 `x [2,109,128]`。
2. 拼接静态对象位置，执行 `pos_emb = FourierEmbedding(3,128,64)`。
3. 合并 padding mask `[2,109]`，将位置编码加到 `x`。
4. 按实际配置展开四个独立的 `encoder_blocks[i]`，再做 `norm`。DropPath 概率来自实际模型的 float32 linspace；attention dropout 和 MLP dropout 均为源码默认 0。
5. `x[:,1:A]` 只取其他 agent，进入 `agent_predictor`。局部尺寸 `A` 使用已有节点的关联副本。
6. 读取 `reference_line.position.shape[1]`，比较 `R > 0`，进入参考线条件分支。
7. 组装 `out` 字典，保留三个字段的实际类型，不将它们拼成 Tensor。

当前输入采样下：

| 字段 | 形状 | 内容 |
| --- | --- | --- |
| trajectory | `[2,3,12,80,6]` | 自车候选轨迹，按参考线和模式组织 |
| probability | `[2,3,12]` | 未归一化的 logits；此处没有 softmax 或 top1 |
| prediction | `[2,48,80,6]` | 周围 agent 的位置、朝向向量和速度 |

最后一维 6 为 `loc(2) + yaw(2) + vel(2)`。主模型中 `prediction` 注释里的末维 2 与实际代码不一致，画布按实际输出记录。2、49、58、3 等是该批次尺寸，不是固定模型容量。

## 子画布

- `encoder_blocks[i]`：pre-norm → 带 padding mask 的注意力 → DropPath/残差 → pre-norm → MLP → DropPath/残差。
- `agent_predictor`：三个独立 MLPLayer，分别输出 loc/yaw/vel，经 View 后 Cat。
- `if ref_line_available`：双击进入当前条件分支，再进入 `planning_decoder`。
- `planning_decoder`：参考线特征和位置编码 → 参考线/模式 query → 四层 DecoderLayer → loc/yaw/vel/pi 四个独立头。
- `decoder_blocks[i]`：R2R、M2M、场景 cross-attention、FFN 和各残差。复杂重排/筛选暂时使用固定函数代码模板，选中后在右侧查看。R2R/M2M 分组的输出统一恢复为 `[B,R,M,D]`，与源码的计算顺序及结果等价。

模式参数 `m_emb`、`m_pos` 是独立可训练参数，初始化 std=0.01。当前 `cat_x=False`。图中保留源码 R2R 的 `mask.repeat(M,1)`，没有修改 PLUTO 算法。

## 条件与类型

新增通用字典组装、值变量、整数大于和 Squeeze 节点。Python bool 与 None 均为独立值类型，None 不模拟成空 Tensor。

参考线条件使用**当前输入形状的分支快照**，不是通用动态控制流执行器。当采集的新输入使 R 在 0 / 非 0 之间变化时，在分支节点修改“参考线存在”配置以重建子画布；校验器会报出快照与条件不一致，避免静默绘制错误分支。R=0 时子图不包含 PlanningDecoder，输出 trajectory/probability=None，prediction 仍正常计算。R>0 但全部参考线点无效并不等价于 R=0，这一点与源码相同。

项目超参数用于记录当前配置，已创建复合模块使用自己的配置快照；全局参数修改不会自动重建全部模块。计算库只推导结构和形状，不执行 PyTorch 或检查实际 Tensor 有限值。

由于只画到字典构建，完整模型验证仍保留原有的 `MISSING_OUTPUT` 提示。不要把这一处 out 标成函数最终返回，以免掩盖尚未绘制的后续代码。

## 验证

- 新模块测试覆盖注意力 mask 约束、三个预测头、cat_x 两种配置、R=0 / R>0 分支、字典 None、动态端口及保存重开。
- 使用本地 mini cache 的两个真实训练场景，CPU / train 模式 / no_grad 下核对当前输出及四层 encoder、四层 decoder 的逐块数值；未计算 loss、未反向传播、未执行 optimizer 或写 checkpoint。
- 浏览器检查主图、嵌套子图、None 显示、节点不重叠和保存重开。既有图内容保持不变；仅新增局部参数的关联副本。
