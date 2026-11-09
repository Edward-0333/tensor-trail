# PLUTO MapEncoder 完整画布

`pluto.tensor-trail.json` 的主画布新增 `data → map_encoder → x_polygon`，对应 `PlanningModel.forward()` 第 146 行。原有节点、位置和连线保持不变。主模型后续模块尚未绘制，顶层最终 Output 的提示仍保留。

## 层级和当前配置

- 双击 `map_encoder`：查看 `MapEncoder.forward()` 的字典取值、几何特征拼接、类别/route/红绿灯嵌入、限速编码与两次回填，最后相加返回。
- 双击 `polygon_encoder`：查看 `PointsEncoder.forward()`；再进入 `first_mlp` 或 `second_mlp`，查看 Linear、BatchNorm1d、ReLU、Linear。
- 双击 `speed_limit_emb`：查看 `FourierEmbedding.forward()`；频率参数、正余弦编码、各独立 MLP、Stack、Sum、to_out 都有节点。

当前主模型在构造时显式设置 `use_lane_boundary=True`，因此基础 6 维特征再加 4 维边界偏移，实际为 10 维。dim=128，训练模式记录为 True。节点参数是构造快照，在 MapEncoder 节点详情中修改会重建相应子图；修改项目超参数不会自动重建已有图。关闭边界开关后只保留三路几何特征，通道恢复为 6。

当前已保存输入规格为 B=2、M=58、P=20；M 来自此样本，不是固定超参数：

| 位置 | 形状 |
| --- | --- |
| polygon_feature | `[2,58,20,10]` |
| feature reshape / mask view | `[116,20,10]` / `[116,20]` |
| PointsEncoder | `[116,128]` |
| 恢复地图维度 | `[2,58,128]` |
| 有限速元素选择 | `[N_speed]` |
| FourierEmbedding 输入 / 输出 | `[N_speed,1]` / `[N_speed,128]` |
| 未知限速权重 | `[1,128]`，广播回填 |
| 最终 x_polygon | `[2,58,128]` |

只有局部参数采用关联副本；张量多处使用时从原节点连线。`N_points`、`N_speed`、`N_unknown_speed` 是掩码 True 数量的符号，未从形状记录猜测其数值。

## 本次补齐的通用操作

1. BooleanIndex：多维 bool 掩码必须逐轴匹配输入的前缀轴，不对掩码做广播。输出为 `[count_nonzero(mask), ...剩余轴]`。当前不支持零维 bool 索引。
2. BooleanAssign：支持同样的多维前缀掩码。默认仍不广播，兼容旧图；开启“允许赋值广播”后，写入值从尾轴对齐到索引结果，支持这里的 `[1,128]` 行权重或标量。写入值 rank 不超过索引结果，目标形状不会扩张。
3. Max：指定 dim 和 keepdim，只返回 `.max(...)[0]` 的 values。保留未归约轴含义；空归约轴报错。
4. Sum：指定 dim 和 keepdim；整数和 bool 默认输出 int64。
5. Repeat：各轴重复次数可以是常量或 d0–d7 的局部整数输入，支持新增前导轴和零次重复。它与广播视图有区别。
6. Reshape：保留旧模块类型和纯数字配置，补上 d0–d7 动态尺寸输入；允许一个 -1，并校验元素数量。语义保持为按需返回视图或复制，不强加 View 的 stride 限制。

新操作的画布只显示紧凑名称，参数和表达式在右侧。旧图没有广播标志时仍按不广播处理。BatchNorm1d 的二维输入规则也核对了 PyTorch：训练时拒绝单个样本，但允许空 batch。

## 必须保持的计算语义

PointsEncoder 两组 MLP 的 Linear 都带 bias。每次 MLP 只处理有效点，随后将无效点填零再做 Max。填零点参与最大值计算；不能换成忽略无效点的 masked max。当有效特征为负且存在填零点时，最大值可能为零。

`freqs.weight` 与 `unknown_speed_emb.weight` 是可训练参数，不是常量或重复创建的独立权重。当前记录 PlanningModel 的 Embedding 初始化规则 std=0.02。FourierEmbedding 的 64 个频率产生 `64+64+1=129` 维正余弦与原值拼接，再经 MLP 编码。当前 input_dim=1，但图模板也支持多维连续输入的独立 MLP 和求和。

未知限速位置广播写入同一份 `[1,128]` 权重，其梯度来自所有写入位置。最终遵守原代码的属性求和顺序，再加到几何特征；填零点或空几何特征不表示最终输出也必为零。

## 验证

- TypeScript 测试覆盖多维索引、掩码不广播、广播回填、空集合、Max/Sum、动态 Repeat/Reshape、两种车道边界配置、独立 MLP、序列化和旧图兼容。
- 用本地真实 PLUTO CPU 计算核对训练/推理、边界开启/关闭、全部/部分/没有限速、所有地图点无效的结果；逐项重建 PointsEncoder 和属性融合结果一致。前向 hook 保存了输出副本，避免原代码 `x_polygon += ...` 修改早先记录的共享 Tensor。
- 单独验证负特征遇到填零点的 Max 行为、FourierEmbedding 空集合/多维输入及频率权重梯度，以及未知限速权重广播梯度累加。
- 浏览器检查所有子图可进入、节点无新增重叠、紧凑操作显示、保存重开，以及 AgentEncoder 仍输出原形状。

此实现用于结构、形状和数据流可视化；网页没有执行训练张量或加载真实权重数值，也没有启动训练或生成缓存。
