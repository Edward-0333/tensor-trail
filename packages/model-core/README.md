# 模型核心

`core/` 定义图协议、序列化、形状推导与校验；`modules/` 定义算子；`agent/` 提供构建与修改 API；`editing/` 保存纯图复制和关联值编辑算法。

常用入口：

```ts
import { buildModel, validateModel, patchModel } from '@tensor-trail/model-core/agent';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { inferProject } from '@tensor-trail/model-core/core/engine';
import { registry } from '@tensor-trail/model-core/modules';
```

这些源码入口用于 TypeScript 工具链；原生 Node 调用独立 SDK `dist/tensor-trail-agent.js`，先在仓库根执行 `npm run build:agent`。CLI 用法见[AI 操作指南](../../docs/AI操作指南.md)。

本包单独使用 ES2022 与 Node 类型检查，不引入 DOM、React 或 Electron。新增算子或图规则继续保持这一边界。
