export function ForwardHierarchyHelp() {
  return (
    <details
      className="forward-hierarchy-help"
      onKeyDown={(event) => {
        if (event.key === 'Escape') {
          event.currentTarget.open = false;
          event.currentTarget.querySelector('summary')?.focus();
        }
      }}
    >
      <summary>分层规则</summary>
      <div className="forward-hierarchy-popover">
        <strong>按 forward 层级查看模型</strong>
        <p>手动搭建与模型导入共用以下约定：</p>
        <ol>
          <li>根画布对应主模型的 forward()，直接展示这一层的完整结构。</li>
          <li>本层的取值、切片、拼接、掩码等计算在本层表达，并保留条件和循环关系。</li>
          <li>调用的子模块保持封装；双击进入其 forward()，内部子模块继续保持封装。</li>
          <li>通过上方面包屑返回任意上层。进入与返回只切换视图，不改变计算和连接。</li>
        </ol>
        <p className="muted">
          当前支持已有子图的分层浏览；PyTorch 自动导入及完整控制流表达仍待实现。
        </p>
      </div>
    </details>
  );
}
