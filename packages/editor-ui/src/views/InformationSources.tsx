import type { RootInputSource, ValueSpec } from '@tensor-trail/model-core/core/types';
import {
  isDictionarySpec,
  isModuleSpec,
  isTensorSequenceSpec,
} from '@tensor-trail/model-core/core/types';
import {
  VALUE_ORIGIN_LABELS,
  INITIALIZATION_LABELS,
} from '@tensor-trail/model-core/core/informationSources';

const sourceColors = [
  '#2563eb',
  '#b45309',
  '#0f766e',
  '#be185d',
  '#4d7c0f',
  '#475569',
  '#9f1239',
  '#0369a1',
];
export function sourceColor(index: number): string {
  return sourceColors[index - 1] ?? `hsl(${Math.round((index * 137.508) % 360)} 65% 32%)`;
}

export function RootSourceBadge({ source }: { source: RootInputSource }) {
  return (
    <span
      className="lineage-source"
      data-source-id={source.id}
      data-source-index={source.index}
      style={{ backgroundColor: sourceColor(source.index) }}
      title={`根层输入 ${source.index}：${source.label}`}
      aria-label={`来源 ${source.index}：${source.label}`}
    >
      {source.index}
    </span>
  );
}

export function LineageBadges({ value }: { value?: ValueSpec }) {
  const lineage = value?.lineage;
  if (!lineage || (!lineage.sources.length && !lineage.unknown)) return null;
  return (
    <div
      className="lineage-badges"
      aria-label="根层输入来源"
      title="编号只由根层原始输入产生，沿数值依赖传播；与梯度和本层 IN 序号无关。"
    >
      {lineage.sources.map((source) => (
        <RootSourceBadge key={source.id} source={source} />
      ))}
      {lineage.unknown && (
        <span
          className="lineage-unknown"
          title="存在尚未支持分析的路径或达到分析上限，来源可能不完整"
          aria-label="来源待确认"
        >
          ?
        </span>
      )}
    </div>
  );
}

export function LineageDetails({ value }: { value?: ValueSpec }) {
  if (!value?.lineage) return null;
  const lineage = value.lineage;
  const container = isDictionarySpec(value) || isTensorSequenceSpec(value) || isModuleSpec(value);
  return (
    <section className="lineage-details" aria-label="信息来源详情">
      <div className="section-label">
        根层输入来源 · {lineage.evidence === 'root' ? '自动编号' : '静态传播'}
      </div>
      <LineageBadges value={value} />
      <p>
        来源：
        {lineage.sources.map((source) => `${source.index} = ${source.label}`).join('；') ||
          (lineage.unknown ? '尚无法确认' : '未接入根层输入的数值信息')}
      </p>
      {!!lineage.origins.length && (
        <p>其他来源：{lineage.origins.map((origin) => VALUE_ORIGIN_LABELS[origin]).join('、')}</p>
      )}
      {!!lineage.initializations.length && (
        <p>
          初始化历史：
          {lineage.initializations.map((kind) => INITIALIZATION_LABELS[kind]).join('、')}
          。不表示当前数值仍是初始值。
        </p>
      )}
      {lineage.unknown && (
        <p className="microcopy">存在未确认的来源或算子；已知编号不是完整来源集合。</p>
      )}
      {lineage.origins.includes('state') && (
        <p className="microcopy">模型状态可能来自历史更新或 checkpoint，不按构造初值当作常量。</p>
      )}
      {container && (
        <p className="microcopy">
          这里只汇总容器内的来源，不表示各元素已经特征融合；读取元素后按该元素追溯。
        </p>
      )}
      <p className="microcopy">
        编号仅代表静态数值依赖，不衡量融合效果。shape、dtype、device 不传播输入编号，detach
        不清除编号。没有编号也不等于“随机”或“没有梯度”。
      </p>
    </section>
  );
}
