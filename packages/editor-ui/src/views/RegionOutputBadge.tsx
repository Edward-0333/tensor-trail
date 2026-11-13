import type { CSSProperties } from 'react';

export interface RegionOutputInfo { id: string; label: string; color: string }

export function RegionOutputBadge({ region }: { region: RegionOutputInfo }) {
  return <span className="region-output-badge nodrag nopan" data-region-output={region.id}
    style={{ '--region-color': region.color } as CSSProperties}
    aria-label={`分区输出：${region.label}`}
    title={`属于“${region.label}”的分区输出。仅说明分区边界角色，不新增 forward 接口，不改变梯度或最终返回值。`}>
    分区输出 · {region.label}
  </span>;
}
