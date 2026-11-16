import { Star } from 'lucide-react';
import type { ModelNode } from '@tensor-trail/model-core/core/types';
import { layoutRegionImportance, REGION_IMPORTANCE_LABELS } from '@tensor-trail/model-core/core/layoutRegions';

export function RegionImportance({ model }: { model: ModelNode }) {
  const rating = layoutRegionImportance(model);
  const description = rating ? `本层重要性：${rating}/5星 · ${REGION_IMPORTANCE_LABELS[rating]}` : '本层重要性：未评级';
  const reason = String(model.params.importance_note ?? '').trim();
  return <span className="region-importance" data-importance={rating} role="img" aria-label={description}
    title={`${description}。仅比较当前画布内的分区，不跨层级比较。${reason ? `\n依据：${reason}` : ''}`}>
    {rating ? <><small>本层</small><span className="region-importance-stars" aria-hidden="true">
      {[1, 2, 3, 4, 5].map(value => <Star key={value} size={12} className={value <= rating ? 'filled' : ''} />)}
    </span></> : <><Star size={12} aria-hidden="true" /><small>未评级</small></>}
  </span>;
}
