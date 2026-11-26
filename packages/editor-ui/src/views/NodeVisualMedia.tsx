import type { NodeVisualMedia as Media } from '@tensor-trail/model-core/core/types';

export const defaultTileWidth = 144;

export function NodeVisualMedia({ media }: { media: Media }) {
  const { items, layout = 'auto', tileWidth = defaultTileWidth } = media;
  let columns = Math.min(items.length, 3);
  if (layout === 'column') columns = 1;
  else if (layout === 'row') columns = items.length;
  else if (layout === 'grid' || items.length === 4) columns = 2;
  return (
    <div
      className="node-visual-media"
      role="group"
      aria-label="节点插图"
      style={{ gridTemplateColumns: `repeat(${columns}, ${tileWidth}px)` }}
    >
      {items.map((item) => (
        <figure key={item.id} data-emphasis={item.emphasis ?? 'normal'}>
          <div className="media-frame">
            <img src={item.src} alt={item.caption ?? ''} draggable={false} />
            {item.badge && <span className="media-badge">{item.badge}</span>}
          </div>
          {item.caption && <figcaption>{item.caption}</figcaption>}
        </figure>
      ))}
    </div>
  );
}
