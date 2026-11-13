import { useEffect, useRef } from 'react';
import { ArrowUp, ChevronRight, Network } from 'lucide-react';
import type { ForwardLevel } from '@tensor-trail/model-core/core/hierarchy';

// Hide generated navigation suffixes, not identifiers such as feed_forward.
export function hierarchyLabel(title: string): string {
  return title.replace(/ · forward\(\)$/, '').replace(/ · 参数初始化（非 forward）$/, ' · 参数初始化');
}

export function HierarchyNavigation({ levels, onNavigate }: {
  levels: ForwardLevel[];
  onNavigate: (path: string[]) => void;
}) {
  const scrollRef = useRef<HTMLElement>(null);
  const current = levels.at(-1);
  const pathKey = JSON.stringify(current?.path ?? []);
  const parent = levels.at(-2);

  useEffect(() => {
    const nav = scrollRef.current;
    if (!nav) return;
    const revealCurrent = () => { nav.scrollLeft = nav.scrollWidth; };
    revealCurrent();
    const observer = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(revealCurrent);
    observer?.observe(nav);
    return () => observer?.disconnect();
  }, [pathKey]);

  return (
    <div className="hierarchy-navigation">
      <button type="button" className="hierarchy-up-button" aria-label="上一层级"
        disabled={!parent} title={parent ? `返回 ${hierarchyLabel(parent.title)}` : '当前已是最上层'}
        onClick={() => { if (parent) onNavigate([...parent.path]); }}>
        <ArrowUp size={14} aria-hidden="true" />
        上一层级
      </button>
      <nav ref={scrollRef} className="breadcrumbs" aria-label="模型层级">
        {levels.map((level, i) => {
          const label = hierarchyLabel(level.title);
          return (
            <span key={JSON.stringify(level.path)}>
              {i > 0 && <ChevronRight size={12} aria-hidden="true" />}
              <button type="button" aria-current={i === levels.length - 1 ? 'page' : undefined}
                title={label} onClick={() => onNavigate([...level.path])}>
                {i === 0 && <Network size={14} aria-hidden="true" />}
                <span className="breadcrumb-label">{label}</span>
              </button>
            </span>
          );
        })}
      </nav>
    </div>
  );
}
