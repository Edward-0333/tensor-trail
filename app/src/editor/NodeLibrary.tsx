import { useEffect, useState, type ReactNode, type RefObject } from 'react';
import { Boxes, Calculator, Search } from 'lucide-react';
import type { ModuleDefinition } from '../core/types';
import { registry } from '../modules';
import { tone } from '../views/ModuleNode';
import { startModuleDrag } from './moduleDrag';
import { useEditor } from './store';
import { calculationCategories, libraryCategory } from '../modules/libraryCategories';

const moduleCategories = [
  '输入输出',
  '网络结构',
  '基础层',
  '卷积',
  '循环网络',
  'Transformer',
  '归一化',
];

function Foldout({
  className,
  title,
  query,
  children,
  defaultOpen = false,
  id,
}: {
  className: string;
  title: ReactNode;
  query: string;
  children: ReactNode;
  defaultOpen?: boolean;
  id?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  useEffect(() => {
    if (query) setOpen(true);
  }, [query]);
  return (
    <details
      className={className}
      data-library={id}
      open={open}
      onToggle={(event) => {
        if (event.target === event.currentTarget) setOpen(event.currentTarget.open);
      }}
    >
      <summary>{title}</summary>
      {children}
    </details>
  );
}

export function NodeLibrary({ searchRef }: { searchRef: RefObject<HTMLInputElement | null> }) {
  const state = useEditor();
  const [search, setSearch] = useState('');
  const query = search.trim().toLowerCase();
  // Boundary ports belong to the current submodule, never to the root forward().
  const definitions = registry.list(state.path.length > 0);
  const matches = (definition: ModuleDefinition) =>
    `${definition.label} ${definition.type} ${definition.description} ${libraryCategory(definition)}`
      .toLowerCase()
      .includes(query);
  const variants = state.project.variants.filter((variant) =>
    variant.name.toLowerCase().includes(query),
  );

  function categories(names: string[]) {
    return names.map((category) => {
      const items = definitions.filter(
        (definition) => libraryCategory(definition) === category && matches(definition),
      );
      if (!items.length) return null;
      return (
        <Foldout
          key={category}
          className={`module-group tone-${tone(category)}`}
          query={query}
          defaultOpen={['输入输出', '网络结构', '索引与切片', '形状查询'].includes(category)}
          title={
            <>
              <span className="category-dot" />
              <span>{category}</span>
              <small>{items.length}</small>
            </>
          }
        >
          <div className="module-items" role="list">
            {items.map((definition) => (
              <div
                key={definition.type}
                className={`library-module tone-${tone(definition.category)}`}
                role="listitem"
                tabIndex={0}
                aria-label={`${definition.label}，拖入画布添加`}
                title={definition.label}
                draggable={state.hasProject}
                aria-disabled={!state.hasProject}
                onDragStart={(event) => startModuleDrag(event, { type: definition.type })}
              >
                <strong>{definition.label}</strong>
              </div>
            ))}
          </div>
        </Foldout>
      );
    });
  }

  return (
    <aside className="library" aria-label="模块库与计算库">
      <label className="search-field library-search">
        <Search size={15} />
        <input
          ref={searchRef}
          aria-label="搜索模块或计算"
          placeholder="搜索模块或计算…"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <kbd>/</kbd>
      </label>
      <p className="library-hint">
        {state.hasProject ? '拖动模块或计算到画布添加' : '先打开项目或新建项目'}
      </p>
      <div className="library-scroll">
        {[
          { id: 'modules', label: '模块库', icon: Boxes, categories: moduleCategories },
          {
            id: 'calculations',
            label: '计算库',
            icon: Calculator,
            categories: calculationCategories,
          },
        ].map((library) => {
          const entries = definitions.filter((definition) =>
            library.categories.includes(libraryCategory(definition)),
          );
          const count = entries.filter(matches).length;
          return (
            <Foldout
              key={library.id}
              id={library.id}
              className="library-section"
              query={query}
              defaultOpen
              title={
                <>
                  <library.icon size={17} />
                  <h2>{library.label}</h2>
                  <span className="count-badge">
                    {query ? `${count}/${entries.length}` : entries.length}
                  </span>
                </>
              }
            >
              <div className="library-section-content">
                {categories(library.categories)}
                {!count && !(library.id === 'modules' && variants.length) && (
                  <p className="search-empty">
                    没有匹配的{library.id === 'modules' ? '模块' : '计算'}。
                  </p>
                )}
                {library.id === 'modules' && (!query || variants.length > 0) && (
                  <Foldout
                    className="module-group library-variants"
                    query={query}
                    title={
                      <>
                        <span>我的变体</span>
                        <small>{variants.length}</small>
                      </>
                    }
                  >
                    {variants.length ? (
                      <div className="module-items" role="list">
                        {variants.map((variant) => (
                          <div
                            className="variant-card"
                            key={variant.id}
                            role="listitem"
                            tabIndex={0}
                            aria-label={`${variant.name}，拖入画布添加`}
                            title={variant.name}
                            draggable={state.hasProject}
                            onDragStart={(event) =>
                              startModuleDrag(event, { type: 'Block', variantId: variant.id })
                            }
                          >
                            <span>{variant.name}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="variant-placeholder">
                        将自定义组合保存成变体，在这里重复使用。
                      </div>
                    )}
                  </Foldout>
                )}
              </div>
            </Foldout>
          );
        })}
      </div>
      <div className="library-footer">
        <span className="mode-dot" />
        <div>
          <strong>形状模式</strong>
          <span>无需训练 · 无真实数值</span>
        </div>
      </div>
    </aside>
  );
}
