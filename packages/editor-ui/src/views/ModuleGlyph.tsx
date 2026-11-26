import type { ReactNode } from 'react';
import type { SubmoduleIcon } from '@tensor-trail/model-core/core/types';

const submoduleGlyphs: Record<Exclude<SubmoduleIcon, 'auto'>, ReactNode> = {
  backbone: <path d="m2 6 7-4 7 4-7 4z M2 9.5l7 4 7-4 M2 13l7 4 7-4" />,
  attention: <><path d="M1 9s3-5 8-5 8 5 8 5-3 5-8 5-8-5-8-5Z" /><circle cx="9" cy="9" r="2.3" /></>,
  cross_attention: <path d="M2 4h3c3.5 0 4.5 10 8 10h3m-3-3 3 3-3 3 M2 14h3c1 0 2-1 3-3m2-5c1-1 2-2 3-2h3m-3-3 3 3-3 3" />,
  mlp: <><path d="m3 9 6-6 6 6-6 6z M3 9h12" />
    {[ [3,9], [9,3], [9,9], [9,15], [15,9] ].map(([x,y]) => <circle key={x+'-'+y} cx={x} cy={y} r="1.6" fill="currentColor" />)}</>,
  projection: <><path d="m2 6 6 3-6 3z M9 9h4m-2-2 2 2-2 2 M15 3v12" /></>,
  encoder: <path d="M2 3h4l5 6-5 6H2 M7 3h3l5 6-5 6H7 M15 6h2v6h-2" />,
  decoder: <path d="M2 6h3v6H2z M5 9h4m0 0 4-6h3m-7 6h7m-7 0 4 6h3" />,
  head: <><circle cx="9" cy="9" r="6.8" /><circle cx="9" cy="9" r="3.4" />
    <circle cx="9" cy="9" r="1" fill="currentColor" /></>,
  module: <path d="m9 1.5 7 4v7l-7 4-7-4v-7z M2 5.5l7 4 7-4 M9 9.5v7" />,
};

export function SubmoduleGlyph({ icon }: { icon: Exclude<SubmoduleIcon, 'auto'> }) {
  return <svg width="18" height="18" viewBox="0 0 18 18" data-submodule-icon={icon}
    aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.5"
    strokeLinecap="round" strokeLinejoin="round">{submoduleGlyphs[icon]}</svg>;
}

function DenseLayers({ layers }: { layers: number[] }) {
  const columns = layers.map((count, i) =>
    Array.from({ length: count }, (_, j) => ({
      x: 16 + (i * 64) / (layers.length - 1),
      y: 24 + (j - (count - 1) / 2) * 11,
    })),
  );
  return (
    <>
      <g opacity="0.35">
        {columns
          .slice(1)
          .flatMap((column, i) =>
            columns[i].flatMap((a, j) =>
              column.map((b, k) => (
                <path key={`${i}-${j}-${k}`} d={`M${a.x} ${a.y}L${b.x} ${b.y}`} />
              )),
            ),
          )}
      </g>
      {columns.flatMap((column, i) =>
        column.map((p, j) => (
          <circle key={`${i}-${j}`} cx={p.x} cy={p.y} r="3.5" className="glyph-fill" />
        )),
      )}
    </>
  );
}

function Grid({ x, y, columns, rows }: { x: number; y: number; columns: number; rows: number }) {
  return (
    <>
      {Array.from({ length: rows }, (_, row) =>
        Array.from({ length: columns }, (_, column) => (
          <rect
            key={`${row}-${column}`}
            x={x + column * 9}
            y={y + row * 9}
            width="7"
            height="7"
            rx="1"
          />
        )),
      )}
    </>
  );
}

function diagram(type: string): { label: string; content: ReactNode } {
  switch (type) {
    case 'LayoutRegion':
      return { label: '圆角布局分区', content: <><rect x="12" y="18" width="76" height="64" rx="12" /><path d="M12 36h76" />
        <rect x="24" y="47" width="20" height="20" rx="3" /><rect x="56" y="47" width="20" height="20" rx="3" /></> };
    case 'Linear':
      return { label: '全连接映射', content: <DenseLayers layers={[3, 3]} /> };
    case 'Linspace':
      return {
        label: '区间两端之间的等距采样刻度',
        content: <g strokeWidth="3">
          <path d="M12 24H84" />
          {[12, 30, 48, 66, 84].map(x => <path key={x} d={`M${x} 14v20`} />)}
        </g>,
      };
    case 'OnesLike':
    case 'ZerosLike': {
      const digit = type === 'OnesLike' ? '1' : '0';
      return {
        label: `同形张量网格填充 ${digit}`,
        content: <>
          <rect x="16" y="4" width="64" height="40" rx="3" />
          <path d="M48 4V44M16 24H80" />
          <g className="glyph-value-label" fill="currentColor" stroke="none" fontFamily="monospace" fontSize="17" fontWeight="bold" textAnchor="middle">
            {[18, 39].map(y => [32, 64].map(x => <text key={`${x}-${y}`} x={x} y={y}>{digit}</text>))}
          </g>
        </>,
      };
    }
    case 'Maximum':
      return {
        label: '两路逐元素取最大值',
        content: <>
          <path d="M9 9h9l14 15-14 15H9M66 24h21m-6-5 6 5-6 5" />
          <circle cx="49" cy="24" r="17" />
          <text className="glyph-value-label" x="49" y="29" fill="currentColor" stroke="none" textAnchor="middle" fontSize="14" fontWeight="bold">max</text>
        </>,
      };
    case 'Normalize':
      return {
        label: '逐通道减去均值并按标准差缩放',
        content: (
          <g className="normalize-glyph">
            <path d="M8 10H26M14 24H37M9 38H21M42 24H54M50 20L54 24L50 28" />
            <path d="M74 4V44" strokeDasharray="2 3" opacity="0.5" />
            <path d="M62 10H86M62 24H86M62 38H86" />
            <g fill="currentColor">
              <circle cx="17" cy="10" r="2.5" /><circle cx="25.5" cy="24" r="2.5" /><circle cx="15" cy="38" r="2.5" />
              {[10, 24, 38].map(y => <circle key={y} cx="74" cy={y} r="2.5" />)}
            </g>
          </g>
        ),
      };
    case 'GridSample': {
      const corners = [[32, 11], [62, 11], [32, 35], [62, 35]];
      return {
        label: '网格采样：四个邻域角点与格内采样位置',
        content: (
          <g className="grid-sample-glyph">
            <g opacity="0.45">
              <rect x="9" y="3" width="78" height="42" rx="2" />
              <path d="M32 3V45M62 3V45M9 11H87M9 35H87" />
            </g>
            <rect x="32" y="11" width="30" height="24" fill="currentColor" fillOpacity="0.13" />
            {corners.map(([x, y]) => (
              <line key={`line-${x}-${y}`} x1={x} y1={y} x2="52" y2="21" strokeWidth="1.25" />
            ))}
            {corners.map(([x, y]) => (
              <circle key={`${x}-${y}`} cx={x} cy={y} r="2.4" fill="currentColor" />
            ))}
            <circle className="grid-sample-point" cx="52" cy="21" r="3.6" />
          </g>
        ),
      };
    }
    case 'FFN':
    case 'MMCVFFNContract':
      return { label: '升维与降维的前馈网络', content: <DenseLayers layers={[2, 4, 2]} /> };
    case 'MultiheadAttention':
      return {
        label: 'Q、K、V 三路输入的多头注意力',
        content: (
          <>
            <g fill="currentColor" stroke="none" fontSize="10" fontFamily="Consolas, monospace">
              <text x="6" y="12">
                Q
              </text>
              <text x="6" y="27">
                K
              </text>
              <text x="6" y="42">
                V
              </text>
            </g>
            <path d="M19 9H31L42 19M19 24H42M19 39H31L42 29M64 24H87M83 20L87 24L83 28" />
            <rect x="42" y="13" width="22" height="22" rx="4" />
            <path d="M48 19L58 29M48 29L58 19M48 24H58" />
          </>
        ),
      };
    case 'SelfAttention':
      return {
        label: '各位置之间的注意力关联',
        content: (
          <>
            <DenseLayers layers={[3, 3]} />
            <path d="M30 6Q48 -1 66 6" />
          </>
        ),
      };
    case 'Transformer':
      return {
        label: '注意力与前馈模块的残差组合',
        content: (
          <>
            <path d="M7 29H24M44 29H53M73 29H89M16 29V8H80V29M76 25L80 29L84 25" />
            <rect x="24" y="17" width="20" height="24" rx="3" />
            <rect x="53" y="17" width="20" height="24" rx="3" />
            <path d="M29 23L39 35M29 35L39 23M29 29H39M59 23H67M59 29H67M59 35H67" />
          </>
        ),
      };
    case 'Conv1d':
      return {
        label: '卷积核沿序列滑动',
        content: (
          <>
            {Array.from({ length: 7 }, (_, i) => (
              <rect
                key={i}
                x={11 + i * 11}
                y="20"
                width="8"
                height="16"
                rx="1"
                opacity={i < 3 ? 1 : 0.35}
              />
            ))}
            <path d="M9 17V39H43V17ZM43 10H76M72 6L76 10L72 14" />
          </>
        ),
      };
    case 'Conv2d':
      return {
        label: '二维卷积网格与局部卷积核',
        content: (
          <>
            <g opacity="0.4">
              <Grid x={22} y={7} columns={5} rows={4} />
            </g>
            <rect x="20" y="5" width="28" height="28" rx="2" />
            <path d="M73 14V34M69 30L73 34L77 30" />
          </>
        ),
      };
    case 'LSTM':
    case 'GRU':
    case 'RNN':
      return {
        label: type === 'RNN' ? '带状态回路的循环单元' : `${type} 门控与状态回路`,
        content: (
          <>
            <path d="M9 31H29M67 31H87M74 31V8H21V20M17 16L21 20L25 16" />
            <rect x="29" y="20" width="38" height="22" rx="4" />
            {type === 'RNN' ? (
              <path d="M36 35C40 17 48 43 59 27" />
            ) : (
              <>
                <circle cx="40" cy="31" r="4" />
                <path d="M37 31H43M40 28V34M45 31H51" />
                {type === 'LSTM' ? (
                  <>
                    <circle cx="56" cy="31" r="4" />
                    <path d="M54 29L58 33M58 29L54 33" />
                  </>
                ) : (
                  <path d="M51 35L58 27" />
                )}
              </>
            )}
          </>
        ),
      };
    case 'Permute':
      return {
        label: '交换轴的顺序',
        content: (
          <>
            <path d="M13 12H26C44 12 52 36 70 36H83M13 36H26C44 36 52 12 70 12H83M13 24H83M79 8L83 12L79 16M79 20L83 24L79 28M79 32L83 36L79 40" />
          </>
        ),
      };
    case 'Reshape':
      return {
        label: '重新排列张量形状',
        content: (
          <>
            <Grid x={9} y={10} columns={2} rows={3} />
            <path d="M35 24H53M49 20L53 24L49 28" />
            <Grid x={62} y={15} columns={3} rows={2} />
          </>
        ),
      };
    case 'BooleanIndexAxis':
      return {
        label: '按布尔条件保留整项并保持顺序（筛选示意）',
        content: (
          <>
            <rect x="6" y="4" width="24" height="7" rx="1.5" className="glyph-fill" />
            <rect x="6" y="16" width="24" height="7" rx="1.5" opacity="0.25" />
            <rect x="6" y="28" width="24" height="7" rx="1.5" className="glyph-fill" />
            <rect x="6" y="40" width="24" height="5" rx="1.5" opacity="0.25" />
            <path d="M37 24H56M51 19L56 24L51 29" />
            <rect x="64" y="13" width="26" height="8" rx="1.5" className="glyph-fill" />
            <rect x="64" y="28" width="26" height="8" rx="1.5" className="glyph-fill" />
          </>
        ),
      };
    case 'Detach':
      return {
        label: '数据继续传递，梯度在此截断',
        content: (
          <>
            <path d="M8 12H88M81 5L88 12L81 19" />
            <path d="M88 36H56" strokeDasharray="5 4" />
            <path d="M63 29L56 36L63 43M48 27V45" />
            <path d="M34 36H8" strokeDasharray="5 4" opacity="0.25" />
          </>
        ),
      };
    case 'DictionaryPack':
      return {
        label: '按字段名组装字典，不堆叠或融合张量',
        content: (
          <>
            <path d="M6 10H22M6 24H22M6 38H22" />
            <path d="M28 10H40M36 6L40 10L36 14M28 24H40M36 20L40 24L36 28M28 38H40M36 34L40 38L36 42" opacity="0.65" />
            <path d="M53 4H50Q47 4 47 8V17Q47 24 43 24Q47 24 47 31V40Q47 44 50 44H53M81 4H84Q87 4 87 8V17Q87 24 91 24Q87 24 87 31V40Q87 44 84 44H81" />
            <path d="M57 10H62M70 10H79M57 24H62M70 24H79M57 38H62M70 38H79" />
            <g fill="currentColor" stroke="none">
              <circle cx="66" cy="10" r="1" />
              <circle cx="66" cy="24" r="1" />
              <circle cx="66" cy="38" r="1" />
            </g>
          </>
        ),
      };
    case 'DictionaryGet':
      return {
        label: '按键读取字典字段',
        content: (
          <>
            <path d="M14 5H11Q8 5 8 9V17Q8 24 4 24Q8 24 8 31V39Q8 43 11 43H14M48 5H51Q54 5 54 9V17Q54 24 58 24Q54 24 54 31V39Q54 43 51 43H48" opacity="0.5" />
            <g opacity="0.35">
              <path d="M20 11H27M34 11H44M20 37H27M34 37H44" />
              <circle cx="30" cy="11" r="0.7" />
              <circle cx="30" cy="37" r="0.7" />
            </g>
            <path d="M20 24H27" />
            <circle cx="30" cy="24" r="0.7" />
            <rect x="34" y="20" width="10" height="8" rx="1.5" className="glyph-fill" />
            <path d="M60 24H72M67 19L72 24L67 29" />
            <rect x="78" y="18" width="12" height="12" rx="2" />
          </>
        ),
      };
    case 'TensorSequenceGet':
      return {
        label: '从 list/tuple 中按索引取一个 Tensor',
        content: (
          <>
            <path d="M13 5H8V43H13M49 5H54V43H49" opacity="0.5" />
            <g opacity="0.35">
              <rect x="18" y="8" width="26" height="6" rx="1.5" />
              <rect x="18" y="34" width="26" height="6" rx="1.5" />
            </g>
            <rect x="18" y="21" width="26" height="6" rx="1.5" className="glyph-fill" />
            <path d="M48 24H72M67 19L72 24L67 29" />
            <rect x="78" y="18" width="12" height="12" rx="2" />
          </>
        ),
      };
    case 'Slice':
      return {
        label: '保留指定轴的一段区间',
        content: (
          <>
            <g opacity="0.3">
              <Grid x={17} y={10} columns={6} rows={3} />
            </g>
            <rect x="15" y="8" width="20" height="29" rx="2" />
            <path d="M41 5V43" strokeDasharray="3 3" />
          </>
        ),
      };
    case 'Flatten':
      return {
        label: '将多个轴展平成一行',
        content: (
          <>
            <Grid x={11} y={10} columns={2} rows={3} />
            <path d="M36 24H49M45 20L49 24L45 28" />
            <Grid x={54} y={20} columns={4} rows={1} />
          </>
        ),
      };
    case 'Concat':
      return {
        label: '将两路张量拼接',
        content: (
          <>
            <path d="M10 13H31L46 24M10 35H31L46 24M46 24H55" />
            <rect x="55" y="16" width="14" height="16" rx="2" />
            <rect x="69" y="16" width="18" height="16" rx="2" />
          </>
        ),
      };
    case 'Add':
    case 'Mul':
      return {
        label: type === 'Add' ? '两路逐元素相加' : '两路逐元素相乘',
        content: (
          <>
            <path d="M10 11H25L35 21M10 37H25L35 27M61 24H87" />
            <circle cx="48" cy="24" r="13" />
            {type === 'Add' ? (
              <path d="M41 24H55M48 17V31" />
            ) : (
              <path d="M43 19L53 29M53 19L43 29" />
            )}
          </>
        ),
      };
    case 'MatMul':
      return {
        label: '两矩阵相乘',
        content: (
          <>
            <Grid x={11} y={15} columns={2} rows={2} />
            <path d="M38 19L48 29M48 19L38 29" />
            <Grid x={58} y={10} columns={3} rows={3} />
          </>
        ),
      };
    case 'LayerNorm':
    case 'RMSNorm':
      return {
        label: type === 'LayerNorm' ? '将特征居中并归一化' : '按均方根归一化特征',
        content: (
          <>
            <path d="M10 24H38M15 16V32M24 8V40M33 20V28M44 24H54M50 20L54 24L50 28M60 24H88M65 13V35M74 13V35M83 13V35" />
            {type === 'RMSNorm' && <path d="M62 7H86" />}
          </>
        ),
      };
    case 'GELU':
    case 'ReLU':
    case 'SiLU':
      return {
        label: `${type} 激活曲线`,
        content: (
          <>
            <path d="M10 36H88M43 42V5" opacity="0.3" />
            {type === 'ReLU' ? (
              <path d="M12 35H45L80 7" />
            ) : (
              <path
                d={
                  type === 'GELU'
                    ? 'M12 35C28 35 35 41 43 33S65 13 81 6'
                    : 'M12 35C25 36 32 42 43 34S65 14 81 6'
                }
              />
            )}
          </>
        ),
      };
    case 'Tanh':
      return {
        label: 'Tanh 双曲正切曲线，值域 -1 到 1',
        content: (
          <>
            <path d="M8 24H88M48 44V4" opacity="0.3" />
            <path d="M10 38H22C36 38 40 34 48 24S60 10 74 10H86" />
          </>
        ),
      };
    case 'Identity':
      return { label: '直接传递输入', content: <path d="M12 24H84M76 16L84 24L76 32" /> };
    case 'SplitHeads':
    case 'MergeHeads':
      return {
        label: type === 'SplitHeads' ? '分成多个注意力头' : '合并多个注意力头',
        content: (
          <g transform={type === 'MergeHeads' ? 'translate(96 0) scale(-1 1)' : undefined}>
            <rect x="12" y="16" width="18" height="16" rx="2" />
            <path d="M30 24H43M43 11V37M43 11H65M43 24H65M43 37H65" />
            {[7, 20, 33].map((y) => (
              <rect key={y} x="65" y={y} width="18" height="8" rx="2" />
            ))}
          </g>
        ),
      };
    case 'Scale':
      return {
        label: '按系数缩放',
        content: (
          <>
            <path d="M19 24H77" />
            <circle cx="48" cy="11" r="3" />
            <circle cx="48" cy="37" r="3" />
          </>
        ),
      };
    case 'Softmax':
      return {
        label: '将分数归一化为权重',
        content: (
          <>
            <path d="M13 39H83" opacity="0.35" />
            {[9, 18, 30, 13].map((h, i) => (
              <rect key={i} x={19 + i * 17} y={39 - h} width="9" height={h} rx="2" />
            ))}
            <path d="M14 8Q48 -1 82 8" opacity="0.6" />
          </>
        ),
      };
    case 'CausalMask':
      return {
        label: '只保留当前位置及过去位置',
        content: (
          <>
            {[0, 1, 2, 3].map((row) =>
              [0, 1, 2, 3].map((column) => (
                <rect
                  key={`${row}-${column}`}
                  x={30 + column * 9}
                  y={7 + row * 9}
                  width="7"
                  height="7"
                  rx="1"
                  opacity={column <= row ? 1 : 0.15}
                  className={column <= row ? 'glyph-fill' : undefined}
                />
              )),
            )}
          </>
        ),
      };
    default:
      return {
        label: '组合子网络',
        content: (
          <>
            <path d="M29 24H39M57 24H67" />
            <rect x="11" y="15" width="18" height="18" rx="3" />
            <rect x="39" y="15" width="18" height="18" rx="3" />
            <rect x="67" y="15" width="18" height="18" rx="3" />
          </>
        ),
      };
  }
}

export function ModuleGlyph({ type }: { type: string }) {
  const { label, content } = diagram(type);
  return (
    <svg className="module-glyph" viewBox="0 0 96 48" role="img" aria-label={label}>
      <title>{label}</title>
      <g
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        {content}
      </g>
    </svg>
  );
}
