import type { ReactNode } from 'react';

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
    case 'Linear':
      return { label: '全连接映射', content: <DenseLayers layers={[3, 3]} /> };
    case 'FFN':
      return { label: '升维与降维的前馈网络', content: <DenseLayers layers={[2, 4, 2]} /> };
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
