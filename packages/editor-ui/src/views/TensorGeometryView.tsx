import { useId, useState, type CSSProperties } from 'react';
import {
  projectTensorSlices,
  projectOuterVolumes,
  type GeometryBox,
  type TensorGeometryDiagram,
  type TensorJoin,
} from '@tensor-trail/model-core/core/tensorGeometry';

import { tensorFaceColor as color } from './tensorFaceColor';

type Tones = (GeometryBox['tones'][number] | 'custom')[];
type Face = 'top' | 'front' | 'side';
const faces: Face[] = ['top', 'front', 'side'];
const gradientKey = (tones: Tones) => tones.join('-').replaceAll('#', 'hex');
const paint = (id: string, tones: Tones, face: Face) =>
  tones.length === 1 ? color(tones[0], face) : `url(#${id}-${gradientKey(tones)}-${face})`;
const slope = 0.6;
const extent = (box: GeometryBox) => box.depthPx + (box.depth.sliceCount - 1) * box.sliceGapPx;
const fullShape = (box: GeometryBox) =>
  [
    ...box.leadingChips.map((c) => `${c.label}=${c.size}`),
    ...[...(box.outer ? [box.outer] : []), box.depth, box.height, box.width]
      .filter((a) => a.dimIndex !== undefined)
      .map((a) => `${a.axis}=${a.size}`),
  ].join(' · ') || '标量';

function Gradients({ id, groups }: { id: string; groups: Tones[] }) {
  const blends = new Map(
    groups.filter((tones) => tones.length > 1).map((tones) => [gradientKey(tones), tones]),
  );
  return (
    <defs>
      {[...blends].flatMap(([key, tones]) =>
        faces.map((face) => (
          <linearGradient key={key + face} id={`${id}-${key}-${face}`} x2="100%" y2="60%">
            {tones.map((tone, i) => (
              <stop key={tone} offset={i / (tones.length - 1)} stopColor={color(tone, face)} />
            ))}
          </linearGradient>
        )),
      )}
    </defs>
  );
}

interface PrismProps {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
  d: number;
  tones: Tones;
}
function Prism({ id, x, y, w, h, z, d, tones }: PrismProps) {
  return (
    <g stroke={color(tones[0], 'stroke')} strokeLinejoin="round" className="tv-prism">
      <path data-face="front" fill={paint(id, tones, 'front')} d={`M${x},${y}h${w}v${h}h${-w}Z`} />
      <path
        data-face="top"
        fill={paint(id, tones, 'top')}
        d={`M${x},${y}l${z},${-d}h${w}l${-z},${d}Z`}
      />
      <path
        data-face="side"
        fill={paint(id, tones, 'side')}
        d={`M${x + w},${y}l${z},${-d}v${h}l${-z},${d}Z`}
      />
    </g>
  );
}

// Partition only the visible faces. Painting whole cuboids here would expose hidden interior walls.
function JoinedSlab({
  prism: p,
  join,
  custom,
}: {
  prism: PrismProps;
  join: TensorJoin;
  custom: boolean;
}) {
  const horizontal = join.direction === 'width';
  const vertical = join.direction === 'height';
  const receding = join.direction === 'depth';
  const parts = join.segments.filter((s) => s.end > s.start);
  const { x, y, w, h, z, d, id } = p;
  return (
    <g data-segments={join.direction}>
      {parts.map((s, i) => {
        const fraction = s.end - s.start;
        const a = x + (horizontal ? w : receding ? z : 0) * s.start;
        const b = y + (vertical ? h : receding ? -d : 0) * s.start;
        const width = horizontal ? w * fraction : w,
          height = vertical ? h * fraction : h;
        const dz = receding ? z * fraction : z,
          dy = receding ? d * fraction : d;
        const tones: Tones = custom ? ['custom'] : s.tones;
        return (
          <g key={s.port} data-segment={s.port}>
            {(!receding || i === 0) && (
              <path
                data-face="front"
                fill={paint(id, tones, 'front')}
                d={`M${a},${b}h${width}v${height}h${-width}Z`}
              />
            )}
            {(!vertical || i === 0) && (
              <path
                data-face="top"
                fill={paint(id, tones, 'top')}
                d={`M${a},${b}l${dz},${-dy}h${width}l${-dz},${dy}Z`}
              />
            )}
            {(!horizontal || i === parts.length - 1) && (
              <path
                data-face="side"
                fill={paint(id, tones, 'side')}
                d={`M${a + width},${b}l${dz},${-dy}v${height}l${-dz},${dy}Z`}
              />
            )}
          </g>
        );
      })}
      <g fill="none" stroke={color(p.tones[0], 'stroke')} strokeLinejoin="round">
        <path
          d={`M${x},${y}l${z},${-d}h${w}v${h}l${-z},${d}h${-w}ZM${x},${y}h${w}l${z},${-d}M${x + w},${y}v${h}`}
        />
        {parts.slice(1).map((s) => (
          <path
            key={s.port}
            className="tv-seam"
            strokeDasharray="2.5 2"
            d={
              horizontal
                ? `M${x + w * s.start},${y + h}v${-h}l${z},${-d}`
                : vertical
                  ? `M${x},${y + h * s.start}h${w}l${z},${-d}`
                  : `M${x + z * s.start},${y - d * s.start}h${w}v${h}`
            }
          />
        ))}
      </g>
    </g>
  );
}

function Slabs({
  box,
  join,
  id,
  depthX,
  renderW,
  custom,
}: {
  box: GeometryBox;
  join?: TensorJoin;
  id: string;
  depthX: number;
  renderW: number;
  custom: boolean;
}) {
  const axes = [
    ...box.leadingChips,
    ...(box.outer ? [box.outer] : []),
    box.depth,
    box.height,
    box.width,
  ];
  const symbolic = axes.some((a) => typeof a.size === 'string');
  const empty = axes.some((a) => a.size === 0);
  const partitionFaces =
    join?.direction === 'width' ||
    join?.direction === 'height' ||
    (box.depth.solid && join?.direction === 'depth');
  return (
    <g strokeDasharray={symbolic || empty ? '3 2' : undefined} opacity={empty ? 0.35 : 1}>
      {projectTensorSlices(box, join).map(({ index, segment, tones }) => {
        const shift = index * box.sliceGapPx;
        const prism: PrismProps = {
          id,
          x: 0,
          y: 0,
          w: renderW,
          h: box.hPx,
          z: box.depthPx * depthX,
          d: box.depthPx * slope,
          tones: custom ? ['custom'] : tones,
        };
        return (
          <g
            key={index}
            transform={`translate(${shift * depthX} ${-shift * slope})`}
            data-slice={index}
            data-segment={partitionFaces ? undefined : segment}
          >
            {join && partitionFaces ? (
              <JoinedSlab prism={prism} join={join} custom={custom} />
            ) : (
              <Prism {...prism} />
            )}
          </g>
        );
      })}
    </g>
  );
}

function BoxDrawing({
  box,
  join,
  id,
  baseline,
  width,
  custom,
}: {
  box: GeometryBox;
  join?: TensorJoin;
  id: string;
  baseline: number;
  width: number;
  custom: boolean;
}) {
  const volumes = projectOuterVolumes(box, join);
  const count = volumes.length;
  const depth = extent(box);
  const avail = width - 26;
  const maxGap = box.outer?.elided ? 26 : 14;
  const minGap = box.outer?.elided ? 18 : 8;
  const gap =
    count === 1
      ? 0
      : Math.max(minGap, Math.min(maxGap, (avail - count * (box.wPx + depth)) / (count - 1)));
  const unitBudget = (avail - (count - 1) * gap) / count;
  // Preserve the front face at graph scale while there is room. When both dimensions
  // exceed the budget, reserve some extrusion so even a narrow card reads as a volume.
  const z = Math.min(
    depth,
    Math.max(unitBudget - box.wPx, Math.min(depth * 0.45, unitBudget * 0.3)),
  );
  const w = Math.min(box.wPx, unitBudget - z),
    h = box.hPx;
  const unitW = w + z,
    rowW = count * unitW + (count - 1) * gap;
  const x = (width - rowW + 16) / 2,
    y = baseline - h;
  const widthCenter = x + w / 2,
    depthRight = x + unitW;
  const axisText = (axis: GeometryBox['width'], fallback: string, room: number) => {
    const name = /^axis\d+$/.test(axis.axis) ? fallback : axis.axis;
    const size = String(axis.size);
    return textWidth(name + ' ' + size) <= room ? name + ' ' + size : fitText(size, room);
  };
  const outer = box.outer;
  const outerLabel =
    outer && (/^axis\d+$/.test(outer.axis) ? '×' + outer.size : outer.axis + ' ' + outer.size);
  const depthLabel = axisText(
    { ...box.depth, size: box.depth.elided ? '×' + box.depth.size : box.depth.size },
    'Z',
    width - 8,
  );
  const depthLabelEnd = Math.min(width - 4, Math.max(depthRight, textWidth(depthLabel) + 4));
  return (
    <>
      {volumes.map((v) => (
        <g
          key={v.index}
          data-volume={outer ? v.index : undefined}
          data-segment={v.segment}
          transform={`translate(${x + v.index * (unitW + gap)} ${y})`}
        >
          <Slabs
            box={{ ...box, tones: v.tones }}
            join={join}
            id={id}
            renderW={w}
            depthX={z / depth}
            custom={custom}
          />
        </g>
      ))}
      {outer?.elided && (
        <text
          className="tv-ellipsis"
          x={x + unitW + gap / 2}
          y={y + (h - depth * slope) / 2}
          textAnchor="middle"
          dominantBaseline="central"
        >
          ···
        </text>
      )}
      <g className="tv-guides" fill="none">
        {box.ndim > 0 && <path d={`M${x},${baseline + 3}v3h${w}v-3`} />}
        {box.ndim > 1 && <path d={`M${x - 3},${y}h-3v${h}h3`} />}
        {box.ndim > 2 && <path d={`M${x + w + 2},${y - 3}l${z},${-depth * slope}`} />}
        {outer && <path data-guide="outer" d={`M${x},${baseline + 26}v4h${rowW}v-4`} />}
      </g>
      <g className="tv-dim" textAnchor="middle">
        <text x={widthCenter} y={baseline + 21}>
          {box.ndim
            ? axisText(box.width, 'W', 2 * Math.min(widthCenter, width - widthCenter) - 8)
            : '标量'}
        </text>
        {box.ndim > 1 && (
          <text transform={`translate(${x - 10} ${y + h / 2}) rotate(-90)`}>
            {axisText(box.height, 'H', h + 30)}
          </text>
        )}
        {box.ndim > 2 && (
          <text x={depthLabelEnd} y={y - depth * slope - 8} textAnchor="end">
            {depthLabel}
          </text>
        )}
        {outerLabel && (
          <text x={x + rowW / 2} y={baseline + 44}>
            {fitText(outerLabel, width - 18)}
          </text>
        )}
      </g>
    </>
  );
}

interface Annotation {
  text: string;
  tones?: Tones;
}
// Reserve a full glyph for wide letters and CJK in symbolic sizes and axis names.
const textWidth = (s: string) =>
  Array.from(s).reduce((width, c) => {
    const narrow = /[0-9a-z .|]/.test(c) && c !== 'm' && c !== 'w';
    return width + (narrow ? 6.5 : 10.5);
  }, 0);
function fitText(s: string, width: number) {
  if (textWidth(s) <= width) return s;
  let prefix = '';
  for (const c of s) {
    if (textWidth(prefix + c + '…') > width) break;
    prefix += c;
  }
  return prefix + '…';
}

function layoutAnnotations(diagram: TensorGeometryDiagram, width: number) {
  const groups: Annotation[][] = [
    diagram.output.leadingChips.map((chip) => ({ text: chip.label + '=' + chip.size })),
    diagram.join?.segments.map((s) => ({ text: String(s.size), tones: s.tones })) ?? [],
  ];
  let y = 0;
  const items = [];
  for (const group of groups) {
    if (!group.length) continue;
    let x = 4;
    for (const item of group) {
      const pad = item.tones ? 19 : 10;
      const label = fitText(item.text, width - pad - 8);
      const w = textWidth(label) + pad;
      let divider = !!item.tones && x > 4;
      if (x + w + (divider ? 8 : 0) > width - 3) {
        divider = false;
        x = 4;
        y += 17;
      }
      if (divider) x += 8;
      items.push({ ...item, label, x, y, w, divider });
      x += w + 3;
    }
    y += 17;
  }
  return { items, height: y };
}

export function TensorGeometryView({
  diagram,
  variant = 'node',
}: {
  diagram: TensorGeometryDiagram;
  variant?: 'node' | 'integrated' | 'inspector';
}) {
  const id = 'tv' + useId().replace(/[^a-z0-9_-]/gi, '');
  const [selected, setSelected] = useState(false);
  const inspect = variant === 'inspector';
  const integrated = variant === 'integrated';
  const box = diagram.output;
  // Match the card content width (6 px padding and a 1 px border on each side).
  const width = diagram.width - 14;
  const baseline = (integrated ? 22 : 42) + box.hPx + extent(box) * slope;
  const footer = baseline + 30 + (box.outer ? 22 : 0);
  const labels = layoutAnnotations(diagram, width);
  const height = footer + labels.height;
  const groups = [box.tones, ...(diagram.join?.segments.map((s) => s.tones) ?? [])];
  const description = fullShape(box);
  const dtype = fitText(box.dtype, width / 2);
  const role = (diagram.observeInput ? '输入 ' : '输出 ') + box.port;
  const select = () => setSelected((previous) => !previous);
  return (
    <figure
      className="tensor-geometry-card"
      data-variant={variant}
      data-mode={diagram.mode}
      style={
        {
          width: inspect ? undefined : diagram.width,
          '--tv-custom-base': diagram.customColor,
        } as CSSProperties
      }
    >
      <svg
        className="tensor-geometry-svg"
        style={{ maxWidth: width }}
        xmlns="http://www.w3.org/2000/svg"
        viewBox={`0 0 ${width} ${height}`}
        role={inspect ? 'group' : 'img'}
        aria-labelledby={id + '-title'}
      >
        <title
          id={id + '-title'}
        >{`${diagram.badge} · ${description} · ${diagram.output.dtype}`}</title>
        <Gradients id={id} groups={groups} />
        {!integrated && (
          <g className="tv-heading">
            <text x="5" y="12">
              {fitText(diagram.badge, width - textWidth(dtype) - 20)}
            </text>
            <text x={width - 5} y="12" textAnchor="end" className="tv-dtype">
              {dtype}
            </text>
          </g>
        )}
        <g
          data-box="output"
          className={inspect ? 'tv-inspect-box' : undefined}
          {...(inspect
            ? {
                role: 'button',
                tabIndex: 0,
                'aria-pressed': selected,
                'aria-label': role + ' · ' + description,
                onClick: select,
                onKeyDown: (event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    select();
                  }
                },
              }
            : {})}
        >
          {inspect && (
            <rect className="tv-hit" x="1" y="17" width={width - 2} height={footer - 18} rx="5" />
          )}
          <BoxDrawing
            box={box}
            id={id}
            baseline={baseline}
            width={width}
            join={diagram.join}
            custom={!!diagram.customColor}
          />
        </g>
        <g className="tv-annotations" data-direction={diagram.join?.direction}>
          {labels.items.map((item, i) => (
            <g key={i} transform={`translate(${item.x} ${footer + item.y})`}>
              <title>{item.text}</title>
              {item.divider && (
                <text x="-7" y="10.5">
                  |
                </text>
              )}
              <rect className="tv-chip" width={item.w} height="14" rx="4" />
              {item.tones && (
                <rect
                  x="4"
                  y="4"
                  width="7"
                  height="7"
                  rx="2"
                  fill={paint(id, item.tones, 'front')}
                />
              )}
              <text x={item.tones ? 14 : 5} y="10.5">
                {item.label}
              </text>
            </g>
          ))}
        </g>
      </svg>
      {inspect && selected && (
        <div className="tv-detail" aria-live="polite">
          {description} · {box.dtype}
        </div>
      )}
      {diagram.caption && (
        <figcaption className="tensor-geometry-caption">{diagram.caption}</figcaption>
      )}
    </figure>
  );
}
