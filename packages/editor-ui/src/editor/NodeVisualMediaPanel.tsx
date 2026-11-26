import { useRef, useState } from 'react';
import { ArrowUp, ArrowDown, Trash2, ImagePlus } from 'lucide-react';
import {
  uid,
  type ModelNode,
  type NodeVisualImage,
  type NodeVisualMedia,
} from '@tensor-trail/model-core/core/types';
import { graphAt } from '@tensor-trail/model-core/core/graph';
import { maxImageSourceLength, maxMediaItems } from '@tensor-trail/model-core/core/visualMedia';
import { defaultTileWidth } from '../views/NodeVisualMedia';
import { useEditor } from './store';

const formats = ['image/png', 'image/jpeg', 'image/webp'];

async function originalDataUrl(file: File): Promise<string> {
  if (file.size > (maxImageSourceLength * 3) / 4) throw Error('原图过大，请选择较小的图片。');
  if (typeof FileReader !== 'undefined')
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(Error('无法读取图片。'));
      reader.readAsDataURL(file);
    });
  const bytes = new Uint8Array(await file.arrayBuffer());
  return `data:${file.type};base64,${btoa(Array.from(bytes, (byte) => String.fromCharCode(byte)).join(''))}`;
}

async function readImage(file: File) {
  if (!formats.includes(file.type)) throw Error('请选择 PNG、JPEG 或 WebP 图片。');
  const canvas = typeof document !== 'undefined' ? document.createElement('canvas') : undefined;
  if (
    !canvas ||
    typeof Image === 'undefined' ||
    typeof canvas.toDataURL !== 'function' ||
    typeof URL.createObjectURL !== 'function'
  )
    return originalDataUrl(file);
  let context;
  try {
    context = canvas.getContext('2d');
  } catch {
    return originalDataUrl(file);
  }
  if (!context) return originalDataUrl(file);
  const image = new Image();
  if (typeof image.decode !== 'function') return originalDataUrl(file);
  const objectUrl = URL.createObjectURL(file);
  try {
    image.src = objectUrl;
    await image.decode();
    let scale = Math.min(1, 480 / Math.max(image.naturalWidth, image.naturalHeight));
    for (;;) {
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      const src = canvas.toDataURL(file.type, 0.85);
      if (src.length <= maxImageSourceLength) return src;
      if (canvas.width === 1 && canvas.height === 1) throw Error('压缩后的插图超出大小限制。');
      scale *= 0.75;
    }
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

/** Decode the batch before committing. Edits made during decoding are read from the live node. */
export async function uploadNodeImages(id: string, files: File[], replaceId?: string) {
  if (!files.length) return;
  const { path, documentId, canvasVersion, project } = useEditor.getState();
  const find = (current: typeof project) =>
    graphAt(current.graph, path).nodes.find((n) => n.id === id);
  if (!replaceId && (find(project)?.visualMedia?.items.length ?? 0) + files.length > maxMediaItems)
    throw Error('每个节点最多 12 张插图。');
  const sources: string[] = [];
  for (const file of files) sources.push(await readImage(file));
  const state = useEditor.getState();
  if (
    documentId !== state.documentId ||
    path !== state.path ||
    canvasVersion !== state.canvasVersion
  )
    return;
  const node = find(state.project);
  if (!node) return;
  const media = node.visualMedia ?? { items: [] };
  if (replaceId && !media.items.some((item) => item.id === replaceId)) return;
  const items = replaceId
    ? media.items.map((item) => (item.id === replaceId ? { ...item, src: sources[0] } : item))
    : [...media.items, ...sources.map((src) => ({ id: uid(), src }))];
  if (!state.setNodeVisualMedia(id, { ...media, items })) throw Error(useEditor.getState().notice);
}

function UploadButton({
  replaceId,
  disabled,
  upload,
}: {
  replaceId?: string;
  disabled: boolean;
  upload: (files: File[], replaceId?: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const label = replaceId ? '单张换图' : '上传图片';
  return (
    <>
      <input
        hidden
        ref={input}
        type="file"
        accept={formats.join(',')}
        multiple={!replaceId}
        aria-label={label}
        onChange={(event) => {
          const files = Array.from(event.currentTarget.files ?? []);
          event.currentTarget.value = '';
          upload(files, replaceId);
        }}
      />
      <button className="soft-button" disabled={disabled} onClick={() => input.current?.click()}>
        <ImagePlus size={14} />
        {label}
      </button>
    </>
  );
}

export function NodeVisualMediaPanel({ node }: { node: ModelNode }) {
  const [busy, setBusy] = useState(false);
  const media = node.visualMedia ?? { items: [] };
  const save = (next: NodeVisualMedia) =>
    useEditor.getState().setNodeVisualMedia(node.id, next.items.length ? next : null);
  const update = (id: string, change: Partial<NodeVisualImage>) =>
    save({
      ...media,
      items: media.items.map((item) => (id === item.id ? { ...item, ...change } : item)),
    });
  const move = (index: number, direction: number) => {
    const items = [...media.items];
    [items[index], items[index + direction]] = [items[index + direction], items[index]];
    save({ ...media, items });
  };
  const upload = async (files: File[], replaceId?: string) => {
    setBusy(true);
    try {
      await uploadNodeImages(node.id, files, replaceId);
    } catch (error) {
      useEditor.getState().toast(`上传失败：${(error as Error).message}`);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="media-settings" aria-label="节点插图" aria-busy={busy}>
      <div className="media-settings-heading">
        <h3>节点插图</h3>
        <UploadButton disabled={busy || media.items.length >= maxMediaItems} upload={upload} />
      </div>
      {!!media.items.length && (
        <>
          <label className="param-field">
            <span>插图布局</span>
            <select
              value={media.layout ?? 'auto'}
              onChange={(e) =>
                save({ ...media, layout: e.target.value as NodeVisualMedia['layout'] })
              }
            >
              <option value="auto">自动</option>
              <option value="row">横排</option>
              <option value="grid">2列网格</option>
              <option value="column">竖排</option>
            </select>
          </label>
          <label className="param-field">
            <span>缩略图宽度 · {media.tileWidth ?? defaultTileWidth} px</span>
            <input
              type="range"
              aria-label="缩略图宽度"
              min={80}
              max={240}
              value={media.tileWidth ?? defaultTileWidth}
              onChange={(e) => save({ ...media, tileWidth: e.target.valueAsNumber })}
            />
          </label>
          {media.items.map((item, index) => (
            <fieldset
              key={item.id}
              className="media-item-settings"
              aria-label={`插图 ${index + 1}`}
            >
              <div className="media-item-tools">
                <img src={item.src} alt="" />
                <UploadButton replaceId={item.id} disabled={busy} upload={upload} />
                <button
                  title="上移"
                  aria-label="上移"
                  disabled={!index}
                  onClick={() => move(index, -1)}
                >
                  <ArrowUp size={14} />
                </button>
                <button
                  title="下移"
                  aria-label="下移"
                  disabled={index === media.items.length - 1}
                  onClick={() => move(index, 1)}
                >
                  <ArrowDown size={14} />
                </button>
                <button
                  title="删除插图"
                  aria-label="删除插图"
                  onClick={() =>
                    save({ ...media, items: media.items.filter((i) => i.id !== item.id) })
                  }
                >
                  <Trash2 size={14} />
                </button>
              </div>
              <label className="param-field">
                <span>文字说明</span>
                <input
                  key={item.caption ?? ''}
                  defaultValue={item.caption ?? ''}
                  maxLength={120}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.nativeEvent.isComposing) e.currentTarget.blur();
                  }}
                  onBlur={(e) => {
                    if (e.target.value !== (item.caption ?? ''))
                      update(item.id, { caption: e.target.value });
                  }}
                />
              </label>
              <label className="param-field">
                <span>角标</span>
                <input
                  key={item.badge ?? ''}
                  defaultValue={item.badge ?? ''}
                  maxLength={32}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.nativeEvent.isComposing) e.currentTarget.blur();
                  }}
                  onBlur={(e) => {
                    if (e.target.value !== (item.badge ?? ''))
                      update(item.id, { badge: e.target.value });
                  }}
                />
              </label>
              <label className="param-field">
                <span>样式</span>
                <select
                  value={item.emphasis ?? 'normal'}
                  onChange={(e) =>
                    update(item.id, { emphasis: e.target.value as NodeVisualImage['emphasis'] })
                  }
                >
                  <option value="normal">常规</option>
                  <option value="highlight">高亮</option>
                  <option value="dim">淡化</option>
                </select>
              </label>
            </fieldset>
          ))}
        </>
      )}
    </section>
  );
}
