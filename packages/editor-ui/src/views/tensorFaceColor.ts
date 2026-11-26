export function tensorFaceColor(tone: string, face: 'top' | 'front' | 'side' | 'stroke') {
  if (!tone.startsWith('#')) return `var(--tv-${tone}-${face})`;
  if (face === 'front') return tone;
  return `color-mix(in srgb, ${tone} var(--tv-${face}-weight), var(--tv-${face}-mix))`;
}
