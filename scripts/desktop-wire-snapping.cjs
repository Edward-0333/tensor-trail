const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');

module.exports = async ({
  sdk,
  window,
  web,
  directory,
  scale,
  js,
  model,
  openModel,
  until,
  wait,
  key,
}) => {
  const cases = [];
  const built = sdk.buildModel({
    recipeVersion: 1,
    name: '连线吸附验收',
    nodes: [
      { id: 's', type: 'Input' },
      { id: 'a', type: 'Linear' },
      { id: 'b', type: 'Output' },
    ],
    connections: [
      ['s.out', 'a.in'],
      ['s.out', 'b.in'],
    ],
  });
  assert.equal(built.ok, true);
  const fixture = built.project;
  fixture.graph.nodes.forEach((n) => {
    n.position = { s: { x: 100, y: 240 }, a: { x: 600, y: 100 }, b: { x: 1000, y: 450 } }[n.id];
  });
  const file = path.join(directory, 'fixtures', `wire-snap-${scale}.json`);
  await fs.writeFile(file, JSON.stringify(fixture));
  await openModel(file, 3);
  await wait(150);
  const heading = '.react-flow__node[data-id="a"] .node-heading';
  const source = '.react-flow__node[data-id="s"] .react-flow__handle.source';
  const target = '.react-flow__node[data-id="a"] .react-flow__handle.target';
  const point = (selector) =>
    js(
      `(()=>{const r=document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`,
    );
  const paths = () =>
    js(
      String.raw`(()=>{return [...document.querySelectorAll('.react-flow__edge-interaction')].map(e=>({id:e.closest('.react-flow__edge').dataset.id,points:(e.getAttribute('d').match(/[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:e[-+]?\d+)?/gi)||[]).map(Number)}));})()`,
    );
  await until(async () => (await paths()).length === 2, 'Fan-out routes did not render');
  const routes = await paths();
  assert.ok(
    Math.abs(routes[0].points[2] - routes[1].points[2]) < 1e-6,
    'Different-distance automatic edges use separate trunks',
  );
  assert.ok(Math.abs(routes[0].points[2] - routes[0].points[0]) > 24);
  cases.push('automatic-fanout-shared-trunk-different-target-distances');
  const mouse = (type, p, extra = {}) =>
    web.sendInputEvent({ type, x: Math.round(p.x), y: Math.round(p.y), ...extra });
  const dragNear = async (referenceY, alt = false) => {
    const from = await point(heading),
      port = await point(target);
    const to = { x: from.x + 30, y: from.y + referenceY + 6 - port.y };
    window.focus();
    mouse('mouseMove', from);
    mouse('mouseDown', from, { button: 'left', clickCount: 1, modifiers: alt ? ['alt'] : [] });
    // Cross React Flow's drag-start threshold horizontally before measuring Y alignment.
    mouse(
      'mouseMove',
      { x: from.x + 4, y: from.y },
      { button: 'left', modifiers: alt ? ['leftbuttondown', 'alt'] : ['leftbuttondown'] },
    );
    await wait(20);
    for (let i = 1; i <= 12; i++) {
      mouse(
        'mouseMove',
        { x: from.x + ((to.x - from.x) * i) / 12, y: from.y + ((to.y - from.y) * i) / 12 },
        { button: 'left', modifiers: alt ? ['leftbuttondown', 'alt'] : ['leftbuttondown'] },
      );
      await wait(20);
    }
    if (!alt) {
      await until(
        () => js("document.querySelectorAll('.node-wire-guides line').length>0"),
        'Node wire snap guide missing',
      );
      await fs.writeFile(
        path.join(directory, `wire-snap-preview-${scale}.png`),
        (await web.capturePage()).toPNG(),
      );
    }
    mouse('mouseUp', to, { button: 'left', clickCount: 1, modifiers: alt ? ['alt'] : [] });
    await wait(120);
    const portAfter = await point(target);
    return portAfter.y - (alt ? (await point(source)).y : referenceY);
  };
  const original = (await model()).graph;
  assert.ok(
    Math.abs(await dragNear((await point(source)).y)) < 1.2,
    'Node did not lock its target port onto connected source row',
  );
  const snapped = (await model()).graph;
  assert.notDeepEqual(snapped.nodes, original.nodes);
  assert.deepEqual(snapped.edges, original.edges);
  key('Z');
  await until(
    async () => JSON.stringify((await model()).graph) === JSON.stringify(original),
    'Node snap undo did not restore graph',
  );
  key('Z', ['control', 'shift']);
  await until(
    async () => JSON.stringify((await model()).graph) === JSON.stringify(snapped),
    'Node snap redo failed',
  );
  key('Z');
  await until(
    async () => JSON.stringify((await model()).graph) === JSON.stringify(original),
    'Node snap second undo failed',
  );
  await until(
    () => js(`Boolean(document.querySelector(${JSON.stringify(target)}))`),
    'Node handles did not settle after undo',
  );
  await wait(100);
  cases.push('native-node-wire-snap-preview-and-atomic-undo-redo');
  const gap = await dragNear((await point(source)).y, true);
  assert.ok(gap >= 4 && gap <= 9, `Alt did not bypass node-wire snap: ${gap}`);
  cases.push('alt-bypasses-node-wire-snap');
  key('Z');
  await until(
    async () => JSON.stringify((await model()).graph) === JSON.stringify(original),
    'Alt move undo failed',
  );
  await wait(100);
  const currentPaths = await paths();
  const edgeId = fixture.graph.edges.find((e) => e.target === 'a').id;
  const route = currentPaths.find((e) => e.id === edgeId).points;
  const sourceX = route[0],
    sourceY = route[1],
    targetX = route.at(-2);
  const pin = { x: targetX - 70, y: sourceY + 150 };
  fixture.graph.edges.find((e) => e.id === edgeId).waypoints = [
    { x: sourceX + 80, y: sourceY },
    { x: sourceX + 80, y: pin.y },
    pin,
  ];
  await fs.writeFile(file, JSON.stringify(fixture));
  await openModel(file, 3);
  await wait(150);
  const reference = await js(
    `(()=>{const e=document.querySelector('.react-flow__edge[data-id="${edgeId}"] .react-flow__edge-interaction');return new DOMPoint(${pin.x},${pin.y}).matrixTransform(e.getScreenCTM()).y;})()`,
  );
  assert.ok(
    Math.abs(await dragNear(reference)) < 1.2,
    'Node target stub did not align with its existing manual wire',
  );
  const manual = (await model()).graph;
  key('S');
  await until(
    async () =>
      JSON.stringify(JSON.parse(await fs.readFile(file, 'utf8')).graph) === JSON.stringify(manual),
    'Snapped node position/routes did not save',
  );
  await openModel(file, 3);
  assert.deepEqual((await model()).graph, manual);
  cases.push('manual-endpoint-wire-snap-save-reopen');
  // A redundant, collinear end pin reproduces the reported protrusion: after
  // moving the target left, that old pin would sit to the right of the module.
  await wait(100);
  const tailFixture = await model();
  const tailPath = (await paths()).find((e) => e.id === edgeId).points;
  tailFixture.graph.edges
    .find((e) => e.id === edgeId)
    .waypoints.push({ x: tailPath.at(-2) - 30, y: tailPath.at(-1) });
  await fs.writeFile(file, JSON.stringify(tailFixture));
  await openModel(file, 3);
  await wait(100);
  const beforeTail = (await model()).graph;
  const from = await point(heading),
    oldPort = await point(target);
  const to = { x: from.x - 140, y: from.y };
  window.focus();
  mouse('mouseMove', from);
  mouse('mouseDown', from, { button: 'left', clickCount: 1 });
  mouse(
    'mouseMove',
    { x: from.x - 4, y: from.y },
    { button: 'left', modifiers: ['leftbuttondown'] },
  );
  await wait(20);
  for (let i = 1; i <= 12; i++) {
    mouse(
      'mouseMove',
      { x: from.x + ((to.x - from.x) * i) / 12, y: from.y },
      { button: 'left', modifiers: ['leftbuttondown'] },
    );
    await wait(20);
  }
  const newPort = await point(target);
  assert.ok(newPort.x < oldPort.x - 100, 'Inward drag did not move the module');
  await until(async () => {
    const points = (await paths()).find((e) => e.id === edgeId).points;
    return Math.max(...points.filter((_, i) => i % 2 === 0)) <= points.at(-2) + 0.02;
  }, 'Manual line still protrudes beyond the input during drag');
  const guideRights = await js(
    "[...document.querySelectorAll('.node-wire-guides line')].map(e=>e.getBoundingClientRect().right)",
  );
  assert.ok(guideRights.length > 0);
  assert.ok(
    guideRights.every((x) => x <= newPort.x + 2),
    'Alignment guide crosses the moving module',
  );
  assert.deepEqual((await model()).graph, beforeTail, 'Drag preview changed saved model');
  await fs.writeFile(
    path.join(directory, `endpoint-shortening-preview-${scale}.png`),
    (await web.capturePage()).toPNG(),
  );
  mouse('mouseUp', to, { button: 'left', clickCount: 1 });
  await wait(100);
  const shortened = (await model()).graph;
  assert.notDeepEqual(shortened.edges, beforeTail.edges);
  key('Z');
  await until(
    async () => JSON.stringify((await model()).graph) === JSON.stringify(beforeTail),
    'Endpoint move undo was not atomic',
  );
  key('Z', ['control', 'shift']);
  await until(
    async () => JSON.stringify((await model()).graph) === JSON.stringify(shortened),
    'Endpoint move redo failed',
  );
  key('S');
  await until(
    async () =>
      JSON.stringify(JSON.parse(await fs.readFile(file, 'utf8')).graph) ===
      JSON.stringify(shortened),
    'Shortened manual line did not save',
  );
  await openModel(file, 3);
  assert.deepEqual((await model()).graph, shortened);
  cases.push('native-inward-node-drag-shortens-tail-preview-undo-save');
  return cases;
};
