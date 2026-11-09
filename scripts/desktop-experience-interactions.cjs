const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');

module.exports = async function interactions({
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
}) {
  const cases = [];
  const mouse = (type, x, y, extra = {}) =>
    web.sendInputEvent({ type, x: Math.round(x), y: Math.round(y), ...extra });
  const clickPoint = async (p) => {
    window.focus();
    mouse('mouseMove', p.x, p.y);
    mouse('mouseDown', p.x, p.y, { button: 'left', clickCount: 1 });
    mouse('mouseUp', p.x, p.y, { button: 'left', clickCount: 1 });
    await wait(50);
  };
  const drag = async (p, q, cancel = false) => {
    window.focus();
    mouse('mouseMove', p.x, p.y);
    mouse('mouseDown', p.x, p.y, { button: 'left', clickCount: 1 });
    for (let i = 1; i <= 8; i++) {
      mouse('mouseMove', p.x + ((q.x - p.x) * i) / 8, p.y + ((q.y - p.y) * i) / 8, {
        button: 'left',
        modifiers: ['leftButtonDown'],
      });
      await wait(18);
    }
    if (cancel) key('Escape', []);
    mouse('mouseUp', q.x, q.y, { button: 'left', clickCount: 1 });
    await wait(80);
  };
  const point = (selector) =>
    js(
      `(()=>{const x=document.querySelector(${JSON.stringify(selector)});if(!x)throw Error('Missing '+${JSON.stringify(selector)});const r=x.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`,
    );
  const svgPoint = (selector, ratio = 0.5) =>
    js(
      `(()=>{const x=document.querySelector(${JSON.stringify(selector)});if(!x)throw Error('Missing SVG '+${JSON.stringify(selector)});const p=x.getPointAtLength(x.getTotalLength()*${ratio});const c=new DOMPoint(p.x,p.y).matrixTransform(x.getScreenCTM());return {x:c.x,y:c.y};})()`,
    );
  const clickAria = (label) =>
    js(
      `(()=>{const b=document.querySelector('button[aria-label='+${JSON.stringify(JSON.stringify(label))}+']');if(!b)throw Error('Missing '+${JSON.stringify(label)});b.click();return true;})()`,
    );
  const serialized = () => js('JSON.stringify(window.tensorTrail.getModel().project)');
  const renderedNodes = (ids) =>
    until(
      () =>
        js(
          `JSON.stringify([...document.querySelectorAll('.react-flow__node')].map(x=>x.dataset.id).sort())===${JSON.stringify(JSON.stringify([...ids].sort()))}`,
        ),
      'Child graph node identities did not render',
    );
  const fitAll = async () => {
    await js("document.querySelector('.react-flow__controls-fitview').click();true");
    await until(
      () =>
        js(
          `(()=>{const r=document.querySelector('.react-flow').getBoundingClientRect();const nodes=[...document.querySelectorAll('.react-flow__node')];return nodes.length>0&&nodes.every(n=>{const b=n.getBoundingClientRect();return b.width>0&&b.left>=r.left-1&&b.right<=r.right+1&&b.top>=r.top-1&&b.bottom<=r.bottom+1;});})()`,
        ),
      'Fit-all did not contain every rendered node',
    );
  };

  const initialSize = window.getSize();
  assert.deepEqual(window.getMinimumSize(), [900, 640]);
  window.setSize(900, 640);
  await wait(150);
  const minimum = await js(
    `(()=>{const header=document.querySelector('.topbar');const controls=[...header.querySelectorAll('button,input[aria-label="项目名称"]')].filter(x=>x.getBoundingClientRect().width>0);return {width:innerWidth,height:innerHeight,overflow:document.body.scrollWidth>innerWidth+1||header.scrollWidth>header.clientWidth+1,blocked:controls.filter(x=>{const r=x.getBoundingClientRect();return r.left<0||r.right>innerWidth+1||(!x.disabled&&!x.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)));}).map(x=>x.getAttribute('aria-label')||x.textContent.trim())};})()`,
  );
  assert.equal(minimum.overflow, false);
  assert.deepEqual(minimum.blocked, []);
  await fitAll();
  await fs.writeFile(
    path.join(directory, `minimum-window-${scale}.json`),
    JSON.stringify(minimum, null, 2),
  );
  await fs.writeFile(
    path.join(directory, `minimum-window-${scale}.png`),
    (await web.capturePage()).toPNG(),
  );
  window.setSize(...initialSize);
  await wait(150);
  cases.push('minimum-900x640-controls-and-status');
  const originalProject = await serialized();
  const userProject = await model();
  const compositeIds = userProject.graph.nodes.filter((n) => n.subgraph).map((n) => n.id);
  const userHeading = await js(
    `(()=>{for(const id of ${JSON.stringify(compositeIds)}){const el=document.querySelector('.react-flow__node[data-id="'+id+'"] .node-heading');if(!el)continue;const r=el.getBoundingClientRect(),x=r.x+r.width/2,y=r.y+r.height/2;if(document.elementFromPoint(x,y)?.closest('.react-flow__node')?.dataset.id===id)return {id,x,y};}throw Error('No visible user composite heading');})()`,
  );
  mouse('mouseMove', userHeading.x, userHeading.y);
  mouse('mouseDown', userHeading.x, userHeading.y, { button: 'left', clickCount: 2 });
  mouse('mouseUp', userHeading.x, userHeading.y, { button: 'left', clickCount: 2 });
  await until(
    () => js("document.querySelectorAll('.breadcrumbs button').length>1"),
    'User composite did not open',
  );
  await renderedNodes(
    userProject.graph.nodes.find((n) => n.id === userHeading.id).subgraph.nodes.map((n) => n.id),
  );
  await fitAll();
  await fs.writeFile(
    path.join(directory, `user-subgraph-${scale}.png`),
    (await web.capturePage()).toPNG(),
  );
  await clickPoint(await point('.breadcrumbs button'));
  await until(
    () => js("document.querySelectorAll('.breadcrumbs button').length===1"),
    'User breadcrumb did not return',
  );
  await renderedNodes(userProject.graph.nodes.map((n) => n.id));
  assert.equal(await serialized(), originalProject);
  cases.push('user-copy-subgraph-enter-return-without-mutation');
  const unchanged = await serialized();
  for (const label of ['切换模块库与计算库', '切换属性面板']) await clickAria(label);
  assert.ok(
    await js(
      "document.querySelector('.library').getBoundingClientRect().width<1 && document.querySelector('.inspector').getBoundingClientRect().width<1",
    ),
  );
  for (const label of ['切换模块库与计算库', '切换属性面板']) await clickAria(label);
  assert.equal(await serialized(), unchanged);
  cases.push('panel-toggle-without-model-change');

  const sdk = await import(
    pathToFileURL(path.join(__dirname, '../packages/model-core/dist/tensor-trail-agent.js')).href
  );
  cases.push(
    ...(await require('./desktop-wire-snapping.cjs')({
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
    })),
  );
  const built = sdk.buildModel({
    recipeVersion: 1,
    name: 'D5 连线与拖动',
    nodes: [
      { id: 'input', type: 'Input', params: { shape: '2, 64', axes: 'batch, feature' } },
      { id: 'a', type: 'Linear' },
      { id: 'b', type: 'Linear' },
      { id: 'out_a', type: 'Output' },
      { id: 'out_b', type: 'Output' },
    ],
    connections: [
      ['input.out', 'a.in'],
      ['input.out', 'b.in'],
      ['a.out', 'out_a.in'],
      ['b.out', 'out_b.in'],
    ],
  });
  assert.equal(built.ok, true);
  const fixture = built.project;
  const positions = {
    input: { x: 100, y: 260 },
    a: { x: 600, y: 100 },
    b: { x: 600, y: 450 },
    out_a: { x: 1000, y: 150 },
    out_b: { x: 1000, y: 500 },
  };
  fixture.graph.nodes.forEach((n) => {
    n.position = positions[n.id];
  });
  const fixturePath = path.join(directory, 'fixtures', `interactions-${scale}.json`);
  await fs.writeFile(fixturePath, JSON.stringify(fixture));
  await openModel(fixturePath, 5);
  await wait(100);
  // A real mouse selection rectangle must select multiple cards, not merely update a model API.
  const box = await js(
    `(()=>{const a=document.querySelector('.react-flow__node[data-id="a"]').getBoundingClientRect(),b=document.querySelector('.react-flow__node[data-id="b"]').getBoundingClientRect();return {p:{x:a.x-18,y:a.y-18},q:{x:Math.max(a.right,b.right)+18,y:b.bottom+18}};})()`,
  );
  await drag(box.p, box.q);
  assert.ok(
    await js("document.querySelectorAll('.react-flow__node.selected').length>=2"),
    'Box selection did not select nodes',
  );
  cases.push('native-box-selection');
  await clickPoint(
    await js(
      `(()=>{const r=document.querySelector('.react-flow').getBoundingClientRect();return {x:r.x+25,y:r.y+25};})()`,
    ),
  );

  const search = async (text) => {
    await js(
      `(()=>{const x=document.querySelector('.library .search-field input');x.focus();x.select();return true;})()`,
    );
    await web.insertText(text);
    await wait(70);
  };
  const librarySelector = '.library-module';
  if (!web.debugger.isAttached()) web.debugger.attach('1.3');
  await web.debugger.sendCommand('Input.setInterceptDrags', { enabled: true });
  const dropModule = async (label) => {
    await search(label);
    const source = await js(
      `(()=>{const b=[...document.querySelectorAll(${JSON.stringify(librarySelector)})].find(n=>n.textContent.trim()===${JSON.stringify(label)});if(!b)throw Error('Missing library row '+${JSON.stringify(label)});b.scrollIntoView({block:'nearest'});const r=b.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`,
    );
    const before = (await model()).graph.nodes.length;
    await clickPoint(source);
    assert.equal((await model()).graph.nodes.length, before, 'Click unexpectedly added module');
    const intercepted = new Promise((resolve, reject) => {
      const limit = setTimeout(() => {
        web.debugger.removeListener('message', listener);
        reject(Error('HTML dragstart was not intercepted'));
      }, 5000);
      const listener = (_e, method, params) => {
        if (method === 'Input.dragIntercepted') {
          clearTimeout(limit);
          web.debugger.removeListener('message', listener);
          resolve(params.data);
        }
      };
      web.debugger.on('message', listener);
    });
    window.focus();
    mouse('mouseMove', source.x, source.y);
    mouse('mouseDown', source.x, source.y, { button: 'left', clickCount: 1 });
    mouse('mouseMove', source.x + 15, source.y + 2, {
      button: 'left',
      modifiers: ['leftButtonDown'],
    });
    const data = await intercepted;
    assert.ok(data.items.some((item) => item.mimeType === 'application/x-tensor-trail-module'));
    const target = await js(
      `(()=>{const r=document.querySelector('.react-flow').getBoundingClientRect();return {x:r.x+r.width*.36,y:r.bottom-140};})()`,
    );
    for (const type of ['dragEnter', 'dragOver', 'drop'])
      await web.debugger.sendCommand('Input.dispatchDragEvent', {
        type,
        x: target.x,
        y: target.y,
        data,
      });
    mouse('mouseUp', target.x, target.y, { button: 'left', clickCount: 1 });
    await until(
      async () => (await model()).graph.nodes.length === before + 1,
      'Module drop did not add exactly one node',
    );
    return (await model()).graph.nodes.at(-1);
  };
  const output = await dropModule('输出 Tensor');
  assert.equal(output.type, 'Output');
  cases.push('library-drag-add-and-click-no-add');
  // Connect the new output using the actual handles.
  const sourceHandle = await point('.react-flow__node[data-id="input"] .react-flow__handle.source');
  const targetHandle = await point(
    `.react-flow__node[data-id="${output.id}"] .react-flow__handle.target`,
  );
  const edgesBefore = (await model()).graph.edges.length;
  await drag(sourceHandle, targetHandle);
  await until(
    async () => (await model()).graph.edges.length === edgesBefore + 1,
    'Native port connection failed',
  );
  cases.push('native-port-connection');
  await clickPoint(await point('.react-flow__node[data-id="b"] .node-heading'));
  for (const [label, value] of [
    ['输入端口位置', 'top'],
    ['输出端口位置', 'bottom'],
  ]) {
    await js(
      `(()=>{const x=document.querySelector('select[aria-label='+${JSON.stringify(JSON.stringify(label))}+']');x.value=${JSON.stringify(value)};x.dispatchEvent(new Event('change',{bubbles:true}));return true;})()`,
    );
  }
  await until(
    async () =>
      (await model()).graph.nodes.find((n) => n.id === 'b').portLayout?.output === 'bottom',
    'Port direction was not applied',
  );
  assert.ok(
    await js(
      'Boolean(document.querySelector(\'.react-flow__node[data-id="b"] .react-flow__handle-top\'))',
    ),
  );
  cases.push('top-bottom-port-layout');

  let e = (await model()).graph.edges.find((e) => e.source === 'input' && e.target === 'a').id;
  let edgeRoot = `.react-flow__edge[data-id="${e}"]`;
  await until(
    () =>
      js(
        `Boolean(document.querySelector(${JSON.stringify(edgeRoot + ' .react-flow__edge-interaction')}))`,
      ),
    'Wire SVG did not settle after handle layout update',
  );
  await clickPoint(await svgPoint(edgeRoot + ' .react-flow__edge-interaction', 0.1));
  // Overlapping branches share one physical trunk. Hit testing may select any owner.
  await until(
    () =>
      js(
        "[...document.querySelectorAll('.edge-segment-hit')].some(x=>Number(x.dataset.sharedCount)>=2)",
      ),
    'Shared segment controls absent',
  );
  e = await js(
    "[...document.querySelectorAll('.edge-segment-hit')].find(x=>Number(x.dataset.sharedCount)>=2).closest('.react-flow__edge').dataset.id",
  );
  assert.equal((await model()).graph.edges.find((x) => x.id === e).source, 'input');
  edgeRoot = `.react-flow__edge[data-id="${e}"]`;
  const segment =
    edgeRoot +
    ' .edge-segment-hit[data-shared-count="3"],' +
    edgeRoot +
    ' .edge-segment-hit[data-shared-count="2"]';
  const p = await js(
    `(()=>{const found=[...document.querySelectorAll(${JSON.stringify(segment)})].map(x=>{const length=x.getTotalLength();for(const ratio of [.5,.3,.7]){const p=x.getPointAtLength(length*ratio);const q=new DOMPoint(p.x,p.y).matrixTransform(x.getScreenCTM());if(document.elementFromPoint(q.x,q.y)===x)return {x:q.x,y:q.y,horizontal:x.classList.contains('horizontal'),length};}return null;}).filter(Boolean).sort((a,b)=>b.length-a.length)[0];if(!found)throw Error('No visible shared segment hit target');return found;})()`,
  );
  await js(
    `window.__d5WireEvents=[];for(const t of ['pointerdown','pointermove','pointerup','pointercancel','lostpointercapture','gotpointercapture'])document.addEventListener(t,x=>{if(window.__d5WireEvents.length<60)window.__d5WireEvents.push({type:x.type,target:x.target.getAttribute?.('class'),edge:x.target.closest?.('.react-flow__edge')?.dataset.id,x:x.clientX,y:x.clientY,buttons:x.buttons});},{capture:true});true`,
  );
  const originalPositions = JSON.stringify(
    (await model()).graph.nodes.map((n) => [n.id, n.position]),
  );
  const routesBefore = JSON.stringify((await model()).graph.edges);
  await drag(p, { x: p.x + (p.horizontal ? 0 : 32), y: p.y + (p.horizontal ? 32 : 0) });
  await fs.writeFile(
    path.join(directory, `wire-events-${scale}.json`),
    JSON.stringify(
      await js(
        '({events:window.__d5WireEvents,segments:[...document.querySelectorAll(".edge-segment-hit")].map(x=>({d:x.getAttribute("d"),shared:x.dataset.sharedCount}))})',
      ),
      null,
      2,
    ),
  );
  await until(
    async () => JSON.stringify((await model()).graph.edges) !== routesBefore,
    'Shared wire move did not commit',
  );
  const moved = (await model()).graph.edges.filter(
    (e) => e.source === 'input' && e.waypoints?.length,
  );
  assert.ok(moved.length >= 2, 'Shared wire did not update its peer');
  assert.equal(
    JSON.stringify((await model()).graph.nodes.map((n) => [n.id, n.position])),
    originalPositions,
  );
  const movedRoutes = JSON.stringify((await model()).graph.edges);
  key('Z');
  await until(
    async () => JSON.stringify((await model()).graph.edges) === routesBefore,
    'Shared wire undo was not atomic',
  );
  key('Z', ['control', 'shift']);
  await until(
    async () => JSON.stringify((await model()).graph.edges) === movedRoutes,
    'Shared wire redo failed',
  );
  cases.push('shared-wire-move-atomic-undo-redo');
  await until(
    () =>
      js(
        `Boolean(document.querySelector(${JSON.stringify(edgeRoot + ' .react-flow__edge-interaction')}))`,
      ),
    'Wire SVG did not settle after redo',
  );
  await clickPoint(await svgPoint(edgeRoot + ' .react-flow__edge-interaction', 0.1));
  await until(
    () => js("Boolean(document.querySelector('.edge-segment-hit'))"),
    'Wire selection did not settle after redo',
  );
  e = await js(
    "document.querySelector('.edge-segment-hit').closest('.react-flow__edge').dataset.id",
  );
  edgeRoot = `.react-flow__edge[data-id="${e}"]`;
  const cancelSegment = edgeRoot + ' .edge-segment-hit';
  const cancelPoint = await svgPoint(cancelSegment);
  await drag(cancelPoint, { x: cancelPoint.x + 25, y: cancelPoint.y + 25 }, true);
  assert.equal(JSON.stringify((await model()).graph.edges), movedRoutes);
  cases.push('escape-cancels-wire-preview');
  const labelSelector = edgeRoot + ' .edge-side-label';
  const labelPoint = await point(labelSelector);
  const destination = await svgPoint(edgeRoot + ' .react-flow__edge-interaction', 0.75);
  await drag(labelPoint, destination);
  await until(
    async () => (await model()).graph.edges.find((x) => x.id === e).labelPosition !== undefined,
    'Edge label drag failed',
  );
  cases.push('native-edge-label-drag');
  const pins = await js("document.querySelectorAll('.edge-control-point').length");
  assert.ok(pins > 0);
  const pointBefore = JSON.stringify((await model()).graph.edges);
  const pin = await point('.edge-control-point');
  await drag(pin, { x: pin.x + 20, y: pin.y + 25 });
  await until(
    async () => JSON.stringify((await model()).graph.edges) !== pointBefore,
    'Manual control point did not move',
  );
  cases.push('native-manual-control-point');

  const nat = await dropModule('NATSequenceEncoder');
  assert.ok(nat.subgraph);
  const natId = nat.id;
  const natCenter = await point(`.react-flow__node[data-id="${natId}"] .node-heading`);
  mouse('mouseDown', natCenter.x, natCenter.y, { button: 'left', clickCount: 2 });
  mouse('mouseUp', natCenter.x, natCenter.y, { button: 'left', clickCount: 2 });
  await until(
    () => js("document.querySelectorAll('.breadcrumbs button').length>1"),
    'NAT module did not open',
  );
  await renderedNodes(nat.subgraph.nodes.map((n) => n.id));
  await fitAll();
  await clickPoint(await point('.breadcrumbs button'));
  await until(
    () => js("document.querySelectorAll('.breadcrumbs button').length===1"),
    'Breadcrumb did not return',
  );
  await renderedNodes((await model()).graph.nodes.map((n) => n.id));
  cases.push('pluto-nat-enter-and-breadcrumb-return');
  const snapshot = await model();
  key('S');
  await until(async () => !window.getTitle().startsWith('●'), 'Edited fixture did not save');
  const disk = JSON.parse(await fs.readFile(fixturePath, 'utf8'));
  assert.deepEqual(disk.graph.edges, snapshot.graph.edges);
  assert.deepEqual(disk.graph.nodes, snapshot.graph.nodes);
  await openModel(fixturePath, disk.graph.nodes.length);
  assert.deepEqual((await model()).graph, disk.graph);
  cases.push('layout-and-routes-save-reopen');
  const image = await web.capturePage();
  await fs.writeFile(path.join(directory, `interactions-${scale}.png`), image.toPNG());
  const prior = await js('window.tensorTrail.getModel().revision');
  await js(
    "(()=>{const x=document.querySelector('select[aria-label=\"载入示例\"]');x.value='pluto-input';x.dispatchEvent(new Event('change',{bubbles:true}));return true;})()",
  );
  await until(
    async () =>
      (await js(`window.tensorTrail.getModel().revision!==${JSON.stringify(prior)}`)) &&
      (await model()).graph.nodes.length === 1 &&
      (await model()).graph.nodes[0].type === 'ForwardInput',
    'Built-in PLUTO example did not open',
  );
  assert.ok(await js("Boolean(document.querySelector('.react-flow__node'))"));
  cases.push('built-in-pluto-forward-input-example');
  web.debugger.detach();
  return cases;
};
