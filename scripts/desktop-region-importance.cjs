// Isolated native UI regression; build the SDK and desktop before running.
const { app, BrowserWindow, dialog } = require('electron');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const root=path.resolve(__dirname,'..'),directory=fs.mkdtempSync('/tmp/tensor-trail-importance-'),fixture=path.join(directory,'importance.json');
process.env.TENSOR_TRAIL_PROFILE_DIR=path.join(directory,'profile');delete process.env.TENSOR_TRAIL_DEV_URL;
app.commandLine.appendSwitch('force-device-scale-factor','1');
dialog.showOpenDialog=async()=>({canceled:false,filePaths:[fixture]});
require('../apps/desktop/dist/main/index.cjs');
const wait=ms=>new Promise(r=>setTimeout(r,ms)),timer=setTimeout(()=>app.exit(1),120000);
let web;
async function until(fn,label){for(let i=0;i<200;i++){if(await fn())return;await wait(60);}throw Error(label);}
async function run(){
  const sdk=await import(pathToFileURL(path.join(root,'packages/model-core/dist/tensor-trail-agent.js')).href);
  const built=sdk.buildModel({name:'Local importance',nodes:[
    {id:'source',type:'Input',params:{shape:'2,256',axes:'B,C'},position:{x:80,y:140}},
    {id:'block',type:'Submodule',position:{x:500,y:140},subgraph:{nodes:[
      {id:'input',type:'Inlet',position:{x:60,y:140}},
      {id:'output',type:'Outlet',position:{x:600,y:140}},
      {id:'child-region',type:'LayoutRegion',label:'子层同名分区不继承父层的重要性',params:{width:360,height:200},position:{x:0,y:500}},
    ],connections:[['input.out','output.in']]}},
    {id:'result',type:'Output',position:{x:900,y:140}},
    {id:'region',type:'LayoutRegion',label:'当前层级的核心计算',params:{members:'source,block,result',width:1300,height:500},position:{x:0,y:0}},
  ],connections:[['source.out','block.in'],['block.out','result.in']]});
  assert(built.ok,JSON.stringify(built.diagnostics));fs.writeFileSync(fixture,JSON.stringify(built.project));
  await app.whenReady();await until(()=>BrowserWindow.getAllWindows().length===1,'window');
  const win=BrowserWindow.getAllWindows()[0];web=win.webContents;win.setSize(1600,1000);win.setIgnoreMouseEvents(true);
  const errors=[];web.on('console-message',e=>{if(e.level==='error')errors.push(e.message);});
  const js=s=>web.executeJavaScript(s),model=()=>js('window.tensorTrail.getModel().project');
  const button=text=>js(`Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()===${JSON.stringify(text)}).click()`);
  const node=id=>`.react-flow__node[data-id="${id}"]`;
  const select=async id=>{await js(`document.querySelector('${node(id)} .layout-region-heading').dispatchEvent(new MouseEvent('click',{bubbles:true}))`);await wait(120);};
  const rating=async value=>{await js(`document.querySelector('.region-importance-options input[value="${value}"]').click()`);await wait(100);};
  const badge=(id,value)=>js(`document.querySelector('${node(id)} .region-importance').dataset.importance===${JSON.stringify(String(value))}`);
  const press=(keyCode,modifiers=[])=>{web.sendInputEvent({type:'keyDown',keyCode,modifiers});web.sendInputEvent({type:'keyUp',keyCode,modifiers});};
  await until(async()=>!web.isLoadingMainFrame()&&await js('!!window.tensorTrail').catch(()=>false),'ready');
  await button('打开');await until(()=>js(`!!document.querySelector('${node('region')}')`),'loaded');await wait(400);
  const original=await model();
  await select('region');assert(await badge('region',0));await rating(3);assert(await badge('region',3));
  assert.equal(await js(`document.querySelectorAll('${node('region')} .region-importance .filled').length`),3);
  await js(`document.querySelector('.region-importance-options input[value="3"]').focus()`);
  press('Right');await until(()=>badge('region',4),'keyboard rating');
  await rating(5);
  await js(`(()=>{const e=document.querySelector('textarea[aria-label="评级依据"]');e.focus();
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(e,'本层核心特征计算，不与子层比较');
    e.dispatchEvent(new Event('input',{bubbles:true}));e.blur();})()`);await wait(150);
  const rated=await model(),expected=structuredClone(original);
  Object.assign(expected.graph.nodes.find(n=>n.id==='region').params,{importance:5,importance_note:'本层核心特征计算，不与子层比较'});
  assert.deepEqual(rated,expected,'Rating changed unrelated model content');
  await button('清除评级');assert(await badge('region',0));
  await js('document.activeElement.blur()');press('Z',['control']);await wait(250);assert(await badge('region',5));
  press('Z',['control','shift']);await wait(250);assert(await badge('region',0));
  press('Z',['control']);await wait(250);assert(await badge('region',5));
  await select('region');
  for(const theme of ['dark','light']){
    if(await js('document.documentElement.dataset.theme')!==theme)
      await js(`document.querySelector('button[aria-label="${theme==='light'?'切换浅色主题':'切换深色主题'}"]').click()`);
    await wait(200);fs.writeFileSync(path.join(directory,`${theme}.png`),(await web.capturePage()).toPNG());
  }
  await js(`Array.from(document.querySelectorAll('${node('block')} button')).find(b=>b.textContent.includes('进入')).click()`);
  await until(()=>badge('child-region',0),'child starts ungraded');await select('child-region');await rating(2);
  // A minimum-width region keeps its title, stars and count within its header.
  const bounds=await js(`(()=>{const h=document.querySelector('${node('child-region')} .layout-region-heading').getBoundingClientRect();
    return [...document.querySelectorAll('${node('child-region')} .layout-region-title > *')].map(e=>{const r=e.getBoundingClientRect();return {left:r.left-h.left,right:h.right-r.right,width:r.width};});})()`);
  assert(bounds.every(r=>r.left>=0&&r.right>=0&&r.width>0),JSON.stringify(bounds));
  await wait(200);fs.writeFileSync(path.join(directory,'minimum-width.png'),(await web.capturePage()).toPNG());
  await js(`document.querySelector('button[aria-label="上一层级"]').click()`);await until(()=>badge('region',5),'parent unchanged');
  const final=await model();assert.equal(final.graph.nodes.find(n=>n.id==='block').subgraph.nodes.find(n=>n.id==='child-region').params.importance,2);
  await button('保存');await until(()=>js(`document.querySelector('.save-state').dataset.status==='saved'`),'saved');
  assert.deepEqual(JSON.parse(fs.readFileSync(fixture,'utf8')),final);
  await button('打开');await until(()=>badge('region',5),'reopened');assert.deepEqual(await model(),final);
  assert.deepEqual(errors.filter(e=>!e.includes('ResizeObserver')),[]);
  console.log(JSON.stringify({ok:true,directory,cases:['mouse-rating','keyboard-rating','reason','clear','undo-redo','dark-light','minimum-width','level-isolation','save-reopen','unchanged-geometry']}));
  clearTimeout(timer);app.exit(0);
}
run().catch(async error=>{console.error(error,directory);try{if(web&&!web.isDestroyed())fs.writeFileSync(path.join(directory,'failure.png'),(await web.capturePage()).toPNG());}finally{clearTimeout(timer);app.exit(1);}});
