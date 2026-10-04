"""Offline HTML monitor for one MethodTrail project/session.

The page deliberately has no CDN dependency.  It can be opened directly from
the filesystem while a run is paused or after it finishes.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def write_dashboard(target: str | Path, payload: dict[str, Any]) -> Path:
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = escape(str(payload.get("project", {}).get("project_id", "MethodTrail")))
    target_path.write_text(
        _HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data),
        encoding="utf-8",
    )
    return target_path


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MethodTrail · __TITLE__</title>
<style>
:root { color-scheme: light; --bg:#f6f7f9; --panel:#ffffff; --line:#d9dee7; --text:#202631; --muted:#697386; --accent:#356ae6; --good:#2d8a63; --warn:#b7791f; --bad:#c4475a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { padding:18px 24px 12px; border-bottom:1px solid var(--line); display:flex; gap:20px; align-items:center; flex-wrap:wrap; }
h1 { margin:0; font-size:22px; }
select, button { background:#fff; color:var(--text); border:1px solid #c8ced8; border-radius:5px; padding:7px 10px; }
button { cursor:pointer; }
.layout { display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:14px; padding:14px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; min-width:0; }
.toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
.toolbar span { color:var(--muted); }
#canvas { width:100%; height:650px; background:#fff; border-radius:6px; border:1px solid #d9dee7; touch-action:none; }
.hint { color:var(--muted); margin:5px 0 0; }
#details { overflow:auto; max-height:570px; color:var(--text); }
#details h3 { margin:0 0 8px; font-size:16px; }
#details .detail-kind { color:var(--muted); font-size:12px; margin-bottom:8px; }
#details .detail-row { display:grid; grid-template-columns:112px minmax(0,1fr); gap:8px; padding:7px 0; border-bottom:1px solid #edf0f4; align-items:start; }
#details .detail-label { color:var(--muted); font-size:12px; }
#details .detail-value { white-space:pre-wrap; overflow-wrap:anywhere; color:var(--text); }
#details .detail-value.metric { color:var(--good); font-weight:700; }
#details .detail-list { margin:0; padding-left:18px; }
#details .detail-object { display:grid; gap:4px; }
#details .raw-details { margin-top:12px; border-top:1px solid var(--line); padding-top:8px; }
#details .raw-details summary { cursor:pointer; color:var(--muted); font-size:12px; }
#details pre { margin:7px 0 0; white-space:pre-wrap; overflow-wrap:anywhere; font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; }
.metric { font-size:20px; color:var(--good); margin:4px 0 10px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px; margin-top:10px; }
.card { border:1px solid var(--line); border-radius:6px; padding:9px; background:#fff; }
.card b { color:var(--accent); }
.event { border-left:3px solid var(--accent); padding:6px 9px; margin-top:7px; background:#f8fafc; border-radius:4px; }
.event.good { border-left-color:var(--good); }
.event.bad { border-left-color:var(--bad); }
svg text { font-family:inherit; }
.edge { stroke:#9aa5b5; stroke-width:1.6; opacity:.88; marker-end:url(#arrow); pointer-events:none; }
.edge-hit { stroke:transparent; stroke-width:14; fill:none; cursor:pointer; pointer-events:stroke; }
.edge-group.selected .edge { stroke:#b7791f; stroke-width:3; opacity:1; }
.edge-group.selected .edge-label { fill:#8b5e13; font-weight:700; }
.edge.lineage, .edge.follows { stroke:#3d9a70; }
.edge.evidence, .edge.same_family { stroke:#8b6fc4; stroke-dasharray:6 4; }
.edge.shared_factor, .edge.related { stroke:#c58a2b; stroke-dasharray:3 4; }
.edge-label { fill:#4b5565; font-size:9px; pointer-events:none; paint-order:stroke; stroke:#fff; stroke-width:3px; stroke-linejoin:round; text-anchor:middle; }
.edge.related { stroke:#8b6fc4; stroke-dasharray:5 4; }
.edge.candidate { stroke:#356ae6; stroke-dasharray:3 4; }
.edge.execution { stroke:#3d9a70; }
.node { cursor:grab; stroke-width:2; }
.node:active { cursor:grabbing; }
.node.proposal { stroke:#356ae6; }
.node.outcome { stroke:#3d9a70; }
.node.origin { stroke-dasharray:4 2; }
.node.failed { stroke:#c4475a; }
.node.selected { stroke:#b7791f; stroke-width:4; }
.node-text { fill:#202631; font-size:11px; pointer-events:none; text-anchor:middle; }
.node-round { fill:#4b5565; font-size:8px; pointer-events:none; text-anchor:middle; }
@media(max-width:900px){ .layout{grid-template-columns:1fr;} #details{max-height:none;} }
</style>
</head>
<body>
<header><h1>MethodTrail · __TITLE__</h1><label>轮次 <select id="iteration"></select></label><button id="methodBtn">方法图</button><button id="memoryBtn">经验图</button><button id="allBtn">全部</button><span id="status"></span></header>
<div class="layout"><main class="panel"><div class="toolbar"><span id="viewName">方法图</span><span>左键点击选择；按住左键拖动节点。箭头表示方向。</span><span id="relationHint"></span></div><svg id="canvas" viewBox="0 0 1000 570" preserveAspectRatio="xMidYMid meet"></svg><div id="summary" class="cards"></div></main><aside class="panel"><h2>节点或边详情</h2><div id="details">选择一个节点或有向边</div></aside></div>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
let view = 'method', round = 'all';
const positionStore = new Map();
let activePositions = null, activeGraphKey = '', dragging = null, lastDragMoved = false;
let selectedNodeId = null, selectedEdgeKey = null;
const positionStoragePrefix = `methodtrail-pos|${String(DATA.project?.project_id || '')}|${String(DATA.session?.session_id || '')}`;
const method = DATA.method_graph || {nodes:[], edges:[]};
const memory = DATA.memory_graph || {nodes:[], edges:[]};
function label(n){ return String(n.title || n.conclusion || n.method_family || n.card_id || n.variant_id || n.node_id || '').slice(0,25); }
function nodeIds(n){ return [n.card_id,n.id,n.node_id,n.variant_id].filter(x=>x!=null).map(String); }
function nodeId(n){ return nodeIds(n)[0]; }
function groupKey(n){ return String(n.method_family || n.method?.family || n.mutation_class || n.relation || 'other'); }
const COLORS=['#b9d0ff','#bde8d3','#ffe2a8','#dccbff','#ffd0c2','#bde8ec','#f4c4df','#d9e9ad'];
function groupColor(key){ let h=0; for(const ch of key) h=(h*31+ch.charCodeAt(0))%COLORS.length; return COLORS[h]; }
function restorePositions(key, positions){
  try{
    const saved=JSON.parse(localStorage.getItem(positionStoragePrefix+'|'+key) || '{}');
    positions.forEach((_, id)=>{ const p=saved[id]; if(p && Number.isFinite(p.x) && Number.isFinite(p.y)) positions.set(id,{x:p.x,y:p.y}); });
  }catch(_){ }
  return positions;
}
function persistPositions(){
  if(!activePositions || !activeGraphKey)return;
  try{ localStorage.setItem(positionStoragePrefix+'|'+activeGraphKey, JSON.stringify(Object.fromEntries(activePositions))); }catch(_){ }
}
function edgeLabel(e){
  const base=String(e.relation||e.edge_type||e.label||'relation').replaceAll('_',' ');
  const targetChange=e.target_change||{};
  if(view==='memory' && targetChange.iteration!=null) return `${base} · 第${targetChange.iteration}轮`;
  return base;
}
function edgeKey(e){ return `${String(e.source)}→${String(e.target)}|${String(e.edge_type||e.relation||'')}`; }
function findNode(id){ const wanted=String(id); return (view==='memory'?memory:method).nodes.find(n=>nodeIds(n).includes(wanted)); }
function changeOf(n){
  if(!n) return {};
  return n.change || n.change_request || {
    iteration:n.iteration, title:n.title, mutation_class:n.mutation_class,
    change_logic:n.change_logic, changed_factors:n.changed_factors,
    method_components:n.method_components, question:n.question,
  };
}
function visibleGraph(){
  const g = view === 'memory' ? memory : method;
  let nodes = (g.nodes || []).filter(n => n.node_type !== 'root' && n.variant_id !== '__project_root__' && n.id !== '__project_root__' && n.node_id !== '__project_root__');
  nodes = nodes.filter(n => round === 'all' || String(n.iteration || '') === String(round));
  if (round !== 'all' && !nodes.length) nodes = (g.nodes || []).filter(n => n.node_type !== 'root' && n.variant_id !== '__project_root__' && n.id !== '__project_root__' && n.node_id !== '__project_root__');
  const ids = new Set(nodes.flatMap(nodeIds));
  let edges = (g.edges || []).filter(e => ids.has(String(e.source)) && ids.has(String(e.target)));
  return {nodes, edges};
}
function layout(nodes, edges){
  const positions=new Map(), points=new Map(), groups=[...new Set(nodes.map(groupKey))];
  const cols=Math.max(1,Math.ceil(Math.sqrt(groups.length))), rows=Math.max(1,Math.ceil(groups.length/cols));
  const centers=new Map(); groups.forEach((g,i)=>{ const col=i%cols, row=Math.floor(i/cols); centers.set(g,{x:180+(col+0.5)*(640/cols),y:110+(row+0.5)*(350/rows)}); });
  nodes.forEach((n,i)=>{ const c=centers.get(groupKey(n)); const angle=(i*2.399)-Math.PI/2; const radius=35+(i%4)*16; points.set(nodeId(n),{x:c.x+Math.cos(angle)*radius,y:c.y+Math.sin(angle)*radius}); });
  const lookup=new Map(); nodes.forEach(n=>nodeIds(n).forEach(id=>lookup.set(id,nodeId(n))));
  for(let step=0;step<100;step++){
    const force=new Map(nodes.map(n=>[nodeId(n),{x:0,y:0}]));
    for(let i=0;i<nodes.length;i++) for(let j=i+1;j<nodes.length;j++){
      const a=points.get(nodeId(nodes[i])), b=points.get(nodeId(nodes[j])); let dx=a.x-b.x, dy=a.y-b.y, d=Math.max(18,Math.hypot(dx,dy));
      const push=1700/(d*d), fx=push*dx/d, fy=push*dy/d; force.get(nodeId(nodes[i])).x+=fx; force.get(nodeId(nodes[i])).y+=fy; force.get(nodeId(nodes[j])).x-=fx; force.get(nodeId(nodes[j])).y-=fy;
    }
    edges.forEach(e=>{ const sa=lookup.get(String(e.source)), sb=lookup.get(String(e.target)); if(!sa||!sb||sa===sb)return; const a=points.get(sa),b=points.get(sb); let dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)); const pull=(d-145)*0.012,fx=pull*dx/d,fy=pull*dy/d; force.get(sa).x+=fx;force.get(sa).y+=fy;force.get(sb).x-=fx;force.get(sb).y-=fy; });
    nodes.forEach(n=>{ const id=nodeId(n), p=points.get(id), c=centers.get(groupKey(n)), f=force.get(id); f.x+=(c.x-p.x)*0.018; f.y+=(c.y-p.y)*0.018; p.x=Math.max(45,Math.min(955,p.x+f.x*0.45)); p.y=Math.max(45,Math.min(525,p.y+f.y*0.45)); });
  }
  nodes.forEach(n=>nodeIds(n).forEach(id=>positions.set(id,points.get(nodeId(n)))));
  return positions;
}
function svgPoint(event){
  const svg=$('canvas'), point=svg.createSVGPoint();
  point.x=event.clientX; point.y=event.clientY;
  return point.matrixTransform(svg.getScreenCTM().inverse());
}
function startDrag(event, node){
  if(event.button!=null && event.button!==0)return;
  event.stopPropagation();
  const ids=nodeIds(node), id=nodeId(node), point=svgPoint(event), old=activePositions.get(id);
  dragging={ids, id, node, startX:point.x, startY:point.y, offsetX:old.x-point.x, offsetY:old.y-point.y, pointerId:event.pointerId, captured:false};
  lastDragMoved=false;
}
function moveDrag(event){
  if(!dragging || !activePositions)return;
  const point=svgPoint(event), p={x:point.x+dragging.offsetX,y:point.y+dragging.offsetY};
  if(!dragging.captured && Math.hypot(point.x-dragging.startX,point.y-dragging.startY)<=4)return;
  if(!dragging.captured){
    dragging.captured=true;
    try{$('canvas').setPointerCapture(dragging.pointerId);}catch(_){ }
  }
  p.x=Math.max(30,Math.min(970,p.x)); p.y=Math.max(30,Math.min(620,p.y));
  const old=activePositions.get(dragging.id); if(Math.hypot(p.x-old.x,p.y-old.y)<1)return;
  dragging.ids.forEach(id=>activePositions.set(id,p)); persistPositions(); lastDragMoved=true; draw(true);
}
function endDrag(event){
  if(!dragging)return;
  const finished=dragging;
  if(finished.captured){try{$('canvas').releasePointerCapture(finished.pointerId);}catch(_){ }}
  dragging=null;
  if(lastDragMoved)setTimeout(()=>{lastDragMoved=false;},0);
}
function draw(){
  const svg = $('canvas'); while(svg.firstChild) svg.removeChild(svg.firstChild);
  const g = visibleGraph(), nodes=g.nodes, edges=g.edges;
  $('viewName').textContent = view === 'memory' ? '经验图' : '方法图';
  $('relationHint').textContent = view === 'memory'
    ? '经验图：箭头为前序证据 → 后续实验；follows=父版本，informed by=决策依据，same family/shared factor=相关证据'
    : '方法图：箭头只表示方法关系；轮次和修改逻辑在节点详情中查看';
  const graphKey=view+'|'+round;
  let positions=positionStore.get(graphKey);
  if(!positions){ positions=restorePositions(graphKey,layout(nodes, edges)); positionStore.set(graphKey,positions); }
  activeGraphKey=graphKey; activePositions=positions;
  const lookup=new Map(); nodes.forEach(n=>nodeIds(n).forEach(id=>lookup.set(id,n)));
  const defs=document.createElementNS('http://www.w3.org/2000/svg','defs');
  defs.innerHTML='<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L10,5 L0,10 Z" fill="#7b8798"/></marker>';
  svg.appendChild(defs);
  edges.forEach((e,index)=>{
    const a=positions.get(String(e.source)),b=positions.get(String(e.target)); if(!a||!b)return;
    const dx=b.x-a.x, dy=b.y-a.y, distance=Math.max(1,Math.hypot(dx,dy)), radius=36;
    const x1=a.x+dx*radius/distance, y1=a.y+dy*radius/distance;
    const x2=b.x-dx*radius/distance, y2=b.y-dy*radius/distance;
    const group=document.createElementNS('http://www.w3.org/2000/svg','g');
    group.setAttribute('class','edge-group'+(selectedEdgeKey===edgeKey(e)?' selected':''));
    const hit=document.createElementNS('http://www.w3.org/2000/svg','line');
    hit.setAttribute('x1',x1); hit.setAttribute('y1',y1); hit.setAttribute('x2',x2); hit.setAttribute('y2',y2); hit.setAttribute('class','edge-hit');
    hit.onclick=event=>{event.stopPropagation();showEdgeDetails(e);};
    const line=document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1',x1); line.setAttribute('y1',y1); line.setAttribute('x2',x2); line.setAttribute('y2',y2); line.setAttribute('class','edge '+(e.edge_type||e.relation||''));
    const title=document.createElementNS('http://www.w3.org/2000/svg','title'); title.textContent=edgeLabel(e)+' · '+(e.reason||''); line.appendChild(title);
    const text=document.createElementNS('http://www.w3.org/2000/svg','text');
    const offset=((index%3)-1)*11;
    text.setAttribute('x',(x1+x2)/2+offset); text.setAttribute('y',(y1+y2)/2-5+offset); text.setAttribute('class','edge-label'); text.textContent=edgeLabel(e);
    group.appendChild(hit); group.appendChild(line); group.appendChild(text); svg.appendChild(group);
  });
  nodes.forEach(n=>{
    const p=positions.get(nodeId(n)); if(!p)return;
    const group=document.createElementNS('http://www.w3.org/2000/svg','g');
    const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',34);c.setAttribute('fill',groupColor(groupKey(n)));
    c.setAttribute('class','node '+(n.node_type||'')+(n.origin?' origin':'')+(n.status==='failed'?' failed':'')+(selectedNodeId===nodeId(n)?' selected':''));
    group.onpointerdown=e=>startDrag(e,n);
    group.onclick=e=>{e.stopPropagation();if(!lastDragMoved)showDetails(n);};
    group.appendChild(c);
    const title=document.createElementNS('http://www.w3.org/2000/svg','title');
    title.textContent=`${n.iteration==null?'':`第${n.iteration}轮 · `}${String(n.title||n.conclusion||n.method_family||'')}`;
    group.appendChild(title);
    const t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',p.x);t.setAttribute('y',p.y+4);t.setAttribute('class','node-text');t.textContent=label(n);group.appendChild(t);svg.appendChild(group);
    if(n.iteration!=null){
      const r=document.createElementNS('http://www.w3.org/2000/svg','text');r.setAttribute('x',p.x);r.setAttribute('y',p.y+20);r.setAttribute('class','node-round');r.textContent=`${n.origin?'起点 · ':''}第${n.iteration}轮`;group.appendChild(r);
    }
  });
  const events=(DATA.events||[]).filter(e=>['iteration_finished','technical_attempt','research_stop'].includes(e.kind));
  const shown=round==='all'?events:events.filter(e=>String(e.payload?.iteration||'')===String(round));
  const eventHtml=shown.slice(-6).map(e=>{const p=e.payload||{}; const bad=e.kind==='technical_attempt'; return `<div class="event ${bad?'bad':'good'}"><b>${e.kind}</b> · 第${p.iteration||'?'}轮 · ${p.decision||p.reason||''}<br>指标：${p.metric==null?'未测得':p.metric} ${p.next_question?'<br>下一问题：'+p.next_question:''}</div>`;}).join('');
  const groups=[...new Set(nodes.map(groupKey))]; const legend=groups.map(g=>`<span style="color:${groupColor(g)}">● ${g}</span>`).join(' · ');
  $('summary').innerHTML = `<div class="card"><b>节点</b><br>${nodes.length}</div><div class="card"><b>有向关系</b><br>${edges.length}</div><div class="card"><b>方法族</b><br>${groups.length}<br>${legend}</div>${eventHtml}`;
}
const FIELD_LABELS={
  id:'节点 ID', title:'标题', node_type:'节点类型', iteration:'研究轮次',
  parent_variant_id:'代码父版本', evidence_parent_ids:'证据父版本', method_family:'方法族',
  relation:'关系', edge_type:'边类型', direction:'方向', status:'状态', decision:'评估决定',
  metric:'主指标', wall_seconds:'运行时间（秒）', question:'研究问题', change_logic:'修改逻辑',
  changed_factors:'改动因素', method_components:'方法组件', conclusion:'经验结论',
  evidence_summary:'证据摘要', applicable_conditions:'适用条件', reason:'关系说明',
  shared_factors:'共享改动因素', target_change:'目标节点的本轮修改', source:'来源节点', target:'目标节点'
};
function detailValue(parent,key,value,depth=0){
  if(value==null || value==='' || (Array.isArray(value)&&value.length===0)){parent.textContent='未记录';return;}
  if(depth>2 && typeof value==='object'){parent.textContent=JSON.stringify(value);return;}
  if(Array.isArray(value)){
    const list=document.createElement('ul'); list.className='detail-list';
    value.forEach(item=>{const li=document.createElement('li'); detailValue(li,'',item,depth+1); list.appendChild(li);});
    parent.appendChild(list); return;
  }
  if(typeof value==='object'){
    const object=document.createElement('div'); object.className='detail-object';
    Object.entries(value).forEach(([childKey,childValue])=>{
      const row=document.createElement('div'); row.className='detail-row';
      const label=document.createElement('div'); label.className='detail-label'; label.textContent=FIELD_LABELS[childKey]||childKey;
      const body=document.createElement('div'); body.className='detail-value'; detailValue(body,childKey,childValue,depth+1);
      row.append(label,body); object.appendChild(row);
    });
    parent.appendChild(object); return;
  }
  parent.textContent=String(value);
  if(key==='metric')parent.classList.add('metric');
}
function renderDetails(kind, title, payload){
  const box=$('details'); box.replaceChildren();
  const h=document.createElement('h3'); h.textContent=title; box.appendChild(h);
  const k=document.createElement('div'); k.className='detail-kind'; k.textContent=kind; box.appendChild(k);
  if(payload && typeof payload==='object' && !Array.isArray(payload)){
    Object.entries(payload).forEach(([key,value])=>{
      const row=document.createElement('div'); row.className='detail-row';
      const label=document.createElement('div'); label.className='detail-label'; label.textContent=FIELD_LABELS[key]||key;
      const body=document.createElement('div'); body.className='detail-value'; detailValue(body,key,value);
      row.append(label,body); box.appendChild(row);
    });
  }else{
    const body=document.createElement('div'); body.className='detail-value'; detailValue(body,'',payload); box.appendChild(body);
  }
  if(payload && typeof payload==='object'){
    const raw=document.createElement('details'); raw.className='raw-details';
    const summary=document.createElement('summary'); summary.textContent='查看原始数据'; raw.appendChild(summary);
    const pre=document.createElement('pre'); pre.textContent=JSON.stringify(payload,null,2); raw.appendChild(pre); box.appendChild(raw);
  }
}
function showDetails(n){
  selectedNodeId=nodeId(n); selectedEdgeKey=null;
  renderDetails('节点', String(n.title || n.conclusion || n.method_family || label(n)), {
    id:nodeId(n), title:n.title, node_type:n.node_type, origin:n.origin, origin_reason:n.origin_reason, iteration:n.iteration,
    parent_variant_id:n.parent_variant_id, evidence_parent_ids:n.evidence_parent_ids,
    method_family:n.method_family || n.method?.family, relation:n.relation,
    status:n.status, decision:n.decision, metric:n.metric, wall_seconds:n.wall_seconds, question:n.question,
    change_logic:n.change_logic || n.change?.change_logic,
    changed_factors:n.changed_factors || n.method?.changed_factors,
    method_components:n.method_components || n.method?.components,
    conclusion:n.conclusion, evidence_summary:n.evidence_summary,
    applicable_conditions:n.applicable_conditions,
  });
  draw();
}
function showEdgeDetails(e){
  selectedEdgeKey=edgeKey(e); selectedNodeId=null;
  const source=findNode(e.source), target=findNode(e.target);
  renderDetails('有向边', `${edgeLabel(e)} · ${String(target?.title || label(target||{}))}`, {
    source:{id:e.source,title:source&&String(source.title || label(source))}, target:{id:e.target,title:target&&String(target.title || label(target))},
    relation:e.relation, edge_type:e.edge_type, direction:'source → target', reason:e.reason,
    shared_factors:e.shared_factors||[], target_change:e.target_change||changeOf(target),
  });
  draw();
}
function init(){
  const iterations=new Set(['all']); [...(method.nodes||[]),...(memory.nodes||[])].forEach(n=>{if(n.iteration!=null)iterations.add(String(n.iteration));});
  $('iteration').innerHTML=[...iterations].map(x=>`<option value="${x}">${x==='all'?'全部':('第 '+x+' 轮')}</option>`).join('');
  $('iteration').onchange=e=>{round=e.target.value;selectedNodeId=null;selectedEdgeKey=null;draw();};
  $('canvas').addEventListener('pointermove',moveDrag); $('canvas').addEventListener('pointerup',endDrag); $('canvas').addEventListener('pointercancel',endDrag);
  $('canvas').addEventListener('contextmenu',event=>event.preventDefault());
  $('canvas').addEventListener('click',event=>{if(event.target===$('canvas')){selectedNodeId=null;selectedEdgeKey=null;renderDetails('图','未选择','点击节点或有向边查看详情');draw();}});
  $('methodBtn').onclick=()=>{view='method';selectedNodeId=null;selectedEdgeKey=null;draw();};
  $('memoryBtn').onclick=()=>{view='memory';selectedNodeId=null;selectedEdgeKey=null;draw();};
  $('allBtn').onclick=()=>{round='all';$('iteration').value='all';selectedNodeId=null;selectedEdgeKey=null;draw();};
  const p=DATA.project||{}; const s=DATA.session||{}; $('status').textContent=`${s.status||''} · 当前版本 ${p.incumbent_variant_id||'baseline'}`; draw();
}
init();
</script>
</body></html>'''
