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
#details { white-space:pre-wrap; overflow:auto; max-height:570px; color:var(--text); }
.metric { font-size:20px; color:var(--good); margin:4px 0 10px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px; margin-top:10px; }
.card { border:1px solid var(--line); border-radius:6px; padding:9px; background:#fff; }
.card b { color:var(--accent); }
.event { border-left:3px solid var(--accent); padding:6px 9px; margin-top:7px; background:#f8fafc; border-radius:4px; }
.event.good { border-left-color:var(--good); }
.event.bad { border-left-color:var(--bad); }
svg text { font-family:inherit; }
.edge { stroke:#9aa5b5; stroke-width:1.5; opacity:.88; marker-end:url(#arrow); cursor:pointer; }
.edge.lineage, .edge.follows { stroke:#3d9a70; }
.edge.evidence, .edge.same_family { stroke:#8b6fc4; stroke-dasharray:6 4; }
.edge.shared_factor, .edge.related { stroke:#c58a2b; stroke-dasharray:3 4; }
.edge-label { fill:#4b5565; font-size:10px; pointer-events:none; paint-order:stroke; stroke:#fff; stroke-width:4px; stroke-linejoin:round; }
.edge.related { stroke:#8b6fc4; stroke-dasharray:5 4; }
.edge.candidate { stroke:#356ae6; stroke-dasharray:3 4; }
.edge.execution { stroke:#3d9a70; }
.node { cursor:grab; stroke-width:2; }
.node:active { cursor:grabbing; }
.node.proposal { stroke:#356ae6; }
.node.outcome { stroke:#3d9a70; }
.node.failed { stroke:#c4475a; }
.node.selected { stroke:#b7791f; stroke-width:4; }
.node-text { fill:#202631; font-size:11px; pointer-events:none; text-anchor:middle; }
@media(max-width:900px){ .layout{grid-template-columns:1fr;} #details{max-height:none;} }
</style>
</head>
<body>
<header><h1>MethodTrail · __TITLE__</h1><label>轮次 <select id="iteration"></select></label><button id="methodBtn">方法图</button><button id="memoryBtn">经验图</button><button id="allBtn">全部</button><span id="status"></span></header>
<div class="layout"><main class="panel"><div class="toolbar"><span id="viewName">方法图</span><span>箭头表示方向；点击节点或边查看修改逻辑、内容和证据</span></div><svg id="canvas" viewBox="0 0 1000 570" preserveAspectRatio="xMidYMid meet"></svg><div id="summary" class="cards"></div></main><aside class="panel"><h2>节点或边详情</h2><div id="details">选择一个节点或有向边</div></aside></div>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
let view = 'method', round = 'all';
const positionStore = new Map();
let activePositions = null, activeGraphKey = '', dragging = null, lastDragMoved = false;
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
function edgeLabel(e){ const c=e.target_change||{}; const base=String(e.label||e.relation||e.edge_type||'relation').replaceAll('_',' '); const round=c.iteration==null?'':` · 第${c.iteration}轮`; const title=c.title?` · ${String(c.title).slice(0,18)}`:''; return base+round+title; }
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
  let nodes = (g.nodes || []).filter(n => round === 'all' || String(n.iteration || '') === String(round));
  if (round !== 'all' && !nodes.length) nodes = g.nodes || [];
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
  event.preventDefault(); event.stopPropagation();
  const ids=nodeIds(node), id=nodeId(node), point=svgPoint(event), old=activePositions.get(id);
  dragging={ids, id, offsetX:old.x-point.x, offsetY:old.y-point.y, pointerId:event.pointerId};
  lastDragMoved=false; $('canvas').setPointerCapture(event.pointerId);
}
function moveDrag(event){
  if(!dragging || !activePositions)return;
  const point=svgPoint(event), p={x:point.x+dragging.offsetX,y:point.y+dragging.offsetY};
  p.x=Math.max(30,Math.min(970,p.x)); p.y=Math.max(30,Math.min(620,p.y));
  const old=activePositions.get(dragging.id); if(Math.hypot(p.x-old.x,p.y-old.y)<1)return;
  dragging.ids.forEach(id=>activePositions.set(id,p)); persistPositions(); lastDragMoved=true; draw(true);
}
function endDrag(event){
  if(!dragging)return;
  try{$('canvas').releasePointerCapture(dragging.pointerId);}catch(_){ }
  dragging=null;
  if(lastDragMoved)setTimeout(()=>{lastDragMoved=false;},0);
}
function draw(){
  const svg = $('canvas'); while(svg.firstChild) svg.removeChild(svg.firstChild);
  const g = visibleGraph(), nodes=g.nodes, edges=g.edges;
  $('viewName').textContent = view === 'memory' ? '经验图' : '方法图';
  const graphKey=view+'|'+round;
  let positions=positionStore.get(graphKey);
  if(!positions){ positions=restorePositions(graphKey,layout(nodes, edges)); positionStore.set(graphKey,positions); }
  activeGraphKey=graphKey; activePositions=positions;
  const lookup=new Map(); nodes.forEach(n=>nodeIds(n).forEach(id=>lookup.set(id,n)));
  const defs=document.createElementNS('http://www.w3.org/2000/svg','defs'); defs.innerHTML='<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#9aa5b5"/></marker>'; svg.appendChild(defs);
  edges.forEach(e=>{ const a=positions.get(String(e.source)),b=positions.get(String(e.target)); if(!a||!b)return; const line=document.createElementNS('http://www.w3.org/2000/svg','line'); line.setAttribute('x1',a.x);line.setAttribute('y1',a.y);line.setAttribute('x2',b.x);line.setAttribute('y2',b.y);line.setAttribute('class','edge '+(e.edge_type||e.relation||'')); line.onclick=()=>showEdgeDetails(e); const title=document.createElementNS('http://www.w3.org/2000/svg','title'); title.textContent=edgeLabel(e)+' · '+(e.reason||''); line.appendChild(title); svg.appendChild(line); const text=document.createElementNS('http://www.w3.org/2000/svg','text'); text.setAttribute('x',(a.x+b.x)/2); text.setAttribute('y',(a.y+b.y)/2-5); text.setAttribute('class','edge-label'); text.textContent=edgeLabel(e); text.onclick=()=>showEdgeDetails(e); svg.appendChild(text); });
  nodes.forEach(n=>{ const p=positions.get(nodeId(n)); if(!p)return; const group=document.createElementNS('http://www.w3.org/2000/svg','g'); const c=document.createElementNS('http://www.w3.org/2000/svg','circle'); c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',34);c.setAttribute('fill',groupColor(groupKey(n)));c.setAttribute('class','node '+(n.node_type||'')+(n.status==='failed'?' failed':'')); group.onpointerdown=e=>startDrag(e,n); group.onclick=()=>{if(!lastDragMoved)showDetails(n);}; group.appendChild(c); const t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',p.x);t.setAttribute('y',p.y+4);t.setAttribute('class','node-text');t.textContent=label(n);group.appendChild(t);svg.appendChild(group); });
  const events=(DATA.events||[]).filter(e=>['iteration_finished','technical_attempt','research_stop'].includes(e.kind));
  const shown=round==='all'?events:events.filter(e=>String(e.payload?.iteration||'')===String(round));
  const eventHtml=shown.slice(-6).map(e=>{const p=e.payload||{}; const bad=e.kind==='technical_attempt'; return `<div class="event ${bad?'bad':'good'}"><b>${e.kind}</b> · 第${p.iteration||'?'}轮 · ${p.decision||p.reason||''}<br>指标：${p.metric==null?'未测得':p.metric} ${p.next_question?'<br>下一问题：'+p.next_question:''}</div>`;}).join('');
  const groups=[...new Set(nodes.map(groupKey))]; const legend=groups.map(g=>`<span style="color:${groupColor(g)}">● ${g}</span>`).join(' · ');
  $('summary').innerHTML = `<div class="card"><b>节点</b><br>${nodes.length}</div><div class="card"><b>有向关系</b><br>${edges.length}</div><div class="card"><b>方法族</b><br>${groups.length}<br>${legend}</div>${eventHtml}`;
}
function showDetails(n){ $('details').textContent=JSON.stringify(n,null,2); }
function showEdgeDetails(e){ const source=findNode(e.source), target=findNode(e.target); $('details').textContent=JSON.stringify({relation:e.relation,edge_type:e.edge_type,reason:e.reason,shared_factors:e.shared_factors||[],source:{id:e.source,title:source&&label(source)},target:{id:e.target,title:target&&label(target),change:changeOf(target)},target_change:e.target_change||null},null,2); }
function init(){
  const iterations=new Set(['all']); [...(method.nodes||[]),...(memory.nodes||[])].forEach(n=>{if(n.iteration!=null)iterations.add(String(n.iteration));});
  $('iteration').innerHTML=[...iterations].map(x=>`<option value="${x}">${x==='all'?'全部':('第 '+x+' 轮')}</option>`).join('');
  $('iteration').onchange=e=>{round=e.target.value;draw();};
  $('canvas').addEventListener('pointermove',moveDrag); $('canvas').addEventListener('pointerup',endDrag); $('canvas').addEventListener('pointercancel',endDrag);
  $('methodBtn').onclick=()=>{view='method';draw();}; $('memoryBtn').onclick=()=>{view='memory';draw();}; $('allBtn').onclick=()=>{round='all';$('iteration').value='all';draw();};
  const p=DATA.project||{}; const s=DATA.session||{}; $('status').textContent=`${s.status||''} · 当前版本 ${p.incumbent_variant_id||'baseline'}`; draw();
}
init();
</script>
</body></html>'''
