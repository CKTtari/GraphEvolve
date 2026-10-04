"""Offline HTML monitor for one MethodTrail project/session.

The page deliberately has no CDN dependency.  It can be opened directly from
the filesystem while a run is paused or after it finishes.
"""

from __future__ import annotations

import json
from html import escape
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any


def write_dashboard(target: str | Path, payload: dict[str, Any]) -> Path:
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    project_id = escape(str(payload.get("project", {}).get("project_id", "local")))
    try:
        release = distribution_version("graphevolve")
    except PackageNotFoundError:
        release = "0.1.0"
    brand = escape(f"GraphEvolve v{release}")
    document_title = escape(f"GraphEvolve v{release} · {project_id}")
    target_path.write_text(
        _HTML_TEMPLATE.replace("__BRAND__", brand)
        .replace("__PROJECT__", project_id)
        .replace("__DOCUMENT_TITLE__", document_title)
        .replace("__DATA__", data),
        encoding="utf-8",
    )
    return target_path


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__DOCUMENT_TITLE__</title>
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
.project-name { color:var(--muted); font-size:12px; max-width:420px; overflow-wrap:anywhere; }
#canvas { width:100%; height:min(76vh,820px); min-height:650px; background:#fff; border-radius:6px; border:1px solid #d9dee7; touch-action:none; cursor:grab; }
#canvas:active { cursor:grabbing; }
.zoom-controls { display:inline-flex; gap:4px; margin-left:auto; }
.zoom-controls button { min-width:32px; padding:5px 8px; }
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
.chart-panel { margin-top:10px; border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }
.chart-panel h3 { margin:0 0 4px; font-size:15px; }
.chart-note { color:var(--muted); font-size:12px; margin-bottom:6px; }
#scoreChart svg { width:100%; height:250px; display:block; }
.score-grid { stroke:#edf0f4; stroke-width:1; }
.score-axis { stroke:#aeb7c5; stroke-width:1; }
.score-line { fill:none; stroke:var(--accent); stroke-width:2.5; }
.score-dot { fill:#fff; stroke:var(--accent); stroke-width:2; cursor:pointer; }
.score-dot.selected { fill:var(--warn); stroke:var(--warn); stroke-width:3; }
.score-label { fill:var(--muted); font-size:10px; text-anchor:middle; }
.score-value { fill:var(--text); font-size:10px; text-anchor:middle; }
.score-y-label { fill:var(--muted); font-size:10px; text-anchor:end; }
.candidate-summary { margin-top:8px; color:var(--muted); font-size:12px; }
.candidate-table-wrap { overflow:auto; }
.candidate-table { width:100%; border-collapse:collapse; font-size:12px; }
.candidate-table th, .candidate-table td { text-align:left; padding:6px 7px; border-bottom:1px solid #edf0f4; vertical-align:top; }
.candidate-table th { color:var(--muted); font-weight:600; white-space:nowrap; }
.candidate-table .candidate-selected { background:#fff8e7; }
.candidate-table .candidate-history { color:#596579; }
.candidate-table .candidate-status { white-space:nowrap; }
.candidate-table .candidate-title { min-width:220px; overflow-wrap:anywhere; }
.candidate-table .priority { font-variant-numeric:tabular-nums; white-space:nowrap; }
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
<header><h1>__BRAND__</h1><span class="project-name">__PROJECT__</span><label>轮次 <select id="iteration"></select></label><button id="methodBtn">方法图</button><button id="memoryBtn">经验图</button><button id="allBtn">全部</button><span id="status"></span></header>
<div class="layout"><main class="panel"><div class="toolbar"><span id="viewName">方法图</span><span>左键点击选择；按住左键拖动节点；空白处拖动画布；滚轮缩放。</span><span id="relationHint"></span><span class="zoom-controls"><button id="zoomOut" title="缩小">−</button><button id="zoomIn" title="放大">＋</button><button id="resetView">重置视图</button></span></div><svg id="canvas" viewBox="0 0 1200 700" preserveAspectRatio="xMidYMid meet"></svg><section id="scoreChart" class="chart-panel"></section><section id="candidatePanel" class="chart-panel"></section><div id="summary" class="cards"></div></main><aside class="panel"><h2>节点或边详情</h2><div id="details">选择一个节点或有向边</div></aside></div>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
let view = 'method', round = 'all';
const positionStore = new Map();
const transformStore = new Map();
const VIEW_WIDTH = 1200, VIEW_HEIGHT = 700;
const WORLD_WIDTH = 1800, WORLD_HEIGHT = 1100;
let activePositions = null, activeGraphKey = '', activeTransform = null;
let dragging = null, panDragging = null, suppressNextClick = false;
let selectedNodeId = null, selectedEdgeKey = null;
// Layout v2 deliberately ignores coordinates saved by the old fixed 1000×570 canvas.
const positionStoragePrefix = `methodtrail-pos-v2|${String(DATA.project?.project_id || '')}|${String(DATA.session?.session_id || '')}`;
const method = DATA.method_graph || {nodes:[], edges:[]};
const memory = DATA.memory_graph || {nodes:[], edges:[]};
function label(n){ return String(n.title || n.conclusion || n.method_family || n.card_id || n.variant_id || n.node_id || '').slice(0,25); }
function nodeIds(n){ return [n.card_id,n.id,n.node_id,n.variant_id].filter(x=>x!=null).map(String); }
function nodeId(n){ return nodeIds(n)[0]; }
function groupKey(n){ return String(n.method_family || n.method?.family || n.mutation_class || n.relation || 'other'); }
// Node colors are recomputed for the currently visible round.  They encode a
// cheap distance projection, not a stored graph relation: method metadata is
// tokenized, pairwise weighted-Jaccard distances are measured, and distances
// to three deterministic landmark nodes become R/G/B channels.  This keeps
// colors stable within a view while allowing a newly completed round to update
// the whole visible graph without changing any graph edge or node metadata.
let activeNodeColors = new Map(), activeColorKey = '';
function colorTokens(n){
  const values=new Map();
  const add=(value,weight=1)=>{
    if(value==null)return;
    const text=String(value).toLowerCase().replace(/[^\p{L}\p{N}]+/gu,' ');
    text.split(/\s+/).filter(Boolean).forEach(token=>values.set(token,(values.get(token)||0)+weight));
  };
  add(n.method_family || n.method?.family,3);
  add(n.mutation_class,2); add(n.relation,1);
  const factors=n.changed_factors || n.method?.changed_factors || [];
  (Array.isArray(factors)?factors:[factors]).forEach(value=>add(value,2));
  const components=n.method_components || n.method?.components || {};
  Object.entries(components||{}).forEach(([key,value])=>{add(key,1);add(value,2);});
  return values;
}
function colorDistance(a,b){
  const left=colorTokens(a), right=colorTokens(b);
  let intersection=0, union=0;
  new Set([...left.keys(),...right.keys()]).forEach(token=>{
    const l=left.get(token)||0, r=right.get(token)||0;
    intersection+=Math.min(l,r); union+=Math.max(l,r);
  });
  return union?1-intersection/union:1;
}
function normalizeColorChannel(values){
  const low=Math.min(...values), high=Math.max(...values);
  if(high-low<1e-9)return values.map(()=>0.72);
  return values.map(value=>0.38+0.62*(value-low)/(high-low));
}
function computeNodeColors(nodes,key){
  if(activeColorKey===key)return;
  activeColorKey=key; activeNodeColors=new Map();
  if(!nodes.length)return;
  const ordered=[...nodes].sort((a,b)=>nodeId(a).localeCompare(nodeId(b)));
  const anchorCount=Math.min(3,ordered.length), anchors=[ordered[0]];
  while(anchors.length<anchorCount){
    let best=null,bestDistance=-1;
    ordered.forEach(candidate=>{
      if(anchors.includes(candidate))return;
      const distance=Math.min(...anchors.map(anchor=>colorDistance(candidate,anchor)));
      if(distance>bestDistance || (Math.abs(distance-bestDistance)<1e-9 && nodeId(candidate)<nodeId(best))){best=candidate;bestDistance=distance;}
    });
    if(best)anchors.push(best);else break;
  }
  const channels=nodes.map(node=>anchors.map(anchor=>1-colorDistance(node,anchor)));
  const normalized=[0,1,2].map(channel=>normalizeColorChannel(channels.map(row=>row[channel] ?? 0.72)));
  nodes.forEach((node,index)=>{
    const rgb=normalized.map(channel=>Math.round(255*channel[index]));
    activeNodeColors.set(nodeId(node),`rgb(${rgb[0]} ${rgb[1]} ${rgb[2]})`);
  });
}
function nodeColor(n){return activeNodeColors.get(nodeId(n)) || 'rgb(210 220 235)';}
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
function restoreTransform(key, positions){
  try{
    const raw=JSON.parse(localStorage.getItem(positionStoragePrefix+'|transform|'+key) || 'null');
    if(raw && Number.isFinite(raw.x) && Number.isFinite(raw.y) && Number.isFinite(raw.scale) && raw.scale>0){
      return {x:raw.x,y:raw.y,scale:raw.scale};
    }
  }catch(_){ }
  if(!positions || !positions.size)return {x:0,y:0,scale:0.8};
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  positions.forEach(p=>{minX=Math.min(minX,p.x);minY=Math.min(minY,p.y);maxX=Math.max(maxX,p.x);maxY=Math.max(maxY,p.y);});
  const width=Math.max(1,maxX-minX+150), height=Math.max(1,maxY-minY+150);
  const scale=Math.max(0.42,Math.min(1,(VIEW_WIDTH-70)/width,(VIEW_HEIGHT-70)/height));
  return {x:(VIEW_WIDTH-(minX+maxX)*scale)/2,y:(VIEW_HEIGHT-(minY+maxY)*scale)/2,scale};
}
function persistTransform(){
  if(!activeGraphKey || !activeTransform)return;
  try{ localStorage.setItem(positionStoragePrefix+'|transform|'+activeGraphKey, JSON.stringify(activeTransform)); }catch(_){ }
}
function applyTransform(){
  const layer=$('graphLayer');
  if(layer && activeTransform)layer.setAttribute('transform',`translate(${activeTransform.x} ${activeTransform.y}) scale(${activeTransform.scale})`);
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
  nodes = nodes.filter(n => round === 'all' || (Number(n.iteration) || 0) <= Number(round));
  if (round !== 'all' && !nodes.length) nodes = (g.nodes || []).filter(n => n.node_type !== 'root' && n.variant_id !== '__project_root__' && n.id !== '__project_root__' && n.node_id !== '__project_root__');
  const ids = new Set(nodes.flatMap(nodeIds));
  let edges = (g.edges || []).filter(e => ids.has(String(e.source)) && ids.has(String(e.target)));
  return {nodes, edges};
}
function metricSeries(){
  const byRound=new Map();
  (DATA.events||[]).forEach(event=>{
    if(event.kind!=='iteration_finished')return;
    const payload=event.payload||{}, iteration=Number(payload.iteration), metric=Number(payload.metric);
    if(payload.metric==null || payload.metric==='' || !Number.isFinite(iteration) || !Number.isFinite(metric) || payload.completed_research===false)return;
    byRound.set(iteration,{iteration,metric,decision:payload.decision||'',variant_id:payload.variant_id||''});
  });
  return [...byRound.values()].sort((a,b)=>a.iteration-b.iteration);
}
function prioritySeries(){
  const items=(DATA.artifacts||[]).filter(item=>item.kind==='candidate_priorities' && item.payload?.ranked);
  items.sort((a,b)=>String(a.created_at||'').localeCompare(String(b.created_at||'')));
  return items.map((item,index)=>({
    iteration:Number(item.payload.iteration)||index+1,
    parent_variant_id:item.payload.parent_variant_id||'',
    ranked:item.payload.ranked||[],
  }));
}
function svgNode(name, attrs={}){
  const node=document.createElementNS('http://www.w3.org/2000/svg',name);
  Object.entries(attrs).forEach(([key,value])=>node.setAttribute(key,String(value)));
  return node;
}
function formatMetric(value){
  return Number(value).toLocaleString(undefined,{maximumFractionDigits:6});
}
function renderScoreChart(){
  const box=$('scoreChart'); box.replaceChildren();
  const series=metricSeries();
  const heading=document.createElement('h3'); heading.textContent='每轮主指标变化'; box.appendChild(heading);
  if(!series.length){
    const empty=document.createElement('div'); empty.className='chart-note'; empty.textContent='暂无已完成研究轮次的独立评测指标'; box.appendChild(empty); return;
  }
  const values=series.map(item=>item.metric), minValue=Math.min(...values), maxValue=Math.max(...values);
  const span=Math.max(maxValue-minValue,Math.abs(maxValue)*0.04,0.000001), low=minValue-span*0.12, high=maxValue+span*0.12;
  const width=960, height=240, left=62, right=24, top=24, bottom=42, plotWidth=width-left-right, plotHeight=height-top-bottom;
  const svg=svgNode('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':'每轮主指标折线图'});
  for(let index=0;index<=4;index++){
    const value=high-(high-low)*index/4, y=top+plotHeight*index/4;
    svg.appendChild(svgNode('line',{x1:left,y1:y,x2:width-right,y2:y,class:'score-grid'}));
    const text=svgNode('text',{x:left-8,y:y+3,class:'score-y-label'}); text.textContent=formatMetric(value); svg.appendChild(text);
  }
  svg.appendChild(svgNode('line',{x1:left,y1:top+plotHeight,x2:width-right,y2:top+plotHeight,class:'score-axis'}));
  const points=series.map((item,index)=>{
    const x=left+plotWidth*(series.length===1?0.5:index/(series.length-1));
    const y=top+plotHeight*(high-item.metric)/(high-low);
    return {item,x,y};
  });
  svg.appendChild(svgNode('polyline',{points:points.map(point=>`${point.x},${point.y}`).join(' '),class:'score-line'}));
  points.forEach(({item,x,y})=>{
    const dot=svgNode('circle',{cx:x,cy:y,r:4,class:`score-dot${round===String(item.iteration)?' selected':''}`});
    const title=svgNode('title'); title.textContent=`第${item.iteration}轮 · ${formatMetric(item.metric)} · ${item.decision||'完成'}`; dot.appendChild(title); svg.appendChild(dot);
    const value=svgNode('text',{x,y:y-10,class:'score-value'}); value.textContent=formatMetric(item.metric); svg.appendChild(value);
    const label=svgNode('text',{x,y:height-19,class:'score-label'}); label.textContent=`第${item.iteration}轮`; svg.appendChild(label);
  });
  box.appendChild(svg);
  const note=document.createElement('div'); note.className='chart-note';
  note.textContent=`${series.length} 个已完成研究轮次 · 最新 ${formatMetric(series.at(-1).metric)} · 最小 ${formatMetric(minValue)} · 最大 ${formatMetric(maxValue)}；折线按研究轮次连接，具体指标方向由任务合同决定`;
  box.appendChild(note);
}
function renderCandidatePanel(){
  const box=$('candidatePanel'); box.replaceChildren();
  const heading=document.createElement('h3'); heading.textContent='每轮候选方法排序'; box.appendChild(heading);
  const all=prioritySeries();
  let selected=round==='all'?all.at(-1):all.filter(item=>String(item.iteration)===String(round)).at(-1);
  if(!selected && round!=='all')selected=all.at(-1);
  if(!selected){
    const empty=document.createElement('div'); empty.className='chart-note'; empty.textContent='暂无候选排序记录'; box.appendChild(empty); return;
  }
  const rows=selected.ranked, methodNodes=method.nodes||[];
  const source=document.createElement('div'); source.className='candidate-summary';
  const newCount=rows.filter(row=>{const node=methodNodes.find(item=>nodeIds(item).includes(String(row.candidate?.variant_id))); return node && Number(node.iteration)===Number(selected.iteration);}).length;
  const oldCount=rows.length-newCount;
  source.textContent=`第${selected.iteration}轮实际排序 ${rows.length} 个候选：本轮新增 ${newCount} 个，历史候选/回溯分支 ${oldCount} 个。${rows.length<4?'当前少于4个可比较候选。':'已达到至少4个候选的比较要求。'}${round==='all'?'（当前显示最近一轮；选择轮次可查看对应批次）':''}`;
  box.appendChild(source);
  const wrap=document.createElement('div'); wrap.className='candidate-table-wrap';
  const table=document.createElement('table'); table.className='candidate-table';
  const head=document.createElement('thead'); head.innerHTML='<tr><th>排序</th><th>候选方法</th><th>关系</th><th>优先级</th><th>状态</th></tr>'; table.appendChild(head);
  const body=document.createElement('tbody');
  const selectedVariant=metricSeries().find(item=>item.iteration===selected.iteration)?.variant_id;
  rows.forEach((row,index)=>{
    const candidate=row.candidate||{}, node=methodNodes.find(item=>nodeIds(item).includes(String(candidate.variant_id))), originalIteration=node?.iteration;
    const tr=document.createElement('tr');
    if(String(candidate.variant_id)===String(selectedVariant))tr.className='candidate-selected';
    const rank=document.createElement('td'); rank.textContent=String(index+1); tr.appendChild(rank);
    const title=document.createElement('td'); title.className='candidate-title'; title.textContent=String(candidate.title||node?.title||candidate.method?.family||'未命名方法');
    if(originalIteration!=null && Number(originalIteration)!==Number(selected.iteration))title.classList.add('candidate-history'); tr.appendChild(title);
    const relation=document.createElement('td'); relation.textContent=String(candidate.relation||''); tr.appendChild(relation);
    const priority=document.createElement('td'); priority.className='priority'; priority.textContent=Number.isFinite(Number(row.priority))?Number(row.priority).toFixed(4):'不可用'; tr.appendChild(priority);
    const status=document.createElement('td'); status.className='candidate-status'; status.textContent=row.graph_signal?.feasible===false?'预算不可行':(String(candidate.variant_id)===String(selectedVariant)?'本轮已执行':(originalIteration!=null && Number(originalIteration)!==Number(selected.iteration)?'历史候选':'待选')); tr.appendChild(status);
    body.appendChild(tr);
  });
  table.appendChild(body); wrap.appendChild(table); box.appendChild(wrap);
}
function layout(nodes, edges){
  const positions=new Map(), points=new Map(), groups=[...new Set(nodes.map(groupKey))];
  const cols=Math.max(1,Math.ceil(Math.sqrt(groups.length))), rows=Math.max(1,Math.ceil(groups.length/cols));
  const centers=new Map(); groups.forEach((g,i)=>{ const col=i%cols, row=Math.floor(i/cols); centers.set(g,{x:220+(col+0.5)*((WORLD_WIDTH-440)/cols),y:170+(row+0.5)*((WORLD_HEIGHT-340)/rows)}); });
  nodes.forEach((n,i)=>{ const c=centers.get(groupKey(n)); const angle=(i*2.399)-Math.PI/2; const radius=82+(i%5)*28; points.set(nodeId(n),{x:c.x+Math.cos(angle)*radius,y:c.y+Math.sin(angle)*radius}); });
  const lookup=new Map(); nodes.forEach(n=>nodeIds(n).forEach(id=>lookup.set(id,nodeId(n))));
  for(let step=0;step<140;step++){
    const force=new Map(nodes.map(n=>[nodeId(n),{x:0,y:0}]));
    for(let i=0;i<nodes.length;i++) for(let j=i+1;j<nodes.length;j++){
      const a=points.get(nodeId(nodes[i])), b=points.get(nodeId(nodes[j])); let dx=a.x-b.x, dy=a.y-b.y, d=Math.max(18,Math.hypot(dx,dy));
      const push=12000/(d*d), fx=push*dx/d, fy=push*dy/d; force.get(nodeId(nodes[i])).x+=fx; force.get(nodeId(nodes[i])).y+=fy; force.get(nodeId(nodes[j])).x-=fx; force.get(nodeId(nodes[j])).y-=fy;
    }
    edges.forEach(e=>{ const sa=lookup.get(String(e.source)), sb=lookup.get(String(e.target)); if(!sa||!sb||sa===sb)return; const a=points.get(sa),b=points.get(sb); let dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)); const pull=(d-220)*0.009,fx=pull*dx/d,fy=pull*dy/d; force.get(sa).x+=fx;force.get(sa).y+=fy;force.get(sb).x-=fx;force.get(sb).y-=fy; });
    nodes.forEach(n=>{ const id=nodeId(n), p=points.get(id), c=centers.get(groupKey(n)), f=force.get(id); f.x+=(c.x-p.x)*0.035; f.y+=(c.y-p.y)*0.035; p.x=Math.max(90,Math.min(WORLD_WIDTH-90,p.x+f.x*0.55)); p.y=Math.max(90,Math.min(WORLD_HEIGHT-90,p.y+f.y*0.55)); });
  }
  // A deterministic overlap resolver guarantees a readable initial state even
  // when the force layout converges with two nodes in the same small pocket.
  const minimum=104;
  for(let pass=0;pass<90;pass++){
    let moved=false;
    for(let i=0;i<nodes.length;i++) for(let j=i+1;j<nodes.length;j++){
      const a=points.get(nodeId(nodes[i])), b=points.get(nodeId(nodes[j]));
      let dx=a.x-b.x,dy=a.y-b.y,d=Math.hypot(dx,dy);
      if(d>=minimum)continue;
      if(d<0.001){ const angle=((i*37+j*17)%360)*Math.PI/180; dx=Math.cos(angle);dy=Math.sin(angle);d=1; }
      const shift=(minimum-d)/2, sx=dx/d*shift, sy=dy/d*shift;
      a.x=Math.max(90,Math.min(WORLD_WIDTH-90,a.x+sx)); a.y=Math.max(90,Math.min(WORLD_HEIGHT-90,a.y+sy));
      b.x=Math.max(90,Math.min(WORLD_WIDTH-90,b.x-sx)); b.y=Math.max(90,Math.min(WORLD_HEIGHT-90,b.y-sy)); moved=true;
    }
    if(!moved)break;
  }
  nodes.forEach(n=>nodeIds(n).forEach(id=>positions.set(id,points.get(nodeId(n)))));
  return positions;
}
function svgPoint(event){
  const svg=$('canvas'), point=svg.createSVGPoint();
  point.x=event.clientX; point.y=event.clientY;
  return point.matrixTransform(svg.getScreenCTM().inverse());
}
function worldPoint(event){
  const point=svgPoint(event), transform=activeTransform||{x:0,y:0,scale:1};
  return {x:(point.x-transform.x)/transform.scale,y:(point.y-transform.y)/transform.scale};
}
function startDrag(event, node){
  if(event.button!=null && event.button!==0)return;
  event.stopPropagation();
  const ids=nodeIds(node), id=nodeId(node), point=worldPoint(event), old=activePositions.get(id);
  dragging={ids, id, node, startX:point.x, startY:point.y, offsetX:old.x-point.x, offsetY:old.y-point.y, pointerId:event.pointerId, captured:false};
  showDetails(node, false);
}
function moveDrag(event){
  if(!dragging || !activePositions)return;
  const point=worldPoint(event), p={x:point.x+dragging.offsetX,y:point.y+dragging.offsetY};
  if(!dragging.captured && Math.hypot(point.x-dragging.startX,point.y-dragging.startY)<=8)return;
  if(!dragging.captured){
    dragging.captured=true;
    try{$('canvas').setPointerCapture(dragging.pointerId);}catch(_){ }
  }
  p.x=Math.max(90,Math.min(WORLD_WIDTH-90,p.x)); p.y=Math.max(90,Math.min(WORLD_HEIGHT-90,p.y));
  const old=activePositions.get(dragging.id); if(Math.hypot(p.x-old.x,p.y-old.y)<1)return;
  dragging.ids.forEach(id=>activePositions.set(id,p)); persistPositions(); draw(true);
}
function startPan(event){
  if(event.button!=null && event.button!==0)return;
  if(event.target!==$('canvas'))return;
  event.preventDefault();
  const point=svgPoint(event);
  panDragging={startX:point.x,startY:point.y,x:activeTransform?.x||0,y:activeTransform?.y||0,pointerId:event.pointerId,moved:false,captured:false};
  try{$('canvas').setPointerCapture(event.pointerId);panDragging.captured=true;}catch(_){ }
}
function movePan(event){
  if(!panDragging || !activeTransform)return;
  const point=svgPoint(event), dx=point.x-panDragging.startX, dy=point.y-panDragging.startY;
  if(Math.hypot(dx,dy)>3)panDragging.moved=true;
  activeTransform.x=panDragging.x+dx; activeTransform.y=panDragging.y+dy;
  applyTransform(); persistTransform();
}
function endDrag(event){
  if(!dragging)return;
  const finished=dragging;
  if(finished.captured){try{$('canvas').releasePointerCapture(finished.pointerId);}catch(_){ }}
  dragging=null;
  suppressNextClick=true;
  if(!finished.captured && finished.node)showDetails(finished.node);
  setTimeout(()=>{suppressNextClick=false;},0);
}
function endPan(event){
  if(!panDragging)return;
  if(panDragging.captured){try{$('canvas').releasePointerCapture(panDragging.pointerId);}catch(_){ }}
  const moved=panDragging.moved;
  panDragging=null;
  if(moved)suppressNextClick=true;
  if(moved)setTimeout(()=>{suppressNextClick=false;},0);
}
function zoomAt(event,factor){
  if(!activeTransform)return;
  const before=svgPoint(event), oldScale=activeTransform.scale;
  const next=Math.max(0.35,Math.min(3.5,oldScale*factor));
  if(Math.abs(next-oldScale)<0.0001)return;
  const worldX=(before.x-activeTransform.x)/oldScale, worldY=(before.y-activeTransform.y)/oldScale;
  activeTransform.scale=next;
  activeTransform.x=before.x-worldX*next; activeTransform.y=before.y-worldY*next;
  applyTransform(); persistTransform();
}
function resetView(){
  // Explicit reset ignores the saved view and fits all current nodes.
  activeTransform=activePositions?.size?(()=>{
    let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
    activePositions.forEach(p=>{minX=Math.min(minX,p.x);minY=Math.min(minY,p.y);maxX=Math.max(maxX,p.x);maxY=Math.max(maxY,p.y);});
    const width=Math.max(1,maxX-minX+150),height=Math.max(1,maxY-minY+150);
    const scale=Math.max(0.42,Math.min(1,(VIEW_WIDTH-70)/width,(VIEW_HEIGHT-70)/height));
    return {x:(VIEW_WIDTH-(minX+maxX)*scale)/2,y:(VIEW_HEIGHT-(minY+maxY)*scale)/2,scale};
  })():{x:0,y:0,scale:0.8};
  transformStore.set(activeGraphKey,activeTransform);
  persistTransform(); applyTransform();
}
function draw(){
  const svg = $('canvas'); while(svg.firstChild) svg.removeChild(svg.firstChild);
  const g = visibleGraph(), nodes=g.nodes, edges=g.edges;
  $('viewName').textContent = view === 'memory' ? '经验图' : '方法图';
  $('relationHint').textContent = view === 'memory'
    ? '经验图：箭头为前序证据 → 后续实验；follows=父版本，informed by=决策依据，same family/shared factor=相关证据；颜色按当前轮相似度投影'
    : '方法图：箭头只表示方法关系；轮次和修改逻辑在节点详情中查看；颜色按当前轮相似度投影';
  const graphKey=view+'|'+round;
  let positions=positionStore.get(graphKey);
  if(!positions){ positions=restorePositions(graphKey,layout(nodes, edges)); positionStore.set(graphKey,positions); }
  activeGraphKey=graphKey; activePositions=positions;
  if(!transformStore.has(graphKey))transformStore.set(graphKey,restoreTransform(graphKey,positions));
  activeTransform=transformStore.get(graphKey);
  computeNodeColors(nodes,graphKey);
  const lookup=new Map(); nodes.forEach(n=>nodeIds(n).forEach(id=>lookup.set(id,n)));
  const defs=document.createElementNS('http://www.w3.org/2000/svg','defs');
  defs.innerHTML='<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L10,5 L0,10 Z" fill="#7b8798"/></marker>';
  svg.appendChild(defs);
  const layer=document.createElementNS('http://www.w3.org/2000/svg','g');
  layer.setAttribute('id','graphLayer');
  svg.appendChild(layer);
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
    group.appendChild(hit); group.appendChild(line); group.appendChild(text); layer.appendChild(group);
  });
  nodes.forEach(n=>{
    const p=positions.get(nodeId(n)); if(!p)return;
    const group=document.createElementNS('http://www.w3.org/2000/svg','g');
    const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',34);c.setAttribute('fill',nodeColor(n));
    c.setAttribute('class','node '+(n.node_type||'')+(n.origin?' origin':'')+(n.status==='failed'?' failed':'')+(selectedNodeId===nodeId(n)?' selected':''));
    group.onpointerdown=e=>startDrag(e,n);
    group.onclick=e=>{e.stopPropagation();if(!suppressNextClick)showDetails(n);};
    group.appendChild(c);
    const title=document.createElementNS('http://www.w3.org/2000/svg','title');
    title.textContent=`${n.iteration==null?'':`第${n.iteration}轮 · `}${String(n.title||n.conclusion||n.method_family||'')}`;
    group.appendChild(title);
    const t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',p.x);t.setAttribute('y',p.y+4);t.setAttribute('class','node-text');t.textContent=label(n);group.appendChild(t);layer.appendChild(group);
    if(n.iteration!=null){
      const r=document.createElementNS('http://www.w3.org/2000/svg','text');r.setAttribute('x',p.x);r.setAttribute('y',p.y+20);r.setAttribute('class','node-round');r.textContent=`${n.origin?'起点 · ':''}第${n.iteration}轮`;group.appendChild(r);
    }
  });
  applyTransform();
  const events=(DATA.events||[]).filter(e=>['iteration_finished','technical_attempt','research_stop'].includes(e.kind));
  const shown=round==='all'?events:events.filter(e=>String(e.payload?.iteration||'')===String(round));
  const eventHtml=shown.slice(-6).map(e=>{const p=e.payload||{}; const bad=e.kind==='technical_attempt'; return `<div class="event ${bad?'bad':'good'}"><b>${e.kind}</b> · 第${p.iteration||'?'}轮 · ${p.decision||p.reason||''}<br>指标：${p.metric==null?'未测得':p.metric} ${p.next_question?'<br>下一问题：'+p.next_question:''}</div>`;}).join('');
  const groups=[...new Set(nodes.map(groupKey))]; const legend=groups.map(g=>`<span>${g}</span>`).join(' · ');
  $('summary').innerHTML = `<div class="card"><b>节点</b><br>${nodes.length}</div><div class="card"><b>有向关系</b><br>${edges.length}</div><div class="card"><b>方法族</b><br>${groups.length}<br>${legend}</div>${eventHtml}`;
  renderScoreChart();
  renderCandidatePanel();
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
function showDetails(n, redraw=true){
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
  if(redraw)draw();
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
  $('iteration').innerHTML=[...iterations].map(x=>`<option value="${x}">${x==='all'?'全部':('截至第 '+x+' 轮')}</option>`).join('');
  $('iteration').onchange=e=>{round=e.target.value;selectedNodeId=null;selectedEdgeKey=null;draw();};
  $('canvas').addEventListener('pointerdown',startPan);
  $('canvas').addEventListener('pointermove',event=>{if(dragging)moveDrag(event);else movePan(event);});
  $('canvas').addEventListener('pointerup',event=>{endDrag(event);endPan(event);});
  $('canvas').addEventListener('pointercancel',event=>{endDrag(event);endPan(event);});
  $('canvas').addEventListener('wheel',event=>{event.preventDefault();zoomAt(event,event.deltaY<0?1.12:1/1.12);},{passive:false});
  $('canvas').addEventListener('contextmenu',event=>event.preventDefault());
  $('canvas').addEventListener('click',event=>{if(event.target===$('canvas')){selectedNodeId=null;selectedEdgeKey=null;renderDetails('图','未选择','点击节点或有向边查看详情');draw();}});
  $('methodBtn').onclick=()=>{view='method';selectedNodeId=null;selectedEdgeKey=null;draw();};
  $('memoryBtn').onclick=()=>{view='memory';selectedNodeId=null;selectedEdgeKey=null;draw();};
  $('allBtn').onclick=()=>{round='all';$('iteration').value='all';selectedNodeId=null;selectedEdgeKey=null;draw();};
  $('zoomIn').onclick=()=>{const rect=$('canvas').getBoundingClientRect();zoomAt({clientX:rect.left+rect.width/2,clientY:rect.top+rect.height/2},1.25);};
  $('zoomOut').onclick=()=>{const rect=$('canvas').getBoundingClientRect();zoomAt({clientX:rect.left+rect.width/2,clientY:rect.top+rect.height/2},0.8);};
  $('resetView').onclick=resetView;
  const p=DATA.project||{}; const s=DATA.session||{}; $('status').textContent=`${s.status||''} · 当前版本 ${p.incumbent_variant_id||'baseline'}`; draw();
}
init();
</script>
</body></html>'''
