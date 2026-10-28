'use strict';
const $=id=>document.getElementById(id);
const fmt=n=>Number(n).toLocaleString('en-US');
const query=new URLSearchParams(location.search);
const state={mode:query.get('mode')||'lesion',site:Number(query.get('site')||20),size:Number(query.get('size')||2000),rule:query.get('rule')||'original',dose:Number(query.get('dose')||100),style:query.get('style')||'voxel',compare:false};
let DATA,current,epoch=0,firingEpoch=0,firingReady=false,sceneReady=false,playing=!matchMedia('(prefers-reduced-motion: reduce)').matches,time=0,lastFrame=0,referencePayload,selectedPayload,sceneSample,visible=true;
const cache=new Map();
const activityCache=new Map();
const orange='#d7652a',refColor='#77838a',selectedColor='#b23c69';
const metricColors={distance_m_norm:'#c76c36',beam_crossings_per_s_norm:'#8571a4',motor_output_norm:'#198b81',takeoff_output_norm:'#3c72a0',LPTC_L_L_norm:'#364a55'};
function color(){return state.mode==='lesion'?current.name==='intact'?refColor:orange:state.rule==='none'||state.dose===0?orange:DATA.rules.find(r=>r.id===state.rule).color;}
function ruleInfo(){return DATA.rules.find(r=>r.id===state.rule);}
function conditionName(){return state.mode==='lesion'?DATA.lesions[state.site][state.size]:state.rule==='none'?'lesion20_2000':DATA.repair[state.rule][state.dose];}
function storeSelection(){const q=new URLSearchParams();for(const k of ['mode','site','size','rule','dose','style'])q.set(k,state[k]);history.replaceState(null,'','?'+q);}
function validateState(){
  if(!['lesion','repair'].includes(state.mode))state.mode='lesion';
  if(![20,25].includes(state.site))state.site=20;
  if(!DATA.sizes[state.site].includes(state.size))state.size=state.site===25?1000:2000;
  if(!['none',...DATA.rules.map(r=>r.id)].includes(state.rule))state.rule='original';
  if(!DATA.doses.includes(state.dose))state.dose=100;
  if(!['original','voxel','journal'].includes(state.style))state.style='voxel';
}
function interval(m){return state.mode==='lesion'?[m.rangeLow,m.rangeHigh]:[m.ciLow,m.ciHigh];}
function optionList(values){return values.map(v=>`<option value="${v}">${fmt(v)}${v===0?' · '+(state.mode==='lesion'?'intact':'no replacements'):''}</option>`).join('');}
function setButtons(){
  document.querySelectorAll('[data-mode]').forEach(b=>{const active=b.dataset.mode===state.mode;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});
  document.querySelectorAll('[data-style]').forEach(b=>{const active=b.dataset.style===state.style;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});
}
function updateControls(){
  $('lesion-controls').hidden=state.mode!=='lesion';$('repair-controls').hidden=state.mode!=='repair';
  $('site').value=state.site;
  $('site-note').textContent=state.site===20?'A posterior central-brain interface carrying signals toward motor circuits.':'An optic-lobe comparison site. Its 1,000-cell lesion spares the two highlighted sensorimotor groups.';
  $('lesion-size').innerHTML=optionList(DATA.sizes[state.site]);$('lesion-size').value=state.size;
  $('lesion-slider').max=DATA.sizes[state.site].length-1;$('lesion-slider').value=DATA.sizes[state.site].indexOf(state.size);
  $('largest-lesion').textContent=fmt(DATA.sizes[state.site].at(-1));
  $('replacement-dose').innerHTML=optionList(DATA.doses);$('replacement-dose').value=state.rule==='none'?0:state.dose;
  $('dose-slider').value=state.rule==='none'?0:DATA.doses.indexOf(state.dose);
  $('dose-slider').disabled=$('replacement-dose').disabled=state.rule==='none';
  document.querySelectorAll('input[name=rule]').forEach(r=>r.checked=r.value===state.rule);
  $('repair-explanation').textContent=state.rule==='none'?'Untreated lesion. This is the shared starting point for every repair condition.':ruleInfo().description;
  $('anatomy').src=`anatomy/${current.name}.png`;
  $('anatomy-heading').textContent=state.mode==='repair'?'Lesion and intended replacement positions':'Lesion positions';
  $('graft-key').hidden=current.graft===0;
  $('graft-key').querySelector('i').style.background=color();
  $('anatomy').alt=`Brain and ventral nerve cord soma positions; ${fmt(current.lost)} lost neurons shown in pink${current.graft?`; ${fmt(current.graft)} intended replacement positions highlighted`:''}. Wiring is not depicted.`;
}
function summary(){
  const repair=state.mode==='repair',untreated=state.rule==='none'||state.dose===0;
  $('condition-context').textContent=repair?'Site 20 · 2,000-cell lesion':'Site '+state.site;
  $('condition-title').textContent=repair?(untreated?'No graft':ruleInfo().label):current.lost===0?'Intact circuit':`${fmt(current.lost)} neurons lost`;
  $('condition-description').textContent=repair?`${fmt(current.graft)} replacements after the same 2,000-cell lesion. Function is measured relative to matched intact.`:current.lost===0?'The matched intact reference defines function as one.':`${(100*current.lost/165122).toFixed(2)}% of the modeled nervous system removed. Readouts are relative to matched intact.`;
  const scale=Math.max(1.25,...DATA.metrics.map(({key})=>interval(current.metrics[key])[1]));
  $('readouts').innerHTML=DATA.metrics.map(({key,label,explanation})=>{
    const m=current.metrics[key],[lo,hi]=interval(m);const abs=m.absolute;
    const percent=v=>Math.max(0,Math.min(100,100*v/scale));
    return `<div class="readout" title="${explanation}"><div class="readout-title">${label}</div><div class="readout-value">${(m.mean*100).toFixed(1)}<span>%</span></div><div class="readout-track" style="--condition-color:${metricColors[key]}"><div class="readout-fill" style="width:${percent(m.mean)}%"></div><span class="readout-ref" style="left:${percent(1)}%" title="Intact = 100%"></span><span class="readout-interval" style="left:${percent(lo)}%;width:${Math.max(.1,percent(hi)-percent(lo))}%"></span></div><div class="interval-value">${(100*lo).toFixed(1)}–${(100*hi).toFixed(1)}%<br>${repair?'95% CI':'95% simulation range'}</div>${abs?`<div class="absolute-value">${abs.mean.toFixed(abs.unit==='Hz'?1:3)} ${abs.unit}</div>`:'<div class="absolute-value">Neural activity ratio</div>'}</div>`;
  }).join('');
  $('uncertainty-note').textContent=repair?'Mean and pointwise 95% confidence interval; n = 100. Movement averages 20 gusts per realization. The vertical mark is intact.':'Mean and central 95% simulation range; n = 100. Movement averages 20 gusts per realization. The vertical mark is intact.';
}
function svgEl(tag,attrs={},text=''){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;return e;}
function curveChart(metric){
  const repair=state.mode==='repair',values=repair?DATA.doses:DATA.sizes[state.site];
  const rules=repair&&state.compare?DATA.rules:repair?[ruleInfo()||{id:'none',label:'No graft',color:orange}]:[{id:'lesion',color:orange}];
  const series=rules.map(r=>({rule:r,points:(r.id==='none'?[0]:values).map(v=>({v,m:DATA.conditions[repair?(r.id==='none'?'lesion20_2000':DATA.repair[r.id][v]):DATA.lesions[state.site][v]].metrics[metric.key]}))}));
  const W=340,H=255,L=45,R=21,T=17,B=46,PW=W-L-R,PH=H-T-B;
  const maximum=Math.max(1.12,...series.flatMap(s=>s.points.map(p=>(repair?p.m.ciHigh:p.m.rangeHigh))));
  const ymax=Math.ceil(maximum*5)/5;
  const px=v=>L+(repair?v/2000:values.indexOf(v)/(values.length-1))*PW;
  const py=v=>T+(1-v/ymax)*PH;
  const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':`${metric.label} across tested ${repair?'replacement doses':'lesion sizes'}, normalized to matched intact.`});
  const plotId='clip-'+metric.key;
  const defs=svgEl('defs'),clip=svgEl('clipPath',{id:plotId});clip.append(svgEl('rect',{x:L,y:T,width:PW,height:PH}));defs.append(clip);svg.append(defs);
  [0,.5,1,...(ymax>=1.5?[1.5]:[]),...(ymax>=2?[2]:[])].filter(v=>v<=ymax).forEach(v=>{
    svg.append(svgEl('line',{x1:L,x2:W-R,y1:py(v),y2:py(v),stroke:v===1?'#95a5a0':'#e8ece9','stroke-dasharray':v===1?'3 3':'','stroke-width':v===1?1.1:.8}));
    svg.append(svgEl('text',{x:L-7,y:py(v)+4,'text-anchor':'end',fill:'#65787d','font-size':11},v.toFixed(1)));
  });
  svg.append(svgEl('line',{x1:L,x2:L,y1:T,y2:H-B,stroke:'#9baba7'}));svg.append(svgEl('line',{x1:L,x2:W-R,y1:H-B,y2:H-B,stroke:'#9baba7'}));
  const ticks=repair?[0,500,1000,1500,2000]:values.length>5?values.filter((v,i)=>i===0||i===2||i===4||i===6||i===8||i===values.length-1):values;
  ticks.forEach(v=>{svg.append(svgEl('line',{x1:px(v),x2:px(v),y1:H-B,y2:H-B+4,stroke:'#9baba7'}));svg.append(svgEl('text',{x:px(v),y:H-B+17,'text-anchor':'middle',fill:'#65787d','font-size':10},v>=1000?v/1000+'k':String(v)));});
  svg.append(svgEl('text',{x:L+PW/2,y:H-5,'text-anchor':'middle',fill:'#53666c','font-size':11},repair?'Replacement cells':'Neurons lost'));
  svg.append(svgEl('text',{x:11,y:T+PH/2,transform:`rotate(-90 11 ${T+PH/2})`,'text-anchor':'middle',fill:'#53666c','font-size':10},'Intact = 1'));
  const group=svgEl('g',{'clip-path':`url(#${plotId})`}),markers=svgEl('g');
  const selectedDose=repair?(state.rule==='none'?0:state.dose):state.size;
  group.append(svgEl('line',{x1:px(selectedDose),x2:px(selectedDose),y1:T,y2:H-B,stroke:selectedColor,'stroke-width':1.2,'stroke-dasharray':'3 4',opacity:.5,'data-selection-guide':'true'}));
  if(repair){const damage=DATA.conditions.lesion20_2000.metrics[metric.key].mean;group.append(svgEl('line',{x1:L,x2:W-R,y1:py(damage),y2:py(damage),stroke:orange,'stroke-dasharray':'5 4','stroke-width':1.2}));}
  series.forEach(({rule,points})=>{
    const ribbon=points.map(p=>`${px(p.v)},${py(repair?p.m.ciHigh:p.m.rangeHigh)}`).concat([...points].reverse().map(p=>`${px(p.v)},${py(repair?p.m.ciLow:p.m.rangeLow)}`)).join(' ');
    const c=repair?rule.color:metricColors[metric.key];
    group.append(svgEl('polygon',{points:ribbon,fill:c,opacity:state.compare?.08:.15}));
    group.append(svgEl('polyline',{points:points.map(p=>`${px(p.v)},${py(p.m.mean)}`).join(' '),stroke:c,fill:'none','stroke-width':2.35,'stroke-linejoin':'round','stroke-linecap':'round'}));
    points.forEach(({v,m})=>{
      const chosen=(repair?v===(state.rule==='none'?0:state.dose)&&(rule.id===state.rule||state.rule==='none'&&state.compare&&rule.id===DATA.rules[0].id):v===state.size);
      if(chosen)markers.append(svgEl('circle',{cx:px(v),cy:py(m.mean),r:10,fill:'white',stroke:selectedColor,'stroke-width':1.2,opacity:.95,'pointer-events':'none'}));
      const circle=svgEl('circle',{cx:px(v),cy:py(m.mean),r:chosen?6.8:2.8,fill:chosen?selectedColor:c,stroke:chosen?'white':c,'stroke-width':chosen?1.8:.6,'data-dose':v,'data-rule':rule.id,'data-selected':chosen,tabindex:0,role:'button','aria-label':`${repair?rule.label+' · ':''}${fmt(v)} cells: ${(100*m.mean).toFixed(1)}% intact`});
      circle.append(svgEl('title',{},`${repair?rule.label+' · ':''}${fmt(v)} cells: ${(100*m.mean).toFixed(1)}% intact`));
      circle.addEventListener('click',()=>{if(repair){state.rule=rule.id;state.dose=rule.id==='none'?0:v;}else state.size=v;update();});
      circle.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();circle.dispatchEvent(new MouseEvent('click'));}});
      markers.append(circle);
    });
  });
  svg.append(group,markers);return svg;
}
function curves(){
  const repair=state.mode==='repair';
  $('curve-title').textContent=repair?'Function across replacement doses':'Function across lesion sizes';
  $('curve-subtitle').textContent=repair?'Function remaining after damage, then each tested replacement dose.':'The same site, with increasing numbers of neurons removed.';
  $('selection-key').textContent='Selected: '+fmt(repair?(state.rule==='none'?0:state.dose):state.size)+(repair?' replacements':' lost');
  $('compare-rules').hidden=!repair;$('compare-rules').setAttribute('aria-pressed',state.compare);
  $('compare-rules').textContent=state.compare?'Show selected rule':'Compare repair rules';
  $('curve-key').innerHTML=repair?(state.compare?DATA.rules:[ruleInfo()||{label:'No graft',color:orange}]).map(r=>`<span><i style="--color:${r.color}"></i>${r.label}</span>`).join('')+'<span><i style="--color:#d7652a"></i>Damage baseline</span>':'<span>Mean and central 95% simulation range</span>';
  $('curves').replaceChildren();
  DATA.metrics.forEach(m=>{const box=document.createElement('div');box.className='curve';box.dataset.metric=m.key;const h=document.createElement('div');h.className='curve-heading';h.innerHTML=`<h3 style="color:${metricColors[m.key]}">${m.label}</h3><span class="curve-value">${(current.metrics[m.key].mean*100).toFixed(1)}<small>% intact</small></span>`;box.append(h,curveChart(m));$('curves').append(box);});
}
async function firingData(name){
  if(!activityCache.has(name))activityCache.set(name,fetch(`activity/${name}.json`).then(r=>{if(!r.ok)throw Error('Recording not available');return r.json();}));
  return activityCache.get(name);
}
function firingChart(d,label,c){
  const W=450,H=285,L=48,R=18,T=35,rasterTop=43,rasterHeight=65,traceTop=140,traceBottom=248,PW=W-L-R;
  const px=t=>L+t/300*PW,py=rate=>traceBottom-(rate/DATA.activity.ymaxHz)*(traceBottom-traceTop);
  const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':`${label}: ${d.meanMotorHz.toFixed(1)} Hz mean looming motor activity. Recorded spikes from 12 motor neurons and trailing 25-ms firing rate.`, 'data-condition':d.condition,'data-mean-hz':d.meanMotorHz});
  svg.append(svgEl('text',{x:L,y:17,fill:c,'font-size':13,'font-weight':600},label));
  const intact=DATA.activity.intactMeanHz;
  svg.append(svgEl('text',{x:W-R,y:17,'text-anchor':'end',fill:'#3d5058','font-size':12},`${d.meanMotorHz.toFixed(1)} Hz · ${(100*d.meanMotorHz/intact).toFixed(1)}% intact`));
  const raster=svgEl('g',{'data-raster':'true'});
  d.spikes.forEach(([t,i])=>{const y=rasterTop+(i+.5)*rasterHeight/12;raster.append(svgEl('line',{x1:px(t),x2:px(t),y1:y-1.9,y2:y+1.9,stroke:c,'stroke-width':.9,opacity:.74}));});
  svg.append(raster);
  svg.append(svgEl('text',{x:L-7,y:rasterTop+8,'text-anchor':'end',fill:'#718187','font-size':10},'TTMn'));
  svg.append(svgEl('text',{x:L-7,y:rasterTop+42,'text-anchor':'end',fill:'#718187','font-size':10},'DLMn'));
  svg.append(svgEl('line',{x1:L,x2:W-R,y1:rasterTop+2*rasterHeight/12,y2:rasterTop+2*rasterHeight/12,stroke:'#e4e9e6','stroke-width':.8}));
  svg.append(svgEl('text',{x:L,y:128,fill:'#61747b','font-size':11},'Looming motor activity'));
  const ticks=[0,50,100,150,200,250,300].filter(v=>v<=DATA.activity.ymaxHz);
  ticks.forEach(v=>{svg.append(svgEl('line',{x1:L,x2:W-R,y1:py(v),y2:py(v),stroke:'#e8ece9','stroke-width':.8}));svg.append(svgEl('text',{x:L-7,y:py(v)+4,'text-anchor':'end',fill:'#718187','font-size':11},String(v)));});
  svg.append(svgEl('line',{x1:L,x2:L,y1:traceTop,y2:traceBottom,stroke:'#a7b4af','stroke-width':.8}));
  [0,100,200,300].forEach(v=>{svg.append(svgEl('line',{x1:px(v),x2:px(v),y1:traceBottom,y2:traceBottom+4,stroke:'#a7b4af'}));svg.append(svgEl('text',{x:px(v),y:traceBottom+17,'text-anchor':'middle',fill:'#718187','font-size':11},String(v)));});
  svg.append(svgEl('text',{x:L+PW/2,y:H-5,'text-anchor':'middle',fill:'#61747b','font-size':11},'Neural model time (ms)'));
  svg.append(svgEl('text',{x:13,y:(traceTop+traceBottom)/2,transform:`rotate(-90 13 ${(traceTop+traceBottom)/2})`,'text-anchor':'middle',fill:'#61747b','font-size':11},'Rate (Hz)'));
  const points=d.motorRateHz.map((r,i)=>`${px(i)},${py(r)}`).join(' ');
  svg.append(svgEl('polyline',{points,fill:'none',stroke:c,'stroke-width':1.75,'stroke-linecap':'round','stroke-linejoin':'round','data-rate-trace':'true'}));
  return svg;
}
async function firing(){
  const request=++firingEpoch;firingReady=false;
  $('firing-status').textContent='Loading matched firing recordings…';
  $('reference-firing').replaceChildren();$('selected-firing').replaceChildren();
  const base=state.mode==='lesion'?'intact':'lesion20_2000',name=current.name;
  try{
    const [a,b]=await Promise.all([firingData(base),firingData(name)]);
    if(request!==firingEpoch)return;
    const refLabel=state.mode==='lesion'?'Intact reference':'No graft';
    const selectedLabel=state.mode==='lesion'?current.lost?fmt(current.lost)+' neurons lost':'Intact reference':state.rule==='none'||state.dose===0?'No graft':ruleInfo().label;
    $('reference-firing').append(firingChart(a,refLabel,state.mode==='lesion'?refColor:orange));
    $('selected-firing').append(firingChart(b,selectedLabel,color()));
    $('firing-status').textContent='Spikes and trailing 25-ms motor rates from the same circuit and neural seed as the flight example. Means use 50–300 ms; n = 1. This neural trial is separate from the flight replay.';
    firingReady=true;
  }catch(e){if(request===firingEpoch)$('firing-status').textContent='The matched firing recording could not be loaded. Please reload to retry.';}
}
async function payload(name){
  if(cache.has(name))return cache.get(name);
  const promise=fetch(`paths/${name}.json`).then(r=>{if(!r.ok)throw Error('Saved path could not be loaded.');return r.json();});
  cache.set(name,promise);if(cache.size>6)cache.delete(cache.keys().next().value);return promise;
}
async function readyFrame(frame,style){
  if(frame.dataset.style!==style){
    frame.dataset.style=style;frame.dataset.initialized='';
    frame._readyPromise=new Promise((resolve,reject)=>{frame.onload=resolve;frame.onerror=reject;frame.src=`scenes/${style}.html`;});
  }
  await frame._readyPromise;
  if(frame.dataset.style!==style)return;
  const w=Math.round(frame.clientWidth),h=Math.round(frame.clientHeight);
  if(!w||!h)throw Error('Animation panel has no size.');
  if(!frame.dataset.initialized){frame.contentWindow.init(w,h);frame.dataset.initialized='yes';}else frame.contentWindow.resizeScene(w,h);
}
async function flights(){
  const request=++epoch;sceneReady=false;$('scene-status').textContent='Loading saved paths…';
  const base=state.mode==='lesion'?'intact':'lesion20_2000';
  $('reference-title').textContent=state.mode==='lesion'?'Intact reference':'No graft · same 2,000-cell lesion';
  $('selected-title').textContent=state.mode==='lesion'?current.lost?`${fmt(current.lost)} lost · site ${state.site}`:'Intact reference':`${state.rule==='none'||state.dose===0?'No graft':ruleInfo().label}${current.graft?' · '+fmt(current.graft)+' cells':''}`;
  $('selected-title').style.color=color();$('reference-title').style.color=state.mode==='lesion'?refColor:orange;
  try{
    const [a,b]=await Promise.all([payload(base),payload(current.name)]);
    if(request!==epoch)return;
    const style=state.style;
    await Promise.all([readyFrame($('reference-scene'),style),readyFrame($('selected-scene'),style)]);
    if(request!==epoch)return;
    referencePayload={...a,color:state.mode==='lesion'?refColor:orange};selectedPayload={...b,color:color()};
    $('reference-scene').contentWindow.setData(referencePayload);$('selected-scene').contentWindow.setData(selectedPayload);
    sceneReady=true;time=0;lastFrame=performance.now();drawFrame();drawPathMap();
    $('scene-status').textContent='Saved example: neural seed 1000 · same gust sequence';
  }catch(e){if(request===epoch){$('scene-status').textContent='The 3D view is unavailable in this browser. The saved path map and numerical results remain available.';const [a,b]=await Promise.all([payload(base),payload(current.name)]);referencePayload=a;selectedPayload=b;drawPathMap();document.querySelector('.path-details').open=true;}}
}
function drawFrame(){
  if(!sceneReady)return;
  try{sceneSample=[$('reference-scene').contentWindow.renderFrame(time),$('selected-scene').contentWindow.renderFrame(time)];
    $('reference-distance').textContent=sceneSample[0].displacement_m.toFixed(2)+' m from start';$('selected-distance').textContent=sceneSample[1].displacement_m.toFixed(2)+' m from start';
    $('time').value=Math.round(time);$('clock').textContent=(time/1000).toFixed(1)+' s';
  }catch(e){sceneReady=false;$('scene-status').textContent='The animation stopped. Reload the page to retry the 3D view.';}
}
function drawPathMap(){
  if(!referencePayload||!selectedPayload)return;
  const W=Math.max(300,Math.min(730,$('path-map').clientWidth||$('curves').clientWidth)),H=330,L=58,R=18,T=W<500?47:35,B=50,[xmin,xmax,ymin,ymax]=DATA.bounds;
  const scale=Math.min((W-L-R)/(xmax-xmin),(H-T-B)/(ymax-ymin));
  const offset=L+((W-L-R)-(xmax-xmin)*scale)/2;
  const px=x=>offset+(x-xmin)*scale,py=y=>H-B-(y-ymin)*scale;
  const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'Saved reference and selected flight trajectories, lateral versus forward displacement in meters.'});
  for(let x=Math.ceil(xmin*2)/2;x<=xmax;x+=.5){svg.append(svgEl('line',{x1:px(x),x2:px(x),y1:py(ymax),y2:py(ymin),stroke:'#e6ebe7'}));svg.append(svgEl('text',{x:px(x),y:py(ymin)+19,'text-anchor':'middle',fill:'#697e83','font-size':12},x.toFixed(1)));}
  for(let y=Math.ceil(ymin*2)/2;y<=ymax;y+=.5){svg.append(svgEl('line',{x1:px(xmin),x2:px(xmax),y1:py(y),y2:py(y),stroke:'#e6ebe7'}));svg.append(svgEl('text',{x:px(xmin)-8,y:py(y)+4,'text-anchor':'end',fill:'#697e83','font-size':12},y.toFixed(1)));}
  svg.append(svgEl('line',{x1:px(xmin),x2:px(xmax),y1:py(ymin),y2:py(ymin),stroke:'#8faaa0'}));svg.append(svgEl('line',{x1:px(xmin),x2:px(xmin),y1:py(ymin),y2:py(ymax),stroke:'#8faaa0'}));
  [referencePayload,selectedPayload].forEach((p,i)=>{const ox=p.traj[0][3],oy=p.traj[0][4],c=i===0?(state.mode==='lesion'?refColor:orange):color();
    const points=p.traj.filter((r,j)=>j%10===0||j===9999).map(r=>`${px(r[4]-oy)},${py(r[3]-ox)}`).join(' ');
    svg.append(svgEl('polyline',{points,fill:'none',stroke:c,'stroke-width':2.3,'stroke-dasharray':i===0?'5 4':'',opacity:i===0?.75:1}));
    for(let j=0;j<=9000;j+=1000){const r=p.traj[j];svg.append(svgEl('circle',{cx:px(r[4]-oy),cy:py(r[3]-ox),r:2.7,fill:c}));}
  });
  svg.append(svgEl('text',{x:(px(xmin)+px(xmax))/2,y:H-10,'text-anchor':'middle',fill:'#4d676f','font-size':12},'Lateral displacement (m)'));
  svg.append(svgEl('text',{x:offset-53,y:H/2,transform:`rotate(-90 ${offset-53} ${H/2})`,'text-anchor':'middle',fill:'#4d676f','font-size':12},'Forward displacement (m)'));
  svg.append(svgEl('text',{x:L,y:17,fill:state.mode==='lesion'?refColor:orange,'font-size':12},state.mode==='lesion'?'Dashed: intact':'Dashed: no graft'));
  svg.append(svgEl('text',{x:W<500?L:L+170,y:W<500?33:17,fill:color(),'font-size':12},'Solid: selected condition'));
  $('path-map').replaceChildren(svg);
}
function setPlaying(value){playing=value;$('play').textContent=playing?'Ⅱ Pause':'▶ Play';$('play').setAttribute('aria-label',playing?'Pause flight replay':'Play flight replay');lastFrame=performance.now();}
function loop(now){
  if(playing&&sceneReady&&visible){
    time=Math.min(9999,time+Math.min(now-lastFrame,80));drawFrame();if(time===9999)setPlaying(false);
  }
  lastFrame=now;requestAnimationFrame(loop);
}
function update(){
  validateState();storeSelection();setButtons();
  $('explorer').hidden=false;
  current=DATA.conditions[conditionName()];updateControls();summary();curves();flights();firing();setPlaying(!matchMedia('(prefers-reduced-motion: reduce)').matches);
}
function downloadCondition(){
  const lines=['condition,site,neurons_lost,replacement_cells,n,metric,mean_intact_normalized,ci95_low,ci95_high,simulation_p025,simulation_p975,absolute_mean,absolute_unit'];
  DATA.metrics.forEach(({key,label})=>{const m=current.metrics[key];lines.push([current.name,current.site,current.lost,current.graft,100,label,m.mean,m.ciLow,m.ciHigh,m.rangeLow,m.rangeHigh,m.absolute?.mean??'',m.absolute?.unit??''].join(','));});
  const blob=new Blob([lines.join('\n')+'\n'],{type:'text/csv'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=current.name+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);
}
async function init(){
  try{
    const [response,activity]=await Promise.all([fetch('data.json'),fetch('activity/index.json')]);if(!response.ok||!activity.ok)throw Error('Data unavailable');DATA=await response.json();DATA.activity=await activity.json();
    document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{state.mode=b.dataset.mode;update();}));
    document.querySelectorAll('[data-style]').forEach(b=>b.addEventListener('click',()=>{state.style=b.dataset.style;storeSelection();setButtons();flights();}));
    $('site').addEventListener('change',e=>{state.site=Number(e.target.value);if(!DATA.sizes[state.site].includes(state.size))state.size=1000;update();});
    $('lesion-size').addEventListener('change',e=>{state.size=Number(e.target.value);update();});
    $('lesion-slider').addEventListener('input',e=>{state.size=DATA.sizes[state.site][Number(e.target.value)];update();});
    document.querySelectorAll('input[name=rule]').forEach(r=>r.addEventListener('change',()=>{state.rule=r.value;update();}));
    $('replacement-dose').addEventListener('change',e=>{state.dose=Number(e.target.value);update();});
    $('dose-slider').addEventListener('input',e=>{state.dose=DATA.doses[Number(e.target.value)];update();});
    $('compare-rules').addEventListener('click',()=>{state.compare=!state.compare;curves();});
    $('play').addEventListener('click',()=>{if(time===9999){time=0;drawFrame();}setPlaying(!playing);});
    $('restart').addEventListener('click',()=>{time=0;drawFrame();setPlaying(true);});
    $('time').addEventListener('input',e=>{setPlaying(false);time=Number(e.target.value);drawFrame();});
    document.querySelector('.path-details').addEventListener('toggle',()=>{if(document.querySelector('.path-details').open)drawPathMap();});
    $('download-condition').addEventListener('click',downloadCondition);
    $('copy-link').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(location.href);$('copy-status').textContent='Selection link copied.';}catch(e){$('copy-status').textContent='Copy the address in your browser to share this selection.';}});
    document.addEventListener('visibilitychange',()=>{visible=!document.hidden;lastFrame=performance.now();});
    const resize=new ResizeObserver(()=>{if(sceneReady){for(const id of ['reference-scene','selected-scene']){const f=$(id);if(f.clientWidth&&f.clientHeight)f.contentWindow.resizeScene(Math.round(f.clientWidth),Math.round(f.clientHeight));}drawFrame();}});resize.observe($('reference-scene'));resize.observe($('selected-scene'));
    $('loading').hidden=true;update();requestAnimationFrame(loop);
    window.flyExplorer={state,get current(){return current;},get sceneSample(){return sceneSample;},get sceneReady(){return sceneReady;},get firingReady(){return firingReady;},get data(){return DATA;},select:patch=>{Object.assign(state,patch);update();},renderAt:ms=>{setPlaying(false);time=ms;drawFrame();return sceneSample;}};
  }catch(e){$('loading').textContent='The results could not be loaded. Please reload this page using the website address.';}
}
init();
