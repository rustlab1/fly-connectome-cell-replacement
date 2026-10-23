"""Publication films from frozen V8 examples. No neural/controller re-simulation."""
from pathlib import Path
import argparse, hashlib, json, subprocess, tempfile, shutil, re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import navigation3d_smooth as smooth

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'videos/v8/publication_movies_20261012'
OUT.mkdir(parents=True,exist_ok=True)
STEMS={1:'Supplementary_Movie_1_lesion_size_and_location',2:'Supplementary_Movie_2_neuronal_responses',
       3:'Supplementary_Movie_3_cell_selection',4:'Supplementary_Movie_4_replacement_identity_and_wiring'}
TITLES={1:'Lesion size and location',2:'Neuronal responses to stimulation',3:'Selecting cells to restore',4:'Replacement identity and wiring'}
LABELS=['No graft','Original repair','Random wiring','Correct-type graft','Non-target graft']
COLORS=['#d7652a','#008c7a','#50545b','#2b70b6','#c16888']
NODE=smooth.base.NODE; PLAYWRIGHT=smooth.base.PLAYWRIGHT
FFMPEG=smooth.base.FFMPEG; FFPROBE=smooth.base.FFPROBE
CHROMIUM='/Users/ruslanrust/Library/Caches/ms-playwright/chromium-1217/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def probe(p):
    return json.loads(subprocess.check_output([FFPROBE,'-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,r_frame_rate,duration,nb_frames','-of','json',str(p)]))['streams'][0]
def verify_video(p,frames,fps):
    q=probe(p);assert q['width']==1920 and q['height']==1080
    assert int(q['nb_frames'])==frames and q['r_frame_rate']==f'{fps}/1'
    subprocess.run([FFMPEG,'-v','error','-threads','2','-i',str(p),'-f','null','-'],check=True)
    return q
def contact_sheet(paths,target):
    sheet=Image.new('RGB',(1920,1080),'white')
    for i,p in enumerate(paths):
        im=Image.open(p).convert('RGB');im.thumbnail((960,540),Image.Resampling.LANCZOS)
        sheet.paste(im,(i%2*960,i//2*540))
    sheet.save(target)

def anatomy(neu,lost,templates,color,path):
    xyz=neu[['x_um','y_um','z_um']].to_numpy().copy();xyz[:,2]=np.where(xyz[:,2]>450,xyz[:,2]-90,xyz[:,2])
    good=np.isfinite(xyz).all(axis=1);ix=np.flatnonzero(good)
    ix=np.random.default_rng(0).choice(ix,min(26000,len(ix)),replace=False)
    fig,ax=plt.subplots(figsize=(1.9,2.0),dpi=150);fig.patch.set_alpha(0);ax.patch.set_alpha(0)
    ax.scatter(xyz[ix,0],-xyz[ix,2],s=.18,color='#b9c6d0',alpha=.26,linewidths=0)
    ll=lost[good[lost]];ax.scatter(xyz[ll,0],-xyz[ll,2],s=.8,color='#df9bb0',alpha=.65,linewidths=0)
    if templates is not None:
        gg=templates[good[templates]];ax.scatter(xyz[gg,0],-xyz[gg,2],s=1.7,color=color,alpha=.9,linewidths=0)
    ax.set_xlim(0,770);ax.set_ylim(-1010,-45);ax.set_aspect('equal');ax.axis('off')
    fig.subplots_adjust(0,0,1,1);fig.savefig(path,transparent=True);plt.close(fig)

def source_cases(number):
    rows=pd.read_csv(ROOT/'results/v8/integrated_20261012/per_realization.csv')
    rows=rows[rows.rep==0].set_index('name');neu=pd.read_parquet(ROOT/'data/v5/neurons.parquet')
    refpath=ROOT/'results/v8/trajectories/intact.csv';reference=pd.read_csv(refpath).to_numpy()
    sources=[ROOT/'results/v8/integrated_20261012/per_realization.csv',ROOT/'data/v5/neurons.parquet',refpath]
    if number==1:
        chapters=[[f'lesion{s}_{n}' for s in [20,25] for n in [1000,4000,16000]]]
        names=[f'{n:,} neurons lost' for s in [20,25] for n in [1000,4000,16000]]
        colours=[COLORS[0]]*3+[COLORS[3]]*3
        notes=['Site 20']*3+['Site 25']*3
    elif number==3:
        chapters=[['lesion20_2000','selection_random_100','selection_interface_100','selection_ranked_100']]
        names=['No graft','Random selection','Projection priority','Motor-control priority']
        colours=[COLORS[0],COLORS[2],'#7a51b4',COLORS[1]]
        notes=['2,000 neurons lost','100 cells; original wiring','100 cells; original wiring','100 cells; original wiring']
    else:
        chapters=[['lesion20_2000','selection_ranked_100','connections_random_partners_100','connections_correct_types_100','non_target_types_100'],
                  ['lesion20_2000','complete_original_2000','review_full_random_partners_2000','typed_dose_correct_types_2000','non_target_types_2000']]
        names=LABELS;colours=COLORS;notes=['']*5
    payloads=[];records=[]
    for chapter,conditions in enumerate(chapters):
        group=[]
        for j,name in enumerate(conditions):
            row=rows.loc[name];assert int(row.seed)==1000 and int(row.intervention_seed)==2000
            directory=ROOT/'results/v8/figure2_full_replacement_20261012' if name.startswith('review_full_') else ROOT/'results/v8'
            tp=directory/'trajectories'/f'{name}.csv';lp=directory/'condition_indices'/f'{name}_lesion.npy'
            gp=directory/'condition_indices'/f'{name}_templates.npy'
            tr=pd.read_csv(tp).to_numpy();lost=np.load(lp);templ=np.load(gp) if int(row.n_graft) else None
            assert tr.shape==(10000,5) and np.isfinite(tr).all() and len(lost)==int(row.n_lesion)
            if templ is not None:assert len(templ)==int(row.n_graft) and np.isin(templ,lost).all()
            sources.extend([tp,lp]+([gp] if templ is not None else []))
            pic=OUT/f'movie{number}_chapter{chapter+1}_{name}_anatomy.png';anatomy(neu,lost,templ,colours[j],pic)
            p=dict(traj=tr.tolist(),reference=reference.tolist(),title=names[j],subtitle=notes[j],color=colours[j],
                   cns=None,fall=None,tier_label='',tier_color=colours[j],stats_html='',inset_label='',graft_size=4.,anatomy_url=pic.as_uri())
            group.append(smooth.prepare_display(p))
            records.append(dict(chapter=chapter+1,source_condition=name,label=names[j],site=int(row.site),n_lesion=int(row.n_lesion),
                n_replacement=int(row.n_graft),neural_seed=1000,intervention_seed=2000,gust_seed=3,
                trajectory_file=str(tp.relative_to(ROOT)),lesion_file=str(lp.relative_to(ROOT)),
                replacement_file=str(gp.relative_to(ROOT)) if templ is not None else None,trajectory_points=len(tr),
                final_net_displacement_m=float(np.linalg.norm(tr[-1,3:5]-tr[0,3:5])),
                **{k:float(row[k]) for k in ['gain_R','gain_L','bias','drive']}))
        payloads.append(group)
    xy=np.concatenate([np.asarray(p['traj'])[:,3:5]-np.asarray(p['traj'])[0,3:5] for group in payloads for p in group]+[reference[:,3:5]-reference[0,3:5]])
    bounds=[float(xy[:,1].min())-.12,float(xy[:,1].max())+.12,float(xy[:,0].min())-.12,float(xy[:,0].max())+.12]
    for group in payloads:
        for p in group:p['reference_bounds']=bounds
    return payloads,records,{str(p.relative_to(ROOT)):sha(p) for p in sources}

MAP=r'''
function drawReference(f,t) {
 const c=document.getElementById('pathmap'),g=c.getContext('2d'),[xmin,xmax,ymin,ymax]=DATA.reference_bounds;
 c.style.display='block';const w=c.width,h=c.height,L=37,R=10,T=21,B=32;
 const px=v=>L+(v-xmin)/(xmax-xmin)*(w-L-R),py=v=>h-B-(v-ymin)/(ymax-ymin)*(h-B-T);
 g.clearRect(0,0,w,h);g.fillStyle='rgba(255,255,255,.96)';g.fillRect(0,0,w,h);
 g.strokeStyle='#b8c2ca';g.lineWidth=1;g.beginPath();g.moveTo(L,T);g.lineTo(L,h-B);g.lineTo(w-R,h-B);g.stroke();
 g.font='11px Arial';g.fillStyle='#44515b';g.textAlign='center';
 for(const v of [xmin,xmax]){g.fillText(v.toFixed(1),px(v),h-B+13);}
 g.textAlign='right';for(const v of [ymin,ymax])g.fillText(v.toFixed(1),L-4,py(v)+3);
 g.textAlign='center';g.fillText('Lateral displacement (m)',w/2+12,h-3);
 g.save();g.translate(11,h/2);g.rotate(-Math.PI/2);g.fillText('Forward (m)',0,0);g.restore();
 function path(tr,color,dash) {
  g.strokeStyle=color;g.lineWidth=2;g.setLineDash(dash);g.beginPath();const x0=tr[0][3],y0=tr[0][4];
  for(let j=0;j<=Math.floor(f);j+=15){const p=tr[j];if(j===0)g.moveTo(px(p[4]-y0),py(p[3]-x0));else g.lineTo(px(p[4]-y0),py(p[3]-x0));}
  const p=sampleArray(tr,f);g.lineTo(px(p[4]-y0),py(p[3]-x0));g.stroke();g.setLineDash([]);
  g.beginPath();g.arc(px(p[4]-y0),py(p[3]-x0),3.2,0,2*Math.PI);g.fillStyle=color;g.fill();
 }
 path(DATA.reference,'#8c969e',[3,3]);path(DATA.traj,DATA.color,[]);
 g.fillStyle='#44515b';g.font='12px Arial';g.textAlign='left';g.fillText('Intact reference',L,13);
}
'''

LANDMARKS=r'''
function buildLandmarks() {
 if (scene.userData.landmarks) scene.remove(scene.userData.landmarks);
 const group=new THREE.Group();scene.userData.landmarks=group;scene.add(group);
 const rng=mulberry32(7),points=[];
 for(let x=-40;x<=100;x+=20)for(let z=-60;z<=60;z+=20){
  const px=x+(rng()-.5)*4,pz=z+(rng()-.5)*4;
  const h=2.8+rng()*.8,r=.85+rng()*.30;
  if(Math.hypot(px,pz)>6)points.push({x:px,z:pz,h,r});
 }
 const trunk=new THREE.InstancedMesh(new THREE.CylinderGeometry(.10,.17,1,7),new THREE.MeshStandardMaterial({color:0xb0a797,roughness:1}),points.length);
 const lower=new THREE.InstancedMesh(new THREE.ConeGeometry(1,1,9),new THREE.MeshStandardMaterial({color:0x9eb595,roughness:1}),points.length);
 const upper=new THREE.InstancedMesh(new THREE.ConeGeometry(1,1,9),new THREE.MeshStandardMaterial({color:0x8fa88a,roughness:1}),points.length);
 const m=new THREE.Matrix4(),scale=new THREE.Vector3(),rotation=new THREE.Quaternion();
 points.forEach((p,i)=>{
  m.compose(new THREE.Vector3(p.x,.6,p.z),rotation,scale.set(1,1.2,1));trunk.setMatrixAt(i,m);
  m.compose(new THREE.Vector3(p.x,1.35,p.z),rotation,scale.set(p.r,1.75,p.r));lower.setMatrixAt(i,m);
  m.compose(new THREE.Vector3(p.x,p.h-.95,p.z),rotation,scale.set(p.r*.72,1.90,p.r*.72));upper.setMatrixAt(i,m);
 });
 for(const mesh of [trunk,lower,upper]){mesh.castShadow=true;mesh.receiveShadow=true;group.add(mesh);}
 scene.userData.landmarkCount=points.length;
 scene.userData.landmarkLayout='fixed-world-grid-seed-7-v1';
}
'''

def navigation_html():
    src=smooth.smooth_html(8)
    start=src.index('function buildLandmarks(');end=src.index('function mulberry32(',start)
    src=src[:start]+LANDMARKS+'\n'+src[end:]
    src=re.sub(r'  buildLandmarks\([^;]+;', '  buildLandmarks();',src)
    src=src.replace('trail_dot_count:nc,dynamic_trails_not_culled:',
        'landmark_count:scene.userData.landmarkCount,landmark_layout:scene.userData.landmarkLayout,trail_dot_count:nc,dynamic_trails_not_culled:')
    src=src.replace('0xf4f7fb','0xffffff').replace('0xe9eee8','0xf9fafb')
    src=src.replace('sun.shadow.mapSize.set(2048, 2048)','sun.shadow.mapSize.set(512, 512)')
    start=src.index('function drawReference(');end=src.index('function renderFrame(',start)
    src=src[:start]+MAP+src[end:]
    src=src.replace('const alt=4.+.15*Math.sin(t*3.1)','const alt=4.')
    src=src.replace('4.+.15*Math.sin(p[0]*3.1)','4.')
    src=src.replace('<canvas id="pathmap" width="270" height="160"></canvas>',
        '<canvas id="pathmap" width="220" height="155"></canvas><img id="anatomy" alt="Lesion and graft">')
    src=src.replace("function setData(d) { DATA = d; buildInset(d.cns, d.graft_size);",
        "function setData(d) { DATA = d; buildInset(d.cns, d.graft_size);document.getElementById('anatomy').src=d.anatomy_url;")
    src=src.replace('</style>',r'''
    #title{left:18px;top:14px;font:600 24px Arial;color:#24313a}
    #sub{left:18px;top:45px;font:18px Arial;color:#44515b}
    #stats{left:18px;bottom:15px;font:19px Arial;color:#24313a;padding:4px 0;background:transparent}
    #time{display:none}#insetlabel,#insetframe{display:none}
    #pathmap{right:14px;bottom:12px;width:220px;height:155px;border:0;border-radius:0}
    body:before{content:'';position:absolute;left:0;top:0;width:100%;height:72px;background:rgba(255,255,255,.98)}
    #anatomy{position:absolute;left:8px;top:72px;width:114px;height:120px;background:rgba(255,255,255,.94)}
    </style>''')
    return src

DRIVER=r'''
const fs=require('fs'),crypto=require('crypto');
const {chromium}=require(process.argv[2]),{spawn}=require('child_process'),{once}=require('events');
(async()=>{
 const cfg=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
 const browser=await chromium.launch({headless:true,executablePath:cfg.chromium,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist','--allow-file-access-from-files']});
 const page=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});const errors=[];
 page.on('pageerror',e=>errors.push(e.message));await page.goto(cfg.wrapperUrl,{waitUntil:'load'});
 await page.waitForFunction(()=>Array.from(document.querySelectorAll('iframe')).every(f=>typeof f.contentWindow.init==='function'));
 await page.evaluate(cfg=>document.querySelectorAll('iframe').forEach(f=>f.contentWindow.init(cfg.panelWidth,cfg.panelHeight)),cfg);
 const samples=[],fps=60,frames=cfg.previewOnly?cfg.previewFrames:Array.from({length:cfg.frames},(_,i)=>i);
 let encoder=null,err='',last=null,duplicates=0,chapterPrevious=-1;
 if(!cfg.previewOnly){encoder=spawn(cfg.ffmpeg,['-y','-loglevel','error','-f','image2pipe','-vcodec','mjpeg','-framerate','60','-i','-','-an','-c:v','libx264','-threads','3','-preset','medium','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',cfg.output]);encoder.stderr.on('data',d=>err+=d);}
 const begin=Date.now();
 for(const n of frames){
  const chapter=Math.floor(n/600),local=n%600,timeMs=local*1000/fps;
  if(chapter!==chapterPrevious){await page.evaluate(({cfg,chapter})=>{
    document.querySelectorAll('iframe').forEach((f,i)=>f.contentWindow.setData(cfg.payloads[chapter][i]));
    document.getElementById('chapter').textContent=cfg.chapterLabels[chapter];
    if(cfg.matrices)document.getElementById('matrix').innerHTML=cfg.matrices[chapter];
  },{cfg,chapter});chapterPrevious=chapter;}
  const sig=await page.evaluate(timeMs=>Array.from(document.querySelectorAll('iframe')).map(f=>f.contentWindow.renderFrame(timeMs)),timeMs);
  samples.push({frame:n,chapter:chapter+1,conditions:sig});
  await page.evaluate(timeMs=>document.getElementById('clock').textContent=(timeMs/1000).toFixed(1)+' s',timeMs);
  if(cfg.previewFrames.includes(n))await page.screenshot({path:cfg.previewPrefix+'_frame_'+String(n).padStart(4,'0')+'.png'});
  if(encoder){const b=await page.screenshot({type:'jpeg',quality:94});const h=crypto.createHash('sha256').update(b).digest('hex');if(h===last)duplicates++;last=h;if(!encoder.stdin.write(b))await once(encoder.stdin,'drain');}
  if(!cfg.previewOnly&&n%120===0)process.stdout.write('Movie '+cfg.movie+': '+n+'/'+cfg.frames+' frames; '+((Date.now()-begin)/1000).toFixed(1)+'s\n');
 }
 if(encoder){encoder.stdin.end();const[code]=await once(encoder,'close');if(code!==0)throw new Error(err);}
 const first=await page.evaluate(()=>Array.from(document.querySelectorAll('iframe')).map(f=>f.contentWindow.renderFrame(3500)));
 await page.evaluate(()=>document.querySelectorAll('iframe').forEach(f=>f.contentWindow.renderFrame(1250)));
 const second=await page.evaluate(()=>Array.from(document.querySelectorAll('iframe')).map(f=>f.contentWindow.renderFrame(3500)));
 fs.writeFileSync(cfg.qa,JSON.stringify({errors,samples,duplicateFrames:duplicates,deterministic_call_order:JSON.stringify(first)===JSON.stringify(second),elapsed_s:(Date.now()-begin)/1000},null,2));
 await browser.close();if(errors.length)throw new Error(errors.join('\n'));
})().catch(e=>{process.stderr.write(String(e.stack||e));process.exit(1)});
'''

def matrix(dose):
    table=pd.read_csv(ROOT/'figures/v8/Fig2_v8_cell_replacement_source_values.csv')
    columns=['Distance','Virtual crossings/s','Wing motor','Looming motor','Spared visual']
    s='<h3>Remaining function</h3><p>Intact = 1; mean, n = 100</p><table><tr><th></th>'
    s+=''.join(f'<th style="color:{c}">{l}</th>' for l,c in zip(['No graft','Original','Random','Correct type','Non-target'],COLORS))+'</tr>'
    for metric in columns:
        s+=f'<tr><td>{metric}</td>'
        for label in LABELS:
            v=table[(table.condition==label)&(table.readout==metric)&(table.dose==(0 if label=='No graft' else dose))].iloc[0]
            s+=f'<td>{v.mean_intact_normalized:.2f}</td>'
        s+='</tr>'
    return s+'</table><div class="key"><span style="color:#df9bb0">● Lesion</span> &nbsp; ● Replacement<br>Grey dashed path: intact reference</div>'

def render_navigation(number,preview=False):
    payloads,records,hashes=source_cases(number);rows,cols=(2,2) if number==3 else (2,3)
    width,height=1920//cols,470;frames=600*len(payloads)
    qa=OUT/f'qa_movie{number}_navigation';qa.mkdir(exist_ok=True);stem=STEMS[number]
    snapshots=[0,180,360,599] if number!=4 else [180,599,780,1199]
    chapterlabels={1:['1,000, 4,000 and 16,000 lost neurons'],3:['2,000 neurons lost; 100 original-wiring replacements'],
                   4:['100 replacement cells','2,000 replacement cells']}[number]
    with tempfile.TemporaryDirectory(prefix=f'fly-curated-{number}-',dir='/private/tmp') as td:
        td=Path(td);(td/'scene.html').write_text(navigation_html())
        cards=''.join(f'<iframe style="left:{i%cols*width}px;top:{110+i//cols*height}px;width:{width}px;height:{height}px" src="scene.html"></iframe>' for i in range(len(payloads[0])))
        extra='<div id="matrix" style="position:absolute;left:1280px;top:580px;width:604px;height:450px;padding:16px"></div>' if number==4 else ''
        css='html,body{margin:0;background:#fff;font-family:Arial;color:#24313a}iframe{position:absolute;border:0;border-bottom:1px solid #dce3e8;box-sizing:border-box}h1{position:absolute;left:30px;top:10px;margin:0;font-size:32px;font-weight:600}#chapter{position:absolute;left:30px;top:62px;font-size:22px}#clock{position:absolute;right:35px;top:35px;font-size:28px}#footer{position:absolute;left:30px;bottom:10px;font-size:18px;color:#44515b}h3{font-size:25px;margin:12px 0}p{font-size:18px;margin:12px 0}table{width:100%;border-collapse:collapse;font-size:18px}td,th{border-bottom:1px solid #dce3e8;padding:11px 3px;text-align:center}th{font-size:16px}td:first-child{text-align:left} .key{margin-top:24px;font-size:18px;line-height:1.8}'
        wrapper=f'<!doctype html><html><head><style>{css}</style></head><body><h1>{TITLES[number]}</h1><div id="chapter"></div><div id="clock"></div>{cards}{extra}<div id="footer">Matched 10-s examples · Grey path: intact reference · 3D appearance is illustrative</div></body></html>'
        (td/'wrapper.html').write_text(wrapper);(td/'driver.cjs').write_text(DRIVER)
        cfg=dict(movie=number,panelWidth=width,panelHeight=height,frames=frames,payloads=payloads,chapterLabels=chapterlabels,
            matrices=[matrix(d) for d in [100,2000]] if number==4 else None,previewOnly=preview,previewFrames=snapshots,
            wrapperUrl=(td/'wrapper.html').as_uri(),output=str(OUT/f'{stem}.mp4'),previewPrefix=str(qa/stem),
            qa=str(qa/('preview_browser_qa.json' if preview else 'browser_qa.json')),ffmpeg=FFMPEG,chromium=CHROMIUM)
        (td/'config.json').write_text(json.dumps(cfg))
        subprocess.run([str(NODE),str(td/'driver.cjs'),PLAYWRIGHT,str(td/'config.json')],check=True)
    images=[qa/f'{stem}_frame_{i:04d}.png' for i in snapshots];contact_sheet(images,OUT/f'{stem}_contact_sheet.png')
    shutil.copy2(images[2],OUT/f'{stem}_preview.png')
    if preview:return
    video=OUT/f'{stem}.mp4';stream=verify_video(video,frames,60)
    b=json.loads((qa/'browser_qa.json').read_text());assert not b['errors'] and b['deterministic_call_order'] and not b['duplicateFrames']
    checks=[]
    for sample in b['samples']:
        chapter=sample['chapter']-1
        for i,s in enumerate(sample['conditions']):
            tr=np.asarray(payloads[chapter][i]['traj']);t=s['model_time']
            expected=[np.interp(t,tr[:,0],tr[:,c]) for c in [3,4]]
            error=float(np.max(np.abs(np.asarray([s['model_x'],s['model_y']])-expected)))
            assert error<1e-12 and s['fly_scale']==[1,1,1] and s['field_of_view']==38 and s['dynamic_trails_not_culled']
            assert s['landmark_count']>=50 and s['landmark_layout']=='fixed-world-grid-seed-7-v1'
            checks.append(error)
    distances=[s['camera_distance'] for sample in b['samples'] for s in sample['conditions']]
    assert np.ptp(distances)<1e-10
    hashes['figures/v8/Fig2_v8_cell_replacement_source_values.csv']=sha(ROOT/'figures/v8/Fig2_v8_cell_replacement_source_values.csv')
    q=dict(status='STRUCTURAL_PASS_PENDING_VISUAL_REVIEW',movie=str(video.relative_to(ROOT)),number=number,title=TITLES[number],
        source_hashes=hashes,conditions=records,output_sha256=sha(video),video_probe=stream,decode_check='PASS',
        browser_qa_file=str((qa/'browser_qa.json').relative_to(ROOT)),maximum_horizontal_interpolation_error_m=max(checks),
        native_fps=60,frame_count=frames,browser_errors=b['errors'],consecutive_duplicate_encoded_source_frames=b['duplicateFrames'],
        deterministic_render_order=True,same_geometry_camera_scale=True,camera_distance=float(distances[0]),camera_field_of_view=38,
        fly_scale=[1,1,1],display_filters=smooth.TAUS,quantitative_position_smoothing=False,no_neural_or_controller_resimulation=True,
        scene_units_per_metre=25,illustrative_altitude_scene_units=4,decorative_wingbeat_hz=11,path_dot_spacing_m=.03,
        common_top_down_map_bounds=payloads[0][0]['reference_bounds'],map_axes=['Lateral displacement (m)','Forward displacement (m)'],
        stationary_tree_landmarks=dict(layout='fixed-world-grid-seed-7-v1',seed=7,
            count=b['samples'][0]['conditions'][0]['landmark_count'],same_world_positions_in_all_panels=True,
            illustrative_only=True,affect_neural_or_controller_model=False),
        chapters=[dict(start_s=i*10,duration_s=10,label=l) for i,l in enumerate(chapterlabels)],
        renderer_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'code/v8/navigation3d_smooth.py',ROOT/'code/v8/render_navigation3d.py',ROOT/'code/web/fly3d.html',ROOT/'code/web/three.min.js']})
    (OUT/f'{stem}_provenance.json').write_text(json.dumps(q,indent=2)+'\n')
    print('COMPLETE',video,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--movie',type=int,choices=[1,3,4],required=True);p.add_argument('--preview',action='store_true')
    a=p.parse_args();render_navigation(a.movie,a.preview)
