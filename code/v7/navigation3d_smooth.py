"""Native60fps 3D playback with deterministic display-only camera damping."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
from PIL import Image

import render_navigation3d as base

ROOT, OUT = base.ROOT, base.OUT
FPS, DURATION = 60, 10
MOVIE5 = 'Supplementary_Movie_5_3D_navigation_current_data'
MOVIE8 = 'Supplementary_Movie_8_3D_lesion_location_and_size'
TAUS = dict(camera_heading=.7, display_body_heading=.08, display_roll=.1, display_turn=.1)


def lowpass(values, tau, dt=.001):
    output = np.array(values, dtype=float)
    retain = np.exp(-dt/tau)
    for i in range(1,len(output)):
        output[i] = retain*output[i-1] + (1-retain)*output[i]
    return output


def prepare_display(payload):
    tr = np.asarray(payload['traj'], dtype=float)
    heading, yaw = tr[:,1], tr[:,2]
    lengths = np.linalg.norm(np.diff(tr[:,3:5],axis=0),axis=1)
    payload['path_length'] = np.concatenate([[0.],np.cumsum(lengths)]).tolist()
    payload['camera_heading'] = lowpass(heading, TAUS['camera_heading']).tolist()
    payload['body_heading'] = lowpass(heading, TAUS['display_body_heading']).tolist()
    payload['display_roll'] = lowpass(np.clip(-yaw*.25,-.6,.6), TAUS['display_roll']).tolist()
    payload['display_turn'] = lowpass(np.clip(yaw/(250*np.pi/180),-1,1), TAUS['display_turn']).tolist()
    return payload


def movie8_payloads():
    manifest_path = ROOT/'videos/v5/Movie1_lesion_location_conditions.json'
    neurons_path = ROOT/'data/processed/neurons.parquet'
    states_path = ROOT/'results/v5/tables/states.csv'
    reference_path = ROOT/'videos/v5/Movie1_v5_lesion_location_size_intact_trajectory.csv'
    manifest = json.loads(manifest_path.read_text())
    neurons, states = pd.read_parquet(neurons_path), pd.read_csv(states_path)
    reference = pd.read_csv(reference_path).to_numpy(dtype=float)
    hashes = {str(p.relative_to(ROOT)):base.sha(p) for p in [manifest_path,neurons_path,states_path,reference_path]}
    payloads, records = [],[]
    for condition in manifest['scenes']:
        name = condition['id']
        state = states[(states['name']==name)&(states.seed==0)].iloc[0]
        for key in ['gain_R','gain_L','bias','drive']:
            assert abs(float(state[key])-float(condition[key]))<1e-12
        trajectory_path = ROOT/f'videos/v5/Movie1_v5_lesion_location_size_{name}_trajectory.csv'
        trajectory = pd.read_csv(trajectory_path).to_numpy(dtype=float)
        lesion_path = ROOT/condition['lesion_idx_file']
        lesion = np.load(lesion_path)
        assert len(lesion)==condition['n_lesion']
        assert trajectory.shape==(10000,5) and np.isfinite(trajectory).all()
        for p in [trajectory_path,lesion_path]:
            hashes[str(p.relative_to(ROOT))]=base.sha(p)
        color = '#e45f41' if condition['site']==20 else '#7c61b8'
        payloads.append(dict(traj=trajectory.tolist(),title=f"{condition['n_lesion']:,} lost neurons · site {condition['site']}",
                             subtitle=condition['region_label'],color=color,
                             tier_label='',tier_color=color,stats_html='',inset_label='Lesion',
                             cns=base.cns_payload(neurons,lesion),fall=None,graft_size=4.,
                             reference=reference.tolist()))
        net=float(np.linalg.norm(trajectory[-1,3:5]-trajectory[0,3:5]))
        length=float(np.linalg.norm(np.diff(trajectory[:,3:5],axis=0),axis=1).sum())
        records.append(dict(source_condition=name,site=condition['site'],n_lesion=condition['n_lesion'],
                            n_replacement=0,neural_seed=0,gust_seed=3,
                            trajectory_file=str(trajectory_path.relative_to(ROOT)),
                            trajectory_points=len(trajectory),final_net_displacement_m=net,
                            final_path_length_m=length,final_straightness=net/length,
                            **{key:float(state[key]) for key in ['gain_R','gain_L','bias','drive']}))
    return payloads,records,hashes


RENDER_FUNCTION = r'''
function sampleArray(array, f) {
  const j=Math.max(0,Math.min(Math.floor(f),array.length-1)), k=Math.min(j+1,array.length-1), u=f-j;
  if(Array.isArray(array[j])) return array[j].map((v,l)=>v+(array[k][l]-v)*u);
  return array[j]+(array[k]-array[j])*u;
}
function drawReference(f,t) {
  if(!DATA.reference) return;
  const canvas=document.getElementById('pathmap'), context=canvas.getContext('2d');
  canvas.style.display='block';
  const cw=canvas.width,ch=canvas.height,bounds=DATA.reference_bounds;
  const pad=20,bw=cw-2*pad,bh=ch-48;
  context.clearRect(0,0,cw,ch);context.fillStyle='rgba(255,255,255,.94)';context.fillRect(0,0,cw,ch);
  const [xmin,xmax,ymin,ymax]=bounds;
  const scale=Math.min(bw/(xmax-xmin),bh/(ymax-ymin));
  const px=x=>pad+(x-xmin)*scale, py=y=>ch-20-(y-ymin)*scale;
  context.strokeStyle='#dce2e8';context.lineWidth=1;
  context.strokeRect(px(xmin),py(ymax),(xmax-xmin)*scale,(ymax-ymin)*scale);
  function path(tr,color,upto,alpha) {
    context.strokeStyle=color;context.globalAlpha=alpha;context.lineWidth=2.2;context.beginPath();
    for(let j=0;j<=upto;j+=15){const p=tr[j];if(j===0)context.moveTo(px(p[3]),py(p[4]));else context.lineTo(px(p[3]),py(p[4]));}
    const p=sampleArray(tr,f);context.lineTo(px(p[3]),py(p[4]));context.stroke();
    context.beginPath();context.arc(px(p[3]),py(p[4]),4,0,2*Math.PI);context.fillStyle=color;context.fill();context.globalAlpha=1;
  }
  path(DATA.reference,'#2b7dc0',Math.floor(f),1);
  path(DATA.traj,DATA.color,Math.floor(f),1);
  context.font='15px Helvetica,Arial,sans-serif';context.fillStyle='#2b7dc0';context.fillText('Intact reference',12,19);
  context.fillStyle=DATA.color;context.fillText('Lesion',150,19);
  context.fillStyle='#53616d';context.font='13px Helvetica,Arial,sans-serif';context.fillText('Same map scale in all panels',12,ch-5);
}
function renderFrame(timeMs) {
  const f=Math.max(0,Math.min(timeMs,DATA.traj.length-1));
  const tr=DATA.traj, [t,psi,omega,x,y]=sampleArray(tr,f);
  const bodyHeading=sampleArray(DATA.body_heading,f), cameraHeading=sampleArray(DATA.camera_heading,f);
  const roll=sampleArray(DATA.display_roll,f), turn=sampleArray(DATA.display_turn,f);
  const alt=4.+.15*Math.sin(t*3.1), X=x*S, Z=-y*S;
  const pitch=.32, freq=11;
  const ampL=1+.35*Math.max(turn,0)-.25*Math.max(-turn,0);
  const ampR=1+.35*Math.max(-turn,0)-.25*Math.max(turn,0);
  const phase=2*Math.PI*freq*t;
  fly.position.set(X,alt,Z);fly.rotation.set(0,bodyHeading,0);
  fly.userData.body.rotation.set(roll,0,pitch);
  wings[-1].rotation.set(.25,.22,.05);wings[1].rotation.set(-.25,-.22,.05);
  for(let q=0;q<3;q++) {
    const ph=phase+(q-1)*.35;
    wings[-1].userData.ghosts[q].rotation.x=-.95*ampL*Math.sin(ph);
    wings[1].userData.ghosts[q].rotation.x=.95*ampR*Math.sin(ph);
  }
  let k=0;
  for(let j=0;j<=Math.floor(f);j+=10) {const p=tr[j];trailPos[k*3]=p[3]*S;trailPos[k*3+1]=.05;trailPos[k*3+2]=-p[4]*S;k++;}
  trailPos[k*3]=X;trailPos[k*3+1]=.05;trailPos[k*3+2]=Z;k++;
  trailGeo.setDrawRange(0,k);trailGeo.attributes.position.needsUpdate=true;
  let nc=0;
  for(let j=0;j<=Math.floor(f)&&nc<400;j+=100) {const p=tr[j];_m4.makeTranslation(p[3]*S,4.+.15*Math.sin(p[0]*3.1),-p[4]*S);crumbs.setMatrixAt(nc++,_m4);}
  crumbs.count=nc;crumbs.instanceMatrix.needsUpdate=true;
  const back=new THREE.Vector3(-Math.cos(cameraHeading),0,Math.sin(cameraHeading)).multiplyScalar(13.);
  const side=new THREE.Vector3(Math.sin(cameraHeading),0,Math.cos(cameraHeading)).multiplyScalar(4.);
  const offset=back.add(side).add(new THREE.Vector3(0,6.,0));
  camera.position.set(X,alt,Z).add(offset);camera.lookAt(fly.position);
  scene.userData.sun.position.set(X+14,30,Z+10);scene.userData.sun.target.position.set(X,0,Z);scene.userData.sun.target.updateMatrixWorld();
  renderer.setViewport(0,0,W,H);renderer.setScissor(0,0,W,H);renderer.setScissorTest(false);renderer.render(scene,camera);
  if(insetCloud) {
    const iw=Math.round(W*.25),ih=Math.round(iw*.78),ix=W-iw-24,iy=H-ih-56;
    renderer.setViewport(ix,iy,iw,ih);renderer.setScissor(ix,iy,iw,ih);renderer.setScissorTest(true);renderer.autoClear=false;
    const c=insetScene.userData.center,R=insetScene.userData.radius,ang=.55*Math.sin(.45*t);
    insetCam.aspect=iw/ih;insetCam.updateProjectionMatrix();
    insetCam.position.set(c.x+R*2.4*Math.sin(ang),c.y+R*2.4*Math.cos(ang),c.z);insetCam.up.set(0,0,-1);insetCam.lookAt(c);
    const fr=document.getElementById('insetframe');fr.style.left=ix+'px';fr.style.top=(H-iy-ih)+'px';fr.style.width=iw+'px';fr.style.height=ih+'px';
    renderer.render(insetScene,insetCam);renderer.autoClear=true;renderer.setScissorTest(false);renderer.setViewport(0,0,W,H);
  }
  const displacement=Math.hypot(x-tr[0][3],y-tr[0][4]), length=sampleArray(DATA.path_length,f), straight=length>0?displacement/length:1;
  document.getElementById('stats').innerHTML=`Distance <b>${displacement.toFixed(2)} m</b><br>Straightness <b>${straight.toFixed(2)}</b>`;
  document.getElementById('time').textContent=`t = ${t.toFixed(1)} s`;
  drawReference(f,t);
  return {model_time:t,model_x:x,model_y:y,raw_heading:psi,camera_heading:cameraHeading,body_heading:bodyHeading,roll:roll,
    camera_distance:camera.position.distanceTo(fly.position),fly_scale:fly.scale.toArray(),field_of_view:camera.fov,
    displacement_m:displacement,path_length_m:length,straightness:straight};
}
'''


def smooth_html(movie):
    source=base.render_html()
    start=source.index('function renderFrame(i) {')
    end=source.index('function pathStraightness(',start)
    source=source[:start]+RENDER_FUNCTION+'\n'+source[end:]
    source=source.replace('sun.shadow.mapSize.set(2048, 2048)','sun.shadow.mapSize.set(1024, 1024)')
    source=source.replace('SphereGeometry(0.06, 8, 8)','SphereGeometry(0.04, 8, 8)')
    source=source.replace('new THREE.MeshStandardMaterial({color: 0x2a78d6, transparent: true, opacity: 0.55})','new THREE.MeshStandardMaterial({color: 0x2a78d6, transparent: true, opacity: 0.35})')
    source=source.replace('<div id="insetframe"></div>','<div id="insetframe"></div><canvas id="pathmap" width="270" height="160"></canvas>')
    source=source.replace('</style>','#pathmap {display:none;position:absolute;right:24px;bottom:24px;width:270px;height:160px;border:1px solid #d7dde6;border-radius:4px;}\n</style>')
    source=source.replace('bottom: 54px','bottom: 24px')
    if movie==8:
        source=source.replace('font-size: 31px','font-size: 27px')
        source=source.replace('font-size: 20px','font-size: 18px')
        source=source.replace('right: 28px; bottom: 24px','right: 318px; bottom: 24px')
    return source


NODE_DRIVER=r'''
const fs=require('fs'),crypto=require('crypto');
const {chromium}=require(process.argv[2]),{spawn}=require('child_process'),{once}=require('events');
(async()=>{
 const cfg=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
 const browser=await chromium.launch({headless:true,executablePath:cfg.chromium,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist','--allow-file-access-from-files']});
 const page=await browser.newPage({viewport:{width:cfg.width,height:cfg.height},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(cfg.wrapperUrl,{waitUntil:'load'});
 await page.waitForFunction(()=>Array.from(document.querySelectorAll('iframe')).every(f=>typeof f.contentWindow.init==='function'));
 await page.evaluate(cfg=>document.querySelectorAll('iframe').forEach((f,i)=>{f.contentWindow.init(cfg.panelWidth,cfg.panelHeight);f.contentWindow.setData(cfg.payloads[i]);}),cfg);
 const samples=[];let duplicateFrames=0,lastHash=null;
 let encoder=null,encoderErrors='';
 if(!cfg.previewOnly){encoder=spawn(cfg.ffmpeg,['-y','-loglevel','error','-f','image2pipe','-vcodec','mjpeg','-framerate',String(cfg.fps),'-i','-','-an','-c:v','libx264','-threads','3','-preset','medium','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',cfg.output]);encoder.stderr.on('data',d=>{encoderErrors+=d;});}
 const frames=cfg.previewOnly?cfg.previewFrames:(cfg.clipPreview?Array.from({length:120},(_,i)=>i+180):Array.from({length:cfg.fps*cfg.duration},(_,i)=>i));
 const start=Date.now();
 for(const n of frames){
  const timeMs=n*1000/cfg.fps;
  const signature=await page.evaluate(timeMs=>Array.from(document.querySelectorAll('iframe')).map(f=>f.contentWindow.renderFrame(timeMs)),timeMs);
  samples.push({frame:n,conditions:signature});
  if(cfg.previewFrames.includes(n)) await page.screenshot({path:cfg.previewPrefix+'_frame_'+String(n).padStart(3,'0')+'.png',type:'png'});
  if(encoder){const frame=await page.screenshot({type:'jpeg',quality:94});const h=crypto.createHash('sha256').update(frame).digest('hex');if(h===lastHash)duplicateFrames++;lastHash=h;if(!encoder.stdin.write(frame))await once(encoder.stdin,'drain');}
  if(!cfg.previewOnly&&n%120===0)process.stdout.write('Movie '+cfg.movie+': rendered '+Math.max(0,n-(cfg.clipPreview?180:0))+'/'+frames.length+' frames in '+((Date.now()-start)/1000).toFixed(1)+'s\n');
 }
 if(encoder){encoder.stdin.end();const[code]=await once(encoder,'close');if(code!==0)throw new Error(encoderErrors||'ffmpeg exited '+code);}
 const first=await page.evaluate(()=>Array.from(document.querySelectorAll('iframe')).map(f=>f.contentWindow.renderFrame(3500)));
 await page.evaluate(()=>document.querySelectorAll('iframe').forEach(f=>f.contentWindow.renderFrame(1250)));
 const second=await page.evaluate(()=>Array.from(document.querySelectorAll('iframe')).map(f=>f.contentWindow.renderFrame(3500)));
 fs.writeFileSync(cfg.browserQa,JSON.stringify({errors,samples,duplicateFrames,deterministic_call_order:JSON.stringify(first)===JSON.stringify(second),elapsed_s:(Date.now()-start)/1000},null,2));
 await browser.close();if(errors.length)throw new Error(errors.join('\n'));
})().catch(e=>{process.stderr.write(String(e.stack||e));process.exit(1)});
'''


def caption(movie):
    common=('All example paths use neural seed 0, the same gust seed 3 and ten seconds of model time. '
            'Horizontal positions are linearly interpolated from the unchanged 1-ms saved controller trajectories for native 60-frame-per-second playback. '
            'Camera direction, displayed body heading and roll are damped for visual clarity. These display filters do not alter horizontal coordinates or reported displacement and straightness. '
            'Fly geometry, camera field of view, camera distance and world scale are identical across panels. '
            'The model determines horizontal motion only; altitude, vegetation, fly appearance and wingbeats are illustrative, with flies enlarged for visibility.')
    if movie==5:
        return ('Supplementary Movie 5 | Three-dimensional visualization of modeled navigation after a focal lesion. '
                'A 2,000-neuron lesion at site 20 is shown without replacement (left) and with 100 lost neurons selected by prioritizing visual-integration and descending cells, retaining their original connections (right). '
                'Orange points identify the same lost neurons and green points the smaller replacement. '+common)
    return ('Supplementary Movie 8 | Three-dimensional navigation after lesions of different size and location. '
            'Rows show removal of 1,000, 4,000 and 16,000 neurons. Columns compare site 20 in the posterior central brain and site 25 in the optic lobe. '
            'CNS insets show the actual affected neurons in orange. Navigation trails and the coloured path insets correspond to each lesion. '
            'Blue inset paths show the population-mean intact reference used in Supplementary Movie 1, with gain_R=gain_L=drive=1 and bias=0, rather than a matched intact neural seed. '
            'The path inset uses one common spatial scale across all six conditions. '+common)


def render(movie,preview_only=False,clip_preview=False):
    OUT.mkdir(parents=True,exist_ok=True)
    if movie==5:
        payloads,records,hashes=base.current_payloads()
        rows,cols,width,height,stem=1,2,2560,720,MOVIE5
    else:
        payloads,records,hashes=movie8_payloads()
        rows,cols,width,height,stem=3,2,2160,1920,MOVIE8
        xy=np.concatenate([np.asarray(p['traj'])[:,3:5] for p in payloads]+[np.asarray(payloads[0]['reference'])[:,3:5]])
        xmin,xmax=float(xy[:,1].min())-.12,float(xy[:,1].max())+.12
        ymin,ymax=float(xy[:,0].min())-.12,float(xy[:,0].max())+.12
        for p in payloads:p['reference_bounds']=[ymin,ymax,xmin,xmax]
    payloads=[prepare_display(p) for p in payloads]
    qa=OUT/f'qa_navigation3d_smooth_movie{movie}'
    qa.mkdir(exist_ok=True)
    frames=[0,180,360,599]
    with tempfile.TemporaryDirectory(prefix=f'fly-v7-3d-smooth-{movie}-',dir='/private/tmp') as temporary:
        temporary=Path(temporary)
        (temporary/'fly3d_v7.html').write_text(smooth_html(movie))
        cards=''.join(f'<iframe style="left:{100*(i%cols)/cols}%;top:{100*(i//cols)/rows}%;width:{100/cols}%;height:{100/rows}%;" src="fly3d_v7.html"></iframe>' for i in range(len(payloads)))
        wrapper='<!doctype html><html><head><style>html,body{margin:0;overflow:hidden;background:#f4f7fb}iframe{position:absolute;border:1px solid #d4dde4;box-sizing:border-box}</style></head><body>'+cards+'</body></html>'
        (temporary/'comparison.html').write_text(wrapper)
        (temporary/'render.cjs').write_text(NODE_DRIVER)
        cfg=dict(movie=movie,width=width,height=height,panelWidth=width//cols,panelHeight=height//rows,
                 fps=FPS,duration=DURATION,payloads=payloads,previewOnly=preview_only,clipPreview=clip_preview,
                 wrapperUrl=(temporary/'comparison.html').as_uri(),output=str((qa/f'{stem}_smooth_clip.mp4') if clip_preview else (OUT/f'{stem}.mp4')),
                 ffmpeg=base.FFMPEG,previewFrames=frames,previewPrefix=str(qa/stem),
                 browserQa=str(qa/('preview_browser_qa.json' if preview_only else ('clip_browser_qa.json' if clip_preview else 'browser_qa.json'))),
                 chromium='/Users/ruslanrust/Library/Caches/ms-playwright/chromium-1217/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing')
        (temporary/'config.json').write_text(json.dumps(cfg))
        subprocess.run([str(base.NODE),str(temporary/'render.cjs'),base.PLAYWRIGHT,str(temporary/'config.json')],check=True)
    if clip_preview:
        clip=qa/f'{stem}_smooth_clip.mp4'
        subprocess.run([base.FFMPEG,'-v','error','-i',str(clip),'-f','null','-'],check=True)
        print('CLIP_PREVIEW',clip)
        return
    thumbw,thumbh=width//2,height//2
    sheet=Image.new('RGB',(thumbw*2,thumbh*2),'white')
    for i,n in enumerate(frames):
        im=Image.open(qa/f'{stem}_frame_{n:03d}.png').convert('RGB');im.thumbnail((thumbw,thumbh),Image.Resampling.LANCZOS)
        sheet.paste(im,((i%2)*thumbw,(i//2)*thumbh))
    sheet.save(qa/f'{stem}_preview_contact_sheet.png')
    if preview_only:
        print('PREVIEW',qa)
        return
    shutil.copy2(qa/f'{stem}_frame_360.png',OUT/f'{stem}_preview.png')
    shutil.copy2(qa/f'{stem}_preview_contact_sheet.png',OUT/f'{stem}_contact_sheet.png')
    output=OUT/f'{stem}.mp4'
    probe=json.loads(subprocess.check_output([base.FFPROBE,'-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,duration,nb_frames','-of','json',str(output)]))
    subprocess.run([base.FFMPEG,'-v','error','-threads','2','-i',str(output),'-f','null','-'],check=True)
    qa_browser=json.loads((qa/'browser_qa.json').read_text())
    assert qa_browser['errors']==[] and qa_browser['deterministic_call_order']
    assert qa_browser['duplicateFrames']==0
    stream=probe['streams'][0]
    assert int(stream['nb_frames'])==FPS*DURATION and stream['r_frame_rate']=='60/1'
    metrics=[]
    for i,(p,r) in enumerate(zip(payloads,records)):
        tr=np.asarray(p['traj'])
        sig=[sample['conditions'][i] for sample in qa_browser['samples']]
        modeltime=np.array([s['model_time'] for s in sig])
        actual=np.array([[s['model_x'],s['model_y']] for s in sig])
        expected=np.column_stack([np.interp(modeltime,tr[:,0],tr[:,column]) for column in [3,4]])
        maximum_position_error=float(np.max(np.abs(actual-expected)))
        assert maximum_position_error<1e-12
        camera=np.array([s['camera_heading'] for s in sig])
        raw=np.array([s['raw_heading'] for s in sig])
        distances=np.array([s['camera_distance'] for s in sig])
        assert np.ptp(distances)<1e-10
        assert all(s['fly_scale']==[1,1,1] and s['field_of_view']==38 for s in sig)
        metrics.append(dict(condition=r['source_condition'],maximum_horizontal_interpolation_error_m=maximum_position_error,
                            camera_distance_min=float(distances.min()),camera_distance_max=float(distances.max()),
                            maximum_raw_heading_step_deg=float(np.max(np.abs(np.diff(raw)))*180/np.pi),
                            maximum_camera_heading_step_deg=float(np.max(np.abs(np.diff(camera)))*180/np.pi),
                            p95_raw_heading_step_deg=float(np.quantile(np.abs(np.diff(raw)),.95)*180/np.pi),
                            p95_camera_heading_step_deg=float(np.quantile(np.abs(np.diff(camera)),.95)*180/np.pi)))
    text=caption(movie)
    (OUT/f'{stem}_caption.txt').write_text(text+'\n')
    provenance=dict(status='STRUCTURAL_PASS_PENDING_VISUAL_REVIEW',movie=str(output.relative_to(ROOT)),caption=text,
                    source_data_version='Frozen corrected V5 data used throughout V7',conditions=records,source_hashes=hashes,
                    output_sha256=base.sha(output),renderer_source_hashes={str(p.relative_to(ROOT)):base.sha(p) for p in [Path(__file__),ROOT/'code/v7/render_navigation3d.py',ROOT/'code/web/fly3d.html',ROOT/'code/web/three.min.js']},
                    display_filters=TAUS,display_filter_definition='Causal exponential filters precomputed at1ms; no screenshot-call dependence',
                    interpolation='Linear interpolation of recorded horizontal positions, raw headings and precomputed visual variables at exact output timestamps',
                    no_neural_or_controller_resimulation=True,quantitative_position_smoothing=False,
                    same_geometry_camera_scale=True,horizontal_world_units='25 scene units per metre',
                    display_altitude='4 +0.15sin(3.1t) arbitrary scene units; illustrative only',
                    decorative_wingbeat_frequency_hz=11,wing_blur='Three wing copies with0.35rad phase spacing; illustrative',
                    native_fps=FPS,frame_timing='600 equally spaced frames,1/60s apart; no optical-flow frame interpolation',
                    browser_errors=qa_browser['errors'],deterministic_render_order=qa_browser['deterministic_call_order'],
                    consecutive_duplicate_encoded_source_frames=qa_browser['duplicateFrames'],motion_qa=metrics,
                    browser_qa_file=str((qa/'browser_qa.json').relative_to(ROOT)),video_probe=probe,decode_check='PASS',
                    choppiness_correction={'prior_movie5_fps':24,'original_historical_movie_fps':30,
                                           'prior_camera':'Rigid orbit followed raw recorded heading at every frame',
                                           'corrected_camera':'Fixed-distance orbit follows exponentially damped heading; body motion and measured positions remain distinct',
                                           'framing':'Movie5 restores original1280x720 wide panel proportions'},
                    intact_reference=({'source':'videos/v5/Movie1_v5_lesion_location_size_intact_trajectory.csv','definition':'Population-mean gain_R=gain_L=drive=1,bias=0','common_map_bounds':payloads[0]['reference_bounds']} if movie==8 else 'No intact fly/reference path in this two-condition film'))
    provenance['outputs']={str(p.relative_to(ROOT)):base.sha(p) for p in [output,OUT/f'{stem}_preview.png',OUT/f'{stem}_contact_sheet.png',OUT/f'{stem}_caption.txt']}
    (OUT/f'{stem}_provenance.json').write_text(json.dumps(provenance,indent=2))
    print('COMPLETE',output)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--movie',type=int,choices=[5,8],default=5)
    parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--clip-preview',action='store_true')
    args=parser.parse_args()
    render(args.movie,args.preview_only,args.clip_preview)


if __name__=='__main__':main()
