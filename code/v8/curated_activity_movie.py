"""One paper-style activity film, with matched population readouts and spike QA."""
from pathlib import Path
import argparse, json, subprocess, shutil
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from curated_movies import ROOT,OUT,STEMS,TITLES,COLORS,sha,verify_video,contact_sheet,FFMPEG

STATES=['intact','lesion20_2000','selection_ranked_100']
NAMES=['Intact','No graft','Original repair']
STATECOLORS=['#707b85',COLORS[0],COLORS[1]]
HOST,INPUT,GRAFT,MOTOR,LOST='#3b7dab','#855bb8',COLORS[1],COLORS[0],'#df9bb0'
FPS=30;HOLD=30;TOTAL=722

def trace(times,cells,group):
    counts,_=np.histogram(times[np.isin(cells,group)],bins=np.arange(301))
    cumulative=np.r_[0,np.cumsum(counts)];i=np.arange(301);prior=np.maximum(i-25,0)
    return (cumulative[i]-cumulative[prior])/np.maximum(np.minimum(i,25),1)/.001/len(group)

def render(preview=False):
    recording=ROOT/'videos/v8/activity_current_model_recordings.npz'
    original_manifest=ROOT/'videos/v8/activity_current_model_manifest.json'
    q=json.loads(original_manifest.read_text())
    assert sha(recording)==q['recording_sha256']
    for rel,value in q['source_hashes'].items():assert sha(ROOT/rel)==value,rel
    with np.load(recording) as z:data={k:z[k] for k in z.files}
    neu=pd.read_parquet(ROOT/'data/v5/neurons.parquet');groups=json.loads((ROOT/'data/v5/cell_groups.json').read_text())
    decoded=np.load(ROOT/'data/v5/decoder.npz');motor_R=decoded['steer_idx']
    ttm=np.flatnonzero(neu.type.eq('TTMn'));dlm=np.flatnonzero(neu.type.astype(str).str.startswith('DLMn'))
    assert len(ttm)==2 and len(dlm)==10
    inputs={'R':np.unique(np.concatenate([groups[x] for x in ['T4a_R','T5a_R','T4b_L','T5b_L']])),
            'loom':np.flatnonzero(neu.type.isin(['LC4','LPLC2']))}
    refs=json.loads((ROOT/'results/v8/movie_neural_states.json').read_text());n=int(data['host_count'])
    xyz=data['xyz_um'].copy();xyz[:,2]=np.where(xyz[:,2]>450,xyz[:,2]-90,xyz[:,2]);xy=xyz[:,[0,2]];xy[:,1]*=-1
    known=np.isfinite(xy).all(axis=1);lost=data['lesion_idx'];templates=data['graft_template_idx']
    allxy=[xy,xy,np.concatenate([xy,xy[templates]])];visible=[known,known,np.r_[known,known[templates]]]
    plt.rcParams.update({'font.family':'Arial','font.size':13,'text.color':'#24313a','axes.labelcolor':'#24313a',
        'xtick.color':'#44515b','ytick.color':'#44515b','axes.edgecolor':'#9ba9b5'})
    video=OUT/f'{STEMS[2]}.mp4';qa=OUT/'qa_movie2_activity';qa.mkdir(exist_ok=True)
    command=[FFMPEG,'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-s','1920x1080','-pix_fmt','rgba','-r',str(FPS),'-i','-',
        '-an','-c:v','libx264','-threads','3','-crf','17','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE) if not preview else None
    sourcechecks=[];summaries=[];trace_exports=[];snapshots=[];written=0
    for chapter,stim in enumerate(['R','loom']):
        motor= motor_R if stim=='R' else np.r_[ttm,dlm]
        series=[(data[f'{s}_{stim}_t'],data[f'{s}_{stim}_i']) for s in STATES]
        curves=[.5*(trace(t,c,ttm)+trace(t,c,dlm)) for t,c in series] if stim=='loom' else []
        for state,(times,cells) in zip(STATES,series):
            keep=(times>=50)&(times<300);rates=np.bincount(cells[keep],minlength=len(data[f'{state}_{stim}_rates']))/.25
            np.testing.assert_array_equal(rates.astype('float32'),data[f'{state}_{stim}_rates'])
            if state!='intact':assert not np.isin(cells[cells<n],lost).any()
            if stim=='loom':
                value=.5*(rates[ttm].mean()+rates[dlm].mean())
                np.testing.assert_allclose(value,.5*(refs[state]['TTMn_loom']+refs[state]['DLMn_loom']),rtol=0,atol=2e-6)
                summaries.append(dict(stimulus=stim,condition=state,looming_motor_rate_hz=float(value)))
            else:
                summaries.append(dict(stimulus=stim,condition=state,right_response_normalized=refs[state]['gain_R']/refs['intact']['gain_R'],
                    left_response_normalized=refs[state]['gain_L']/refs['intact']['gain_L']))
            sourcechecks.append(dict(condition=state,stimulus=stim,spike_rates_match=True,removed_cell_spikes=0 if state!='intact' else None))
        fig=plt.figure(figsize=(16,9),dpi=120,facecolor='white')
        title='Rightward motion' if stim=='R' else 'Looming input'
        fig.text(.028,.952,TITLES[2],fontsize=25,weight='bold')
        fig.text(.028,.906,title,fontsize=18,color=INPUT)
        clock=fig.text(.972,.938,'0 ms',ha='right',fontsize=22,animated=True)
        fig.text(.972,.902,'300 ms of model time; slowed playback',ha='right',fontsize=13)
        dynamic=[clock];host_points=[];graft_points=[];motor_points=[];lines=[];cursors=[];values=[]
        for col,(state,label,color) in enumerate(zip(STATES,NAMES,STATECOLORS)):
            left=.035+col*.326
            fig.text(left,.858,label,fontsize=19,weight='bold',color=color)
            fig.text(left,.825,['No lesion','2,000 neurons lost','2,000 lost; 100 replacements'][col],fontsize=13)
            ax=fig.add_axes([left,.28,.294,.53]);ax.set_xlim(0,770);ax.set_ylim(-1010,-45);ax.set_aspect('equal');ax.axis('off')
            bg=known.copy();
            if col:bg[lost]=False
            bgix=np.flatnonzero(bg);bgix=np.random.default_rng(0).choice(bgix,min(35000,len(bgix)),replace=False)
            ax.scatter(*xy[bgix].T,s=.30,color='#bac7d1',alpha=.28,linewidths=0)
            if col:ax.scatter(*xy[lost[known[lost]]].T,s=1.3,color=LOST,alpha=.5,linewidths=0)
            if col==2:ax.scatter(*xy[templates[known[templates]]].T,s=3,color=GRAFT,alpha=.45,linewidths=0)
            ax.text(.02,.98,'L',transform=ax.transAxes,fontsize=11,color='#44515b');ax.text(.95,.98,'R',transform=ax.transAxes,fontsize=11,color='#44515b')
            host_points.append(ax.scatter([],[],s=[],linewidths=0,animated=True))
            graft_points.append(ax.scatter([],[],s=[],color=GRAFT,linewidths=0,animated=True))
            motor_points.append(ax.scatter([],[],s=[],color=MOTOR,linewidths=0,animated=True))
            ax2=fig.add_axes([left+.017,.114,.276,.128]);ax2.spines[['top','right']].set_visible(False);ax2.tick_params(labelsize=12,length=3)
            if stim=='R':
                vals=[refs[state]['gain_R']/refs['intact']['gain_R'],refs[state]['gain_L']/refs['intact']['gain_L']]
                ax2.bar([0,1],vals,color=['#3b7dab','#855bb8'],width=.46)
                ax2.axhline(1,color='#8c969e',linestyle=':',lw=1);ax2.set_ylim(0,2.1);ax2.set_yticks([0,1,2])
                ax2.set_xticks([0,1],['Rightward','Leftward']);ax2.set_ylabel('Response / intact',fontsize=12)
                for j,v in enumerate(vals):ax2.text(j,v+.07,f'{v:.2f}',ha='center',fontsize=12)
                ax2.set_title('Directional trial summaries',loc='left',fontsize=13,pad=8)
            else:
                ymax=np.ceil(max(v.max() for v in curves)/20)*20
                ax2.set_xlim(0,300);ax2.set_ylim(0,ymax);ax2.set_xticks([0,100,200,300]);ax2.set_yticks([0,ymax/2,ymax])
                ax2.set_xlabel('Model time (ms)',fontsize=12);ax2.set_ylabel('Rate (Hz)',fontsize=12)
                ax2.set_title('Looming motor activity',loc='left',fontsize=13,pad=8)
                line,=ax2.plot([],[],c=color,lw=1.8,animated=True);cursor=ax2.axvline(0,c='#8c969e',lw=.7,animated=True)
                value=ax2.text(.99,1.05,'0 Hz',ha='right',transform=ax2.transAxes,fontsize=13,color=color,animated=True)
                lines.append(line);cursors.append(cursor);values.append(value);dynamic.extend([line,cursor,value])
        dynamic+=host_points+graft_points+motor_points
        handles=[Line2D([],[],marker='o',ls='',color=c,markersize=5,label=l) for c,l in [(INPUT,'Input'),(HOST,'Host activity'),(GRAFT,'Replacement'),(LOST,'Lesion'),(MOTOR,'Motor activity')]]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.016),ncol=5,frameon=False,fontsize=13)
        fig.canvas.draw();background=fig.canvas.copy_from_bbox(fig.bbox)
        recent=[np.full(n+(100 if k==2 else 0),-1e6) for k in range(3)];pointers=[0]*3
        rgb=np.tile(to_rgb(HOST),(n,1));rgb[inputs[stim]]=to_rgb(INPUT)
        for frame in range(301):
            fig.canvas.restore_region(background);clock.set_text(f'{frame} ms')
            for k,(times,cells) in enumerate(series):
                stop=np.searchsorted(times,frame,side='right');np.maximum.at(recent[k],cells[pointers[k]:stop],times[pointers[k]:stop]);pointers[k]=stop
                age=frame-recent[k];show=visible[k]&(age>=0)&(age<=32)
                host=np.flatnonzero(show[:n]);weight=np.exp(-age[host]/8)
                host_points[k].set_offsets(allxy[k][host]);host_points[k].set_sizes(10*weight+.2)
                host_points[k].set_facecolors(np.c_[rgb[host],np.clip(weight,.02,1)])
                extra=np.flatnonzero(show[n:])+n;gw=np.exp(-age[extra]/8)
                graft_points[k].set_offsets(allxy[k][extra]);graft_points[k].set_sizes(22*gw+1)
                graft_points[k].set_alpha(np.clip(gw,.02,1) if len(gw) else 0.)
                mi=host[np.isin(host,motor)];mw=np.exp(-age[mi]/8)
                motor_points[k].set_offsets(allxy[k][mi]);motor_points[k].set_sizes(25*mw+2)
                motor_points[k].set_alpha(np.clip(mw,.03,1) if len(mw) else 0.)
                if stim=='loom':
                    lines[k].set_data(np.arange(frame+1),curves[k][:frame+1]);cursors[k].set_xdata([frame,frame]);values[k].set_text(f'{curves[k][frame]:.0f} Hz')
            for artist in dynamic:fig.draw_artist(artist)
            rgba=np.asarray(fig.canvas.buffer_rgba()).copy()
            if frame in [80,299]:
                path=qa/f'{STEMS[2]}_{stim}_frame_{frame:03d}.png';Image.fromarray(rgba).save(path);snapshots.append(path)
            if encoder:
                if frame==0:
                    for _ in range(HOLD):encoder.stdin.write(rgba.tobytes());written+=1
                encoder.stdin.write(rgba.tobytes());written+=1
                if frame==300:
                    for _ in range(HOLD):encoder.stdin.write(rgba.tobytes());written+=1
            if frame%100==0:print('Movie 2',stim,frame,flush=True)
        plt.close(fig)
        if stim=='loom':
            for state,curve in zip(STATES,curves):trace_exports.extend(dict(condition=state,t_ms=i,looming_motor_rate_hz=float(v)) for i,v in enumerate(curve))
    if encoder:
        encoder.stdin.close();assert encoder.wait()==0 and written==TOTAL
    contact_sheet(snapshots,OUT/f'{STEMS[2]}_contact_sheet.png');shutil.copy2(snapshots[2],OUT/f'{STEMS[2]}_preview.png')
    if preview:return
    pd.DataFrame(summaries).to_csv(OUT/'Movie2_trial_summaries.csv',index=False)
    pd.DataFrame(trace_exports).to_csv(OUT/'Movie2_looming_motor_trace.csv',index=False)
    stream=verify_video(video,TOTAL,FPS)
    sources=[recording,original_manifest,ROOT/'results/v8/movie_neural_states.json',ROOT/'data/v5/neurons.parquet',ROOT/'data/v5/cell_groups.json',ROOT/'data/v5/decoder.npz']
    record=dict(status='STRUCTURAL_PASS_PENDING_VISUAL_REVIEW',number=2,title=TITLES[2],movie=str(video.relative_to(ROOT)),output_sha256=sha(video),
        video_probe=stream,decode_check='PASS',frame_count=written,native_fps=FPS,no_neural_or_controller_resimulation=True,
        neural_seed=1000,intervention_seed=2000,conditions=STATES,spike_recording_checks=sourcechecks,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},renderer_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'code/v8/curated_movies.py']},
        chapters=[dict(start_s=0,duration_s=361/30,stimulus='rightward motion'),dict(start_s=361/30,duration_s=361/30,stimulus='looming')],
        model_time_ms=[0,300],measurement_window_ms=[50,300],display_persistence_tau_ms=8,display_persistence_cutoff_ms=32,
        activity_marker_geometry_identical=True,anatomy_scale_identical=True,unknown_somata_hidden_from_view_only=True,
        rightward_summary='Static matched-intact-normalized directional responses from separate R, L and F trials; not an instantaneous firing trace',
        looming_readout='Arithmetic mean of the TTMn (2 cells) and DLMn (10 cells) population mean firing rates over the preceding 25 ms',
        trial_summaries=summaries,outputs={str(p.relative_to(ROOT)):sha(p) for p in [video,OUT/f'{STEMS[2]}_preview.png',OUT/'Movie2_trial_summaries.csv',OUT/'Movie2_looming_motor_trace.csv']})
    (OUT/f'{STEMS[2]}_provenance.json').write_text(json.dumps(record,indent=2)+'\n');print('COMPLETE',video,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');a=p.parse_args();render(a.preview)
