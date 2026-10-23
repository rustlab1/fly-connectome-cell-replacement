"""Retain original seconds 15 to 24 of the verified activity film."""
from pathlib import Path
import hashlib,json,subprocess,shutil
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'videos/v8/publication_movies_20261012'
ARCHIVE=ROOT/'videos/v8/archive/before_movie1_trim_20261012'
STEM='Supplementary_Movie_2_neuronal_responses'
SOURCE_STEM='Supplementary_Movie_1_neuronal_responses'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def trim():
    source=ARCHIVE/(SOURCE_STEM+'.mp4');movie=OUT/(STEM+'.mp4')
    qa=OUT/'qa_movie2_activity';qa.mkdir(exist_ok=True)
    assert source.exists()
    original=json.loads((ARCHIVE/(SOURCE_STEM+'_provenance.json')).read_text())
    assert sha(source)==original['output_sha256']
    temporary=qa/'activity_excerpt.mp4'
    subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-loglevel','error','-i',str(source),
        '-vf','trim=start_frame=450:end_frame=720,setpts=PTS-STARTPTS','-an','-c:v','libx264',
        '-threads','3','-preset','medium','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],check=True)
    probe=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate,duration,nb_frames','-of','json',str(temporary)]))['streams'][0]
    assert probe['width']==1920 and probe['height']==1080 and probe['nb_frames']=='270'
    assert probe['r_frame_rate']=='30/1' and float(probe['duration'])==9
    subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-i',str(temporary),'-f','null','-'],check=True)
    shutil.move(temporary,movie)
    frames=[]
    for i,t in enumerate([0,3,6,8.966]):
        path=qa/f'excerpt_frame_{i}.png'
        subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-loglevel','error','-ss',str(t),'-i',str(movie),'-frames:v','1',str(path)],check=True)
        frames.append(path)
    shutil.copy2(frames[1],OUT/(STEM+'_preview.png'))
    sheet=Image.new('RGB',(1920,1080),'white')
    for i,path in enumerate(frames):
        image=Image.open(path).convert('RGB');image.thumbnail((960,540),Image.Resampling.LANCZOS)
        sheet.paste(image,(i%2*960,i//2*540))
    sheet.save(OUT/(STEM+'_contact_sheet.png'))
    original.update(number=2,movie=str(movie.relative_to(ROOT)),status='STRUCTURAL_PASS_PENDING_VISUAL_REVIEW',title='Neuronal responses to looming',
        output_sha256=sha(movie),video_probe=probe,frame_count=270,model_time_ms=[59,300],
        chapters=[dict(start_s=0,duration_s=9,stimulus='looming')])
    original['source_hashes'][str(source.relative_to(ROOT))]=sha(source)
    frozen=ROOT/'videos/v8/archive/before_movie_renumbering_20261012/renderer_sources'
    original['renderer_source_hashes']={
        (str((frozen/Path(relative).name).relative_to(ROOT)) if sha(ROOT/relative)!=digest else relative):digest
        for relative,digest in original['renderer_source_hashes'].items()}
    original['renderer_source_hashes'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    original['retained_clip']=dict(source_movie=str(source.relative_to(ROOT)),source_sha256=sha(source),
        original_start_s=15,original_end_s=24,start_frame=450,end_frame_exclusive=720,
        retained_frames=270,model_time_ms=[59,300],neural_or_controller_resimulation=False)
    original['outputs']={str(p.relative_to(ROOT)):sha(p) for p in [movie,OUT/(STEM+'_preview.png'),OUT/'Movie2_trial_summaries.csv',OUT/'Movie2_looming_motor_trace.csv'] if p.exists()}
    (OUT/(STEM+'_provenance.json')).write_text(json.dumps(original,indent=2)+'\n')
    print('Movie 2: exactly 9 seconds, original frames 450 through 719, 30 fps.')

if __name__=='__main__':trim()
