"""Restore the original 3D fly style using the frozen trajectories used in V7.

The movie is a rendering of saved horizontal controller coordinates, not a
new flight or neural simulation. Original movies and renderer are preserved.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'videos/v8'
NODE = Path('/Users/ruslanrust/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
PLAYWRIGHT = '/Users/ruslanrust/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'
FFMPEG = '/opt/homebrew/bin/ffmpeg'
FFPROBE = '/opt/homebrew/bin/ffprobe'
STEM = 'Supplementary_Movie_5_3D_navigation_current_data'
FPS, DURATION, WIDTH, HEIGHT = 24, 10, 1920, 864


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cns_payload(neurons, lesion, templates=None):
    xyz = neurons[['x_um', 'y_um', 'z_um']].to_numpy().copy()
    xyz[:, 2] = np.where(xyz[:, 2] > 450, xyz[:, 2] - 90, xyz[:, 2])
    has = np.isfinite(xyz).all(axis=1)
    available = np.flatnonzero(has)
    lesion = np.asarray(lesion, dtype=int)
    loss = lesion[has[lesion]]
    other = available[~np.isin(available, lesion)]
    sample = np.random.default_rng(0).choice(other, min(35000, len(other)), replace=False)
    selected = np.concatenate([sample, loss])
    center = np.mean(xyz[has], axis=0)
    radius = np.max(np.linalg.norm(xyz[has] - center, axis=1)) * .75
    graft = np.zeros((0, 3)) if templates is None else xyz[np.asarray(templates, dtype=int)]
    graft = graft[np.isfinite(graft).all(axis=1)]
    return dict(xyz=xyz[selected].ravel().tolist(),
                lesion=(np.arange(len(selected)) >= len(sample)).tolist(),
                center=center.tolist(), radius=float(radius), graft=graft.ravel().tolist())


def current_payloads():
    manifest_path = ROOT / 'videos/v5/Movie2_replacement_identity_conditions.json'
    manifest = json.loads(manifest_path.read_text())
    neurons_path = ROOT / 'data/processed/neurons.parquet'
    neurons = pd.read_parquet(neurons_path)
    states_path = ROOT / 'results/v5/tables/states.csv'
    states = pd.read_csv(states_path)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in [manifest_path, neurons_path, states_path]}
    payloads, condition_records = [], []
    for name, title, color in [('size_site20_n2000', 'No replacement', '#e45f41'),
                               ('partial_ranked_p5', '100 selected neurons', '#159979')]:
        condition = next(c for c in manifest['scenes'] if c['id'] == name)
        state = states[(states['name'] == name) & (states.seed == 0)].iloc[0]
        for key in ['gain_R', 'gain_L', 'bias', 'drive']:
            assert abs(float(state[key]) - float(condition[key])) < 1e-12
        trajectory_path = ROOT / f'videos/v5/Movie2_v5_replacement_identity_{name}_trajectory.csv'
        trajectory = pd.read_csv(trajectory_path).to_numpy(dtype=float)
        assert trajectory.shape == (10000, 5)
        assert np.isfinite(trajectory).all()
        assert abs(trajectory[-1, 0] - 9.999) < 1e-8
        lesion_path = ROOT / condition['lesion_idx_file']
        lesion = np.load(lesion_path)
        assert len(lesion) == 2000
        template_path = ROOT / condition['graft_template_idx_file'] if condition.get('graft_template_idx_file') else None
        templates = np.load(template_path) if template_path else None
        if templates is not None:
            assert len(templates) == 100
            assert np.isin(templates, lesion).all()
        for source in [trajectory_path, lesion_path] + ([template_path] if template_path else []):
            hashes[str(source.relative_to(ROOT))] = sha(source)
        n_replaced = 0 if templates is None else len(templates)
        payloads.append(dict(traj=trajectory.tolist(), fps=FPS, title=title,
                             subtitle=f'2,000 neurons lost · {n_replaced} replaced',
                             color=color, tier_label='', tier_color=color,
                             stats_html='', inset_label='Lesion' if not n_replaced else 'Lesion / replacement',
                             cns=cns_payload(neurons, lesion, templates),
                             fall=None, graft_size=4.0,
                             metrics={key:float(state[key]) for key in ['gain_R','gain_L','bias','drive']}))
        net = float(np.linalg.norm(trajectory[-1, 3:5] - trajectory[0, 3:5]))
        length = float(np.linalg.norm(np.diff(trajectory[:, 3:5], axis=0), axis=1).sum())
        condition_records.append(dict(source_condition=name, neural_seed=0, gust_seed=3,
                                      n_lesion=2000, n_replacement=n_replaced,
                                      trajectory_file=str(trajectory_path.relative_to(ROOT)),
                                      trajectory_points=len(trajectory),
                                      final_net_displacement_m=net,
                                      final_path_length_m=length,
                                      final_straightness=net/length,
                                      **{key:float(state[key]) for key in ['gain_R','gain_L','bias','drive']}))
    return payloads, condition_records, hashes


def render_html():
    source = (ROOT / 'code/web/fly3d.html').read_text()
    source = source.replace('src="three.min.js"', f'src="{(ROOT / "code/web/three.min.js").as_uri()}"')
    source = source.replace('font-size: 30px', 'font-size: 31px')
    source = source.replace('left: 28px; top: 62px; font-size: 17px', 'left: 28px; top: 64px; font-size: 20px')
    source = source.replace('left: 28px; bottom: 26px; font-size: 16px', 'left: 28px; bottom: 54px; font-size: 20px')
    source = source.replace('right: 28px; bottom: 26px; font-size: 16px', 'right: 28px; bottom: 54px; font-size: 20px')
    source = source.replace('#tier {', '#tier { display:none;')
    source = source.replace('#stats {', '#stats { background:rgba(255,255,255,.90); padding:10px 14px; border-radius:5px;')
    source = source.replace('#time {', '#time { background:rgba(255,255,255,.90); padding:10px 14px; border-radius:5px;')
    source = source.replace('buildLandmarks(mx, mz, 70);', 'buildLandmarks(0, 0, 70);')
    # Preserve the visible breadcrumb size used in the original 32k illustration.
    # A fixed camera offset in fly-relative coordinates gives equal display scale
    # in both panels. Original chase-camera smoothing changed apparent fly size.
    begin = source.index('  if (!camInit) {')
    end = source.index('  scene.userData.sun.position', begin)
    source = source[:begin] + '''  camera.position.copy(target);
  camera.lookAt(new THREE.Vector3(X, alt, Z));
''' + source[end:]
    source = source.replace('const iw = Math.round(W * 0.27)', 'const iw = Math.round(W * 0.30)')
    source = source.replace("document.getElementById('stats').innerHTML = DATA.stats_html + `path straightness so far: <b>${straight.toFixed(2)}</b>`;", """const displacement = Math.hypot(x-tr[0][3], y-tr[0][4]);
  document.getElementById('stats').innerHTML = `Distance from start <b>${displacement.toFixed(2)} m</b><br>Path straightness <b>${straight.toFixed(2)}</b>`;""")
    # Display the actual first-saved-to-current path measure used by V7 figures.
    source = source.replace('Math.hypot(tr[i][3] - tr[0][3], tr[i][4] - tr[0][4])', 'Math.hypot(tr[i][3] - tr[0][3], tr[i][4] - tr[0][4])')
    return source


NODE_RENDER = r'''
const fs = require('fs');
const { chromium } = require(process.argv[2]);
const { spawn } = require('child_process');
const { once } = require('events');
(async () => {
  const config = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
  const browser = await chromium.launch({headless:true, executablePath:config.chromium, args:[
    '--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader',
    '--ignore-gpu-blocklist','--allow-file-access-from-files']});
  const page = await browser.newPage({viewport:{width:config.width,height:config.height}, deviceScaleFactor:1});
  const errors=[];
  page.on('pageerror', e=>errors.push(e.message));
  await page.goto(config.wrapperUrl, {waitUntil:'load'});
  await page.waitForFunction(() => Array.from(document.querySelectorAll('iframe')).every(f => typeof f.contentWindow.init === 'function'));
  await page.evaluate(d => {
    const frames = Array.from(document.querySelectorAll('iframe'));
    frames.forEach((f,i) => {f.contentWindow.init(d.width/2,d.height);f.contentWindow.setData(d.payloads[i]);});
  }, config);
  const encoder = spawn(config.ffmpeg,['-y','-loglevel','error','-f','image2pipe','-vcodec','png',
    '-framerate',String(config.fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','medium',
    '-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',config.output]);
  let encodingErrors='';encoder.stderr.on('data', d => {encodingErrors += d;});
  for(let n=0;n<config.fps*config.duration;n++) {
    const i=Math.round(n*1000/config.fps);
    await page.evaluate(i => {document.querySelectorAll('iframe').forEach(f=>f.contentWindow.renderFrame(i));}, i);
    const buf=await page.screenshot({type:'png'});
    if(config.previewFrames.includes(n)) fs.writeFileSync(config.previewPrefix+'_frame_'+String(n).padStart(3,'0')+'.png',buf);
    if(!encoder.stdin.write(buf)) await once(encoder.stdin,'drain');
    if(n%48===0) process.stdout.write(`Rendered ${n}/${config.fps*config.duration} frames\n`);
  }
  encoder.stdin.end();
  const [code] = await once(encoder,'close');
  if(code!==0) throw new Error(encodingErrors || `ffmpeg exited ${code}`);
  const scales=await page.evaluate(() => Array.from(document.querySelectorAll('iframe')).map(f=>{
    const w=f.contentWindow;
    return w.eval('({fly_scale:fly.scale.toArray(),camera_field_of_view:camera.fov,camera_offset_distance:camera.position.distanceTo(fly.position),altitude:fly.position.y})');
  }));
  fs.writeFileSync(config.browserQa,JSON.stringify({errors,scales,rendered_frames:config.fps*config.duration},null,2));
  await browser.close();
  if(errors.length) throw new Error(errors.join('\n'));
})().catch(e=>{process.stderr.write(String(e.stack||e));process.exit(1)});
'''


def restore_legacy():
    legacy = OUT / 'legacy'
    legacy.mkdir(exist_ok=True)
    source = ROOT / 'videos/Video3D_best_effect_32000_without_vs_with_therapy.mp4'
    target = legacy / source.name
    shutil.copy2(source, target)
    assert sha(source) == sha(target)
    preview = legacy / (source.stem + '_preview.png')
    subprocess.run([FFMPEG, '-y', '-v', 'error', '-ss', '5', '-i', str(target), '-frames:v', '1', str(preview)], check=True)
    note = {
        'status':'HISTORICAL_ILLUSTRATION_NOT_CURRENT_V7_DATA',
        'source':str(source.relative_to(ROOT)),
        'file':str(target.relative_to(ROOT)),
        'sha256':sha(target),
        'original_bytes_preserved':True,
        'caption':'Original 32,000-neuron comparison, retained as a historical visual example. It uses an earlier model and includes illustrative falling motion. Its numerical labels and outcome classifications are superseded by the corrected V5 data used in V7. No matched 32,000-cell replacement experiment was run in the corrected V5 batch. This film is excluded from the current supplementary results.',
        'current_32k_lesion_only_condition':'size_site20_n32000',
        'original_condition_source':'results/video3d_besteffect_conditions.json',
        'original_condition_sha256':sha(ROOT / 'results/video3d_besteffect_conditions.json'),
    }
    (legacy / 'Video3D_best_effect_32000_provenance.json').write_text(json.dumps(note, indent=2))
    return note


def contact_sheet(preview_prefix, frames):
    thumb_w, thumb_h = 960, 432
    sheet = Image.new('RGB', (thumb_w*2, thumb_h*2), 'white')
    for k, frame in enumerate(frames):
        im = Image.open(f'{preview_prefix}_frame_{frame:03d}.png').convert('RGB')
        im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(im, ((k % 2)*thumb_w, (k // 2)*thumb_h))
    sheet.save(OUT / f'{STEM}_contact_sheet.png')
    shutil.copy2(f'{preview_prefix}_frame_144.png', OUT / f'{STEM}_preview.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qa_dir = OUT / 'qa_navigation3d'
    qa_dir.mkdir(exist_ok=True)
    payloads, records, source_hashes = current_payloads()
    legacy = restore_legacy()
    previews = [0, 72, 144, 239]
    with tempfile.TemporaryDirectory(prefix='fly-v7-3d-', dir='/private/tmp') as tmp:
        tmp = Path(tmp)
        (tmp / 'fly3d_v7.html').write_text(render_html())
        wrapper = '<!doctype html><html><head><style>html,body{margin:0;overflow:hidden;background:#f4f7fb;font-family:Helvetica,Arial,sans-serif}iframe{position:absolute;top:0;width:50%;height:100%;border:0}#right{left:50%;border-left:2px solid #d0d8df;box-sizing:border-box}.foot{position:absolute;bottom:0;left:0;width:100%;height:34px;background:rgba(255,255,255,.94);display:flex;align-items:center;justify-content:center;color:#596770;font-size:17px}</style></head><body><iframe id="left" src="fly3d_v7.html"></iframe><iframe id="right" src="fly3d_v7.html"></iframe><div class="foot">Matched gusts · 10 s model time · stylized fly enlarged for visibility</div></body></html>'
        (tmp / 'comparison.html').write_text(wrapper)
        (tmp / 'render.cjs').write_text(NODE_RENDER)
        config = dict(width=WIDTH, height=HEIGHT, fps=FPS, duration=DURATION,
                      wrapperUrl=(tmp / 'comparison.html').as_uri(), payloads=payloads,
                      output=str(OUT / f'{STEM}.mp4'), ffmpeg=FFMPEG,
                      chromium='/Users/ruslanrust/Library/Caches/ms-playwright/chromium-1217/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
                      previewFrames=previews, previewPrefix=str(qa_dir / STEM),
                      browserQa=str(qa_dir / 'browser_qa.json'))
        (tmp / 'config.json').write_text(json.dumps(config))
        subprocess.run([str(NODE), str(tmp / 'render.cjs'), PLAYWRIGHT, str(tmp / 'config.json')], check=True)
    contact_sheet(qa_dir / STEM, previews)
    output = OUT / f'{STEM}.mp4'
    probe = json.loads(subprocess.check_output([FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames', '-of', 'json', str(output)]))
    subprocess.run([FFMPEG, '-v', 'error', '-threads', '2', '-i', str(output), '-f', 'null', '-'], check=True)
    browser_qa = json.loads((qa_dir / 'browser_qa.json').read_text())
    assert browser_qa['errors'] == []
    scales = browser_qa['scales']
    assert scales[0]['fly_scale'] == scales[1]['fly_scale'] == [1,1,1]
    assert scales[0]['camera_field_of_view'] == scales[1]['camera_field_of_view'] == 38
    assert abs(scales[0]['camera_offset_distance']-scales[1]['camera_offset_distance']) < 1e-9
    stream = probe['streams'][0]
    assert int(stream['nb_frames']) == FPS * DURATION
    assert stream['width'] == WIDTH and stream['height'] == HEIGHT
    caption = ('Supplementary Movie 5 | Three-dimensional visualization of modeled navigation after a focal lesion. '
               'A 2,000-neuron lesion at site 20 is shown with no replacement (left) and replacement of 100 lost neurons selected by prioritizing visual-integration and descending cells, with their original connections (right). '
               'Orange points show the same lost neurons; green points show the 100 replacement templates. '
               'Both panels replay the saved ten-second horizontal controller trajectories used in the current V7 package, with neural seed 0 and the same gust seed 3. '
               'The same fly geometry and camera distance are used in both panels. Distance is reported in metres and path straightness is net displacement divided by path length. '
               'The controller models horizontal motion only. Display altitude, vegetation, fly body proportions and wingbeats are illustrative, and the fly is enlarged for visibility. '
               'This film contains no falling physics or simulated biological recovery over time. '
               'The saved source movie also used a population-mean intact reference, which is omitted from this two-condition comparison.')
    (OUT / f'{STEM}_caption.txt').write_text(caption+'\n')
    provenance = dict(status='STRUCTURAL_PASS_PENDING_VISUAL_REVIEW',
                      movie=str(output.relative_to(ROOT)), caption=caption,
                      source_data_version='Corrected V5 frozen data used throughout V7',
                      conditions=records, source_hashes=source_hashes,
                      original_renderer_source='code/web/fly3d.html',
                      original_renderer_sha256=sha(ROOT / 'code/web/fly3d.html'),
                      renderer_script_sha256=sha(Path(__file__)),
                      horizontal_world_units='25 scene units per metre',
                      identical_fly_geometry=True, identical_camera_relative_offset=True,
                      display_altitude='4 + 0.15 sin(3.1t) arbitrary scene units; not a modeled biological altitude',
                      no_new_neural_or_controller_simulations=True,
                      trajectory_selection='Prespecified site 20, neural seed 0 and gust seed 3; no appearance-based selection',
                      intact_reference='No intact fly in this two-condition film. The source V5 film reference is population-mean intact; main figures instead use a matched intact neural seed.',
                      browser_qa=browser_qa, video_probe=probe,
                      decode_check='PASS',
                      output_sha256=sha(output), legacy=legacy)
    (OUT / f'{STEM}_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(output)


if __name__ == '__main__':
    # The updated driver keeps recorded coordinates intact while improving
    # presentation timing and supporting the six-condition Movie 8.
    from navigation3d_smooth import main as smooth_main
    smooth_main()
