// node driver.cjs movieN.json out.mp4 [previewFramesCSV]
const fs=require('fs'),path=require('path'),{spawn}=require('child_process'),{once}=require('events');
const {chromium}=require('playwright');
(async()=>{
  const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),out=process.argv[3],preview=process.argv[4]?process.argv[4].split(',').map(Number):null;
  const browser=await chromium.launch({headless:true,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist','--allow-file-access-from-files']});
  const page=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});const errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto('file://'+path.resolve(__dirname,'render.html'),{waitUntil:'load'});
  await page.evaluate(d=>window.setMovie(d),data);
  let enc=null,err='';
  if(!preview){enc=spawn('ffmpeg',['-y','-loglevel','error','-f','image2pipe','-vcodec','mjpeg','-framerate','60','-i','-','-an','-c:v','libx264','-preset','medium','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',out]);enc.stderr.on('data',d=>err+=d);}
  const qa=[];const t0=Date.now();
  for(let c=0;c<data.chapters.length;c++){
    await page.evaluate(k=>window.setChapter(k),c);
    const frames=preview?preview:Array.from({length:data.frames},(_,i)=>i);
    for(const n of frames){
      const s=await page.evaluate(n=>window.renderFrame(n),n);
      if(n%60===0)qa.push({chapter:c,frame:n,panels:s});
      if(preview){await page.screenshot({path:out.replace('.png','')+`_c${c}_f${String(n).padStart(3,'0')}.png`});}
      else{const b=await page.screenshot({type:'jpeg',quality:95});if(!enc.stdin.write(b))await once(enc.stdin,'drain');}
      if(!preview&&n%120===0)process.stdout.write(`ch${c} frame ${n} ${((Date.now()-t0)/1000).toFixed(0)}s\n`);
    }
  }
  if(enc){enc.stdin.end();const [code]=await once(enc,'close');if(code!==0)throw new Error(err);}
  fs.writeFileSync((preview?out.replace('.png',''):out.replace('.mp4',''))+'_qa.json',JSON.stringify({errors,qa},null,1));
  await browser.close();if(errors.length){console.error(errors.join('\n'));}
})().catch(e=>{console.error(e);process.exit(1);});
