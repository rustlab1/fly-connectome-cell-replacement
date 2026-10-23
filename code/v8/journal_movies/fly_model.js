function makeFly() {
  const wings={};
  const fly=new THREE.Group();const body=new THREE.Group();fly.add(body);fly.userData.body=body;
  const ochre=new THREE.MeshStandardMaterial({color:0x9b8055,roughness:.78});
  const warm=new THREE.MeshStandardMaterial({color:0xb6a075,roughness:.83});
  const dark=new THREE.MeshStandardMaterial({color:0x3b3630,roughness:.83});
  const eyeMat=new THREE.MeshStandardMaterial({color:0x983c2d,roughness:.57});
  const sphere=new THREE.SphereGeometry(1,40,28);
  function oval(material,pos,size){const mesh=new THREE.Mesh(sphere,material);mesh.position.set(...pos);mesh.scale.set(...size);mesh.castShadow=true;mesh.receiveShadow=true;body.add(mesh);return mesh;}
  oval(ochre,[0,0,0],[.56,.46,.44]);
  oval(warm,[.1,-.21,0],[.47,.29,.35]);
  const abdomenGeo=new THREE.SphereGeometry(1,56,32),position=abdomenGeo.attributes.position,colors=[];
  for(let i=0;i<position.count;i++){
    const x=position.getY(i),y=-position.getX(i),z=position.getZ(i),taper=.78+.22*(x+1)/2;
    position.setXYZ(i,x*.95,y*.35*taper,z*.42*taper);
    const phase=(x+1)*3.1,band=phase-Math.floor(phase)<.22&&y>-.38;
    const color=new THREE.Color(band?0x69553f:y>.35?0xba9b62:0xc1ac80);
    color.multiplyScalar(.9+.1*(y+1)/2);colors.push(color.r,color.g,color.b);
  }
  abdomenGeo.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));abdomenGeo.computeVertexNormals();
  const abdomen=new THREE.Mesh(abdomenGeo,new THREE.MeshStandardMaterial({vertexColors:true,roughness:.83}));
  abdomen.position.set(-1.12,-.055,0);abdomen.castShadow=true;abdomen.receiveShadow=true;body.add(abdomen);
  oval(warm,[.64,.045,0],[.34,.3,.34]);
  for(const s of [-1,1]){
    oval(eyeMat,[.72,.07,s*.265],[.195,.235,.142]);
    oval(dark,[.905,-.005,s*.11],[.065,.06,.048]);
    const arista=new THREE.Mesh(new THREE.CylinderGeometry(.008,.013,.31,5),dark);
    arista.position.set(1.04,.10,s*.16);arista.rotation.z=-.94;arista.rotation.x=s*.3;body.add(arista);
    const hair=new THREE.Mesh(new THREE.CylinderGeometry(.005,.006,.08,4),dark);
    hair.position.set(1.10,.205,s*.16);hair.rotation.z=-.55;body.add(hair);
  }
  oval(ochre,[.73,-.25,0],[.08,.18,.09]);
  const hairGeo=new THREE.CylinderGeometry(.006,.009,.13,5);
  const bristles=new THREE.InstancedMesh(hairGeo,dark,76),rng=mulberry32(12),m=new THREE.Matrix4();
  for(let i=0;i<76;i++){
    const a=rng()*2*Math.PI,u=.15+rng()*.85,normal=new THREE.Vector3(Math.sqrt(1-u*u)*Math.cos(a),u,Math.sqrt(1-u*u)*Math.sin(a));
    const point=new THREE.Vector3(normal.x*.55,normal.y*.46,normal.z*.44).addScaledVector(normal,.04);
    m.compose(point,new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0),normal),new THREE.Vector3(1,1,1));bristles.setMatrixAt(i,m);
  }
  body.add(bristles);
  function segment(a,b,r,mat,parent){const av=new THREE.Vector3(...a),bv=new THREE.Vector3(...b),d=bv.clone().sub(av);
    const mesh=new THREE.Mesh(new THREE.CylinderGeometry(r*.75,r,d.length(),7),mat);mesh.position.copy(av.add(bv).multiplyScalar(.5));
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.normalize());mesh.castShadow=true;parent.add(mesh);}
  for(const s of [-1,1])for(let k=0;k<3;k++){
    const leg=new THREE.Group();leg.position.set(.30-k*.40,-.28,s*.32);body.add(leg);
    const knee=[.15-k*.08,-.27,s*.17],ankle=[.48-k*.32,-.52,s*.21],tip=[.61-k*.37,-.57,s*.26];
    segment([0,0,0],knee,.035,ochre,leg);segment(knee,ankle,.022,dark,leg);segment(ankle,tip,.015,dark,leg);
    segment(tip,[tip[0]+.08,tip[1]+.035,tip[2]+s*.045],.009,dark,leg);
  }
  for(const s of [-1,1]){
    segment([-.46,-.02,s*.38],[-.63,.01,s*.65],.014,ochre,body);
    oval(warm,[-.65,.01,s*.67],[.075,.06,.065]);
  }
  for(const s of [-1,1]){
    const hinge=new THREE.Group();hinge.position.set(-.07,.38,s*.25);body.add(hinge);const ghosts=[];
    const shape=new THREE.Shape();shape.moveTo(0,0);
    shape.bezierCurveTo(-.03,.65,-.75,1.72,-1.48,2.17);
    shape.bezierCurveTo(-1.85,2.41,-2.18,2.10,-2.08,1.71);
    shape.bezierCurveTo(-1.94,1.14,-.88,.35,-.30,.06);shape.lineTo(0,0);
    const geo=new THREE.ShapeGeometry(shape,32);geo.rotateX(Math.PI/2);geo.scale(1,1,s);
    const membrane=new THREE.MeshPhysicalMaterial({color:0xe2e8e6,transparent:true,opacity:.48,roughness:.55,side:THREE.DoubleSide,depthWrite:false});
    const ghostMat=new THREE.MeshBasicMaterial({color:0xd0d9d7,transparent:true,opacity:.055,side:THREE.DoubleSide,depthWrite:false});
    const outline=shape.getPoints(70).map(p=>new THREE.Vector3(p.x,0,s*p.y));
    const branches=[[[0,0],[-.43,.69],[-1.10,1.53],[-1.64,2.15]],[[0,0],[-.58,.68],[-1.20,1.23],[-1.97,1.90]],[[0,0],[-.60,.47],[-1.26,.91],[-2.04,1.58]],[[-.44,.71],[-.62,.66]],[[-1.05,1.48],[-1.25,1.19]],[[-1.30,.97],[-1.48,1.36]]];
    for(let q=0;q<3;q++){
      const ghost=new THREE.Group();hinge.add(ghost);ghosts.push(ghost);ghost.add(new THREE.Mesh(geo,q===0?membrane:ghostMat));
      if(q===0){
        ghost.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(outline),new THREE.LineBasicMaterial({color:0x7d8d8b,transparent:true,opacity:.52})));
        for(const branch of branches)ghost.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(branch.map(([x,z])=>new THREE.Vector3(x,.004,s*z))),new THREE.LineBasicMaterial({color:0x81938f,transparent:true,opacity:.64})));
      }
    }
    hinge.userData.ghosts=ghosts;wings[s]=hinge;
  }
  return {fly,wings};
}
