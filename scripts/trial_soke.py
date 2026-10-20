"""Run real sentences through the isolated replacement model and native rig."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from services.motion.src.soke_provider import SOKEProvider
from services.motion.src.body import BodyModel
p=SOKEProvider(); b=BodyModel(); start=time.monotonic()
p.load(); print('Loaded SOKE in',round(time.monotonic()-start,2),'seconds on',p.device,flush=True)
texts=['Hi! How are you?','Hello, how are you?','My name is Alex.','I would like a glass of water.']
report=[]
for i,text in enumerate(texts):
 motion,meta=p.generate(text)
 directory=ROOT/'runtime/soke-trial'/str(i)
 rendering=b.prepare(motion,directory)
 diagnostics=b.motion_diagnostics(motion)
 row={'text':text,'metadata':meta,'diagnostics':diagnostics,'frames':len(motion),'bodyTemporalStd':float(motion[:,:30].std(0).mean())}
 (directory/'report.json').write_text(json.dumps(row,indent=2))
 (directory/'rendering.json').write_text(json.dumps(rendering))
 report.append(row);print(json.dumps(row),flush=True)
(ROOT/'runtime/soke-trial/report.json').write_text(json.dumps(report,indent=2))
