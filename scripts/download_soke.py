"""Reproducible SOKE inference bundle. Resumable ranges; no checkpoint unpickling.

The checked-in tensor/storage manifests were extracted with a restricted metadata
reader from the pinned checkpoint. CRC checks each storage; SHA-256 checks the
assembled safetensors and all support files against the tested local bundle.
"""
from pathlib import Path
import concurrent.futures as cf
import json, subprocess, struct, zlib, hashlib, pickle, io
import numpy as np
from safetensors.numpy import save_file
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models/soke'
REV = '1385f54f5324c3cb9abaf0a3b8bfcab97b2a0ef2'
URL = f'https://huggingface.co/1-13-am/soke-ckpt/resolve/{REV}/min-csl_DTW_MPJPE_PA_lhandepoch%3D94.ckpt'
SIZE = 4826470052
CHUNK = 8 * 1024 * 1024
MANIFEST = ROOT / 'scripts/manifests'
HASHES = json.loads((MANIFEST / 'soke-sha256.json').read_text())
CODE_REV = '5cbc55d84b5a7cbf05a9cf020c468052e8d94d00'

def verified(path):
    if not path.exists():
        return False
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(CHUNK), b''):
            digest.update(data)
    return digest.hexdigest() == HASHES[path.name]

class NumericMap(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Only a plain numeric vocabulary map is accepted')

def support():
    import torch
    files = {
        'config.json': '189e3DNUiOdfJNgTmxUX6nm5DlVLtk3dp',
        'map_ids.pkl': '1RvyRV7kVlfw866AnUtgxNhNYTZtuoW4p',
        'sentencepiece.bpe.model': '1TRfdAdXKaGPlLCt1cOWm3whU3YreKwnG',
        'mean.pt': '1NH-eVtS0nNjMjCwae-A1ii5sxj44C3bo',
        'std.pt': '1FHHWS0GPM2s6S2PB2JHv4ufdEbzezuKW',
    }
    urls = {name: f'https://drive.google.com/uc?export=download&id={id}' for name, id in files.items()}
    urls['word2code.json'] = f'https://raw.githubusercontent.com/2000ZRL/SOKE/{CODE_REV}/scripts/word2code.json'
    for name, url in urls.items():
        path = DEST / name
        if not verified(path):
            partial = path.with_suffix(path.suffix + '.part')
            curl(url, partial)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != HASHES[name]:
                raise ValueError(f'Support checksum mismatch: {name}')
            partial.replace(path)
        print(f'Ready: {name}', flush=True)
    mapping = NumericMap(io.BytesIO((DEST / 'map_ids.pkl').read_bytes())).load()
    if not isinstance(mapping, dict) or not all(type(k) is int and type(v) is int for k, v in mapping.items()):
        raise ValueError('Invalid vocabulary map')
    (DEST / 'map_ids.json').write_text(json.dumps(mapping))
    for name in ['mean', 'std']:
        values = torch.load(DEST / f'{name}.pt', map_location='cpu', weights_only=True).numpy()
        result = np.concatenate([values[36:159], values[169:]])
        if result.shape != (133,) or not np.isfinite(result).all():
            raise ValueError('Invalid normalization')
        np.save(DEST / f'{name}133.npy', result)

def curl(url, path, bounds=None):
    cmd=['curl','--fail','--location','--silent','--show-error','--retry','4','--retry-all-errors','--max-time','120']
    if bounds: cmd += ['--range',f'{bounds[0]}-{bounds[1]}']
    subprocess.run(cmd+[url,'--output',str(path)],check=True)

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    support()
    if verified(DEST / 'model.safetensors'):
        (DEST/'revision.json').write_text(json.dumps({'repo':'1-13-am/soke-ckpt','revision':REV,'format':'selected-inference-tensors','source':'community checkpoint; evaluation required'},indent=2))
        print('SOKE bundle verified; already installed.', flush=True)
        return
    storage=json.loads((MANIFEST/'soke-storages.json').read_text())
    spec=json.loads((MANIFEST/'soke-tensors.json').read_text())
    chunks=DEST/'chunks';chunks.mkdir(exist_ok=True)
    needed=set()
    for row in storage:
        start=row['header_offset'];end=start+row['file_size']+512
        needed.update(range(start//CHUNK,min(end,SIZE-1)//CHUNK+1))
    def fetch(i):
        path=chunks/str(i);start=i*CHUNK;end=min(start+CHUNK,SIZE)-1
        if not path.exists() or path.stat().st_size!=end-start+1:
            curl(URL,path,(start,end))
        if path.stat().st_size!=end-start+1:raise RuntimeError(f'Bad range {i}')
        return i
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for n,_ in enumerate(pool.map(fetch,sorted(needed)),1):
            if n%16==0 or n==len(needed):print(f'Model ranges {n}/{len(needed)}',flush=True)
    def read(start,length):
        data=bytearray()
        while length:
            i,offset=divmod(start,CHUNK);n=min(length,CHUNK-offset)
            with (chunks/str(i)).open('rb') as f:f.seek(offset);data.extend(f.read(n))
            start+=n;length-=n
        return data
    arrays={}
    for row in storage:
        h=read(row['header_offset'],30)
        if h[:4]!=b'PK\x03\x04':raise ValueError('Invalid local archive header')
        name_len,extra_len=struct.unpack_from('<HH',h,26)
        data=read(row['header_offset']+30+name_len+extra_len,row['file_size'])
        if zlib.crc32(data)!=row['CRC']:raise ValueError('Storage checksum failed')
        arrays[row['filename'].rsplit('/',1)[-1]]=np.frombuffer(data,dtype='<f4')
    tensors={}
    for name,row in spec.items():
        if row['storage'][1]!='FloatStorage':raise ValueError('Unexpected tensor dtype')
        a=arrays[row['storage'][2]];shape=row['shape'];offset=row['offset']
        tensors[name]=a[offset:offset+int(np.prod(shape))].reshape(shape).copy()
    partial = DEST / 'model.safetensors.part'
    save_file(tensors,str(partial))
    if hashlib.sha256(partial.read_bytes()).hexdigest() != HASHES['model.safetensors']:
        raise ValueError('Assembled model checksum mismatch')
    partial.replace(DEST / 'model.safetensors')
    (DEST/'revision.json').write_text(json.dumps({'repo':'1-13-am/soke-ckpt','revision':REV,'format':'selected-inference-tensors','source':'community checkpoint; evaluation required'},indent=2))
    print('Saved tensor-only checkpoint',flush=True)

if __name__=='__main__':main()
