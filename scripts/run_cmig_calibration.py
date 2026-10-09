"""Official PCC backbone versus generic DA V2 under matched calibration budgets."""
from cmig_revision_common import *
import argparse
import cv2, torch, tifffile
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from train_major_revision_baseline import M
from evaluate_major_revision import reference
from c3vdv2_external_depth_validation import read_reference
from src.radial_affine_calibration import fit,radius_squared

def allrows():
    p=check()
    # PCC SCARED held-out results must not be mistaken for independent evaluation.
    return p['train_rows']+p['test_rows']
def gt_for(r):
    if 'kind' not in r:return reference(r)
    if r['kind']=='source':return read_reference(Path(r['gt']))
    xyz=tifffile.imread(r['gt']);g=xyz[...,2].astype(float)
    return g,np.isfinite(xyz).all(-1)&(g>0)

def inference():
    check();torch.set_num_threads(4)
    for backbone in ['small','pcc']:
        out=O/'raw'/backbone;out.mkdir(parents=True,exist_ok=True)
        todo=[r for r in allrows() if not (out/(r['id']+'.npz')).exists() and not (backbone=='pcc' and r.get('cohort')=='scared')]
        if not todo:continue
        if backbone=='small':
            m=AutoModelForDepthEstimation.from_pretrained(M,local_files_only=True).cuda().eval()
            proc=AutoImageProcessor.from_pretrained(M,local_files_only=True,use_fast=False)
        else:
            os.environ['XFORMERS_DISABLED']='1'
            sys.path[:0]=[str(R/'third_party/EndoDAC'),'/tmp/lesion_publication_deps']
            from models.endodac import endodac
            w=R/'datasets/external_models/EndoDAC_PCC/depth_model.pth'
            parent=read(R/'outputs/recent_baselines/pcc_protocol.json');assert sha(w)==parent['checkpoint_sha256']
            state=torch.load(w,map_location='cpu',weights_only=True)
            height=int(state.pop('height'));width=int(state.pop('width'));state.pop('use_stereo',None)
            assert (height,width)==(256,320)
            m=endodac(backbone_size='base',r=4,image_shape=(224,280),lora_type='dvlora',pretrained_path=None,residual_block_indexes=[2,5,8,11],include_cls_token=True)
            m.load_state_dict(state,strict=True);m.cuda().eval()
        for r in todo:
            rgb=cv2.cvtColor(cv2.imread(str(R/r['rgb'])),cv2.COLOR_BGR2RGB)
            if backbone=='small':
                inp=proc(images=rgb,return_tensors='pt',size={'height':224,'width':280},keep_aspect_ratio=False).to('cuda')
                with torch.inference_mode():d=m(**inp).predicted_depth[0].cpu().numpy()
            else:
                from PIL import Image
                im=np.array(Image.fromarray(rgb).resize((width,height),Image.Resampling.LANCZOS))
                inp=torch.from_numpy(im.copy()).permute(2,0,1).float().div(255).unsqueeze(0).cuda()
                with torch.inference_mode():disp=m(inp)[('disp',0)][0,0].cpu().numpy()
                d=1/150+(1/.1-1/150)*disp
            assert np.isfinite(d).all();np.savez_compressed(out/(r['id']+'.npz'),prediction=d)
            print('RAW',backbone,r['id'],flush=True)
        dump(O/f'{backbone}_inference.json',{'parameters':sum(x.numel() for x in m.parameters()),'resolution':[224,280] if backbone=='small' else [height,width],'strict_loading':True,'test_reference_alignment':False,'pcc_scared_test_excluded':backbone=='pcc'})
        del m;torch.cuda.empty_cache()

def indices(rows,total):
    # Exact total, equal object then equal frame allocation up to single-pixel rounding.
    result=[];groups=sorted({str(r['object']) for r in rows})
    for j,obj in enumerate(groups):
        rr=sorted([r for r in rows if str(r['object'])==obj],key=lambda r:r['id'])
        n=total//len(groups)+(j<total%len(groups))
        for k,r in enumerate(rr):
            g,v=gt_for(r);valid=np.flatnonzero(v)
            count=n//len(rr)+(k<n%len(rr));assert len(valid)>=count
            ix=valid[np.linspace(0,len(valid)-1,count,dtype=int)]
            result.append((r,ix))
    assert sum(len(ix) for _,ix in result)==total
    return result

def calibrate():
    p=check();assert not (O/'calibration_heads.json').exists(),'Refusing to refit frozen heads'
    src=[r for r in p['train_rows'] if r['kind']=='source'];tgt=[r for r in p['train_rows'] if r['kind']=='target']
    designs={'source_only':indices(src,16384),'source_target':indices(src,8192)+indices(tgt,8192)}
    heads={};support={}
    for design,samples in designs.items():
        support[design]=[]
        for r,ix in samples:
            support[design].append({'frame':r['id'],'kind':r['kind'],'object':str(r['object']),'n':len(ix),'flat_indices_sha256':hashlib.sha256(ix.astype('<i8').tobytes()).hexdigest()})
        for backbone in ['small','pcc']:
            dd=[];zz=[];rr=[];gg=[]
            for row,ix in samples:
                g,v=gt_for(row)
                with np.load(O/'raw'/backbone/(row['id']+'.npz')) as f:d=cv2.resize(f['prediction'],g.shape[::-1],interpolation=cv2.INTER_LINEAR).astype(float)
                dd.extend(d.ravel()[ix]);zz.extend(g.ravel()[ix]);rr.extend(radius_squared(g.shape).ravel()[ix]);gg.extend([row['kind']]*len(ix))
            for radial in [False,True]:
                name=f'{backbone}_{design}_'+('radial_ls' if radial else 'affine_ls')
                heads[name]=fit(dd,rr,zz,gg,radial,0.)
                assert heads[name]['n_samples']==16384
    dump(O/'calibration_support.json',support)
    dump(O/'calibration_heads.json',{'protocol_sha256':sha(O/'protocol.json'),'heads':heads,'test_reference_used':False,'sampling_support_sha256':sha(O/'calibration_support.json')})
    print('FROZEN HEADS',len(heads),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['inference','calibrate']);a=p.parse_args();globals()[a.phase]()
