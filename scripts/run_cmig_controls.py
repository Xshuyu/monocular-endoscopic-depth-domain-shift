"""Additional exposure controls and input-resolution paired interventions."""
from cmig_revision_common import *
import argparse, time
import torch, cv2
from torch import nn
from transformers import AutoImageProcessor
from train_major_revision_baseline import model, M
from evaluate_major_revision import frames

def train():
    p=check();torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    cached={r['id']:torch.load(OLD/'training_cache'/(r['id']+'.pt'),weights_only=True) for r in p['train_rows']}
    src=[r['id'] for r in p['train_rows'] if r['kind']=='source']
    tgt=[r['id'] for r in p['train_rows'] if r['kind']=='target']
    out=O/'trained_models';out.mkdir(exist_ok=True)
    for seed in SEEDS:
        for regime in REGIMES:
            dest=out/f'{regime}_{seed}.pt'
            if dest.exists():continue
            torch.manual_seed(seed);srng=np.random.default_rng(seed);trng=np.random.default_rng(seed+1000)
            m=model().cuda()
            for x in m.backbone.parameters():x.requires_grad_(False)
            nn.init.normal_(m.head.conv3.weight,0,.001);nn.init.constant_(m.head.conv3.bias,np.log(50))
            m.neck.train();m.head.train()
            pars=[x for x in m.parameters() if x.requires_grad]
            opt=torch.optim.AdamW(pars,lr=1e-4,weight_decay=.01)
            steps=1200 if regime=='adapted_double' else 600
            log=[];draws=[];start=time.monotonic()
            for step in range(steps):
                if regime=='source_half':ids=[src[srng.integers(len(src))]]
                elif regime=='target_only':ids=[tgt[trng.integers(len(tgt))] for _ in range(2)]
                else:ids=[src[srng.integers(len(src))],tgt[trng.integers(len(tgt))]]
                draws.append(ids);batch=[cached[x] for x in ids]
                features=[torch.cat([x['features'][i] for x in batch]).cuda() for i in range(len(batch[0]['features']))]
                pred=m.head(m.neck(features,16,20),16,20)
                gt=torch.stack([x['log_gt'] for x in batch]).cuda();valid=torch.stack([x['valid'] for x in batch]).cuda()
                loss=torch.stack([nn.functional.smooth_l1_loss(pred[i][valid[i]],gt[i][valid[i]],beta=.1) for i in range(len(ids))]).mean()
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(pars,1.);opt.step()
                if step%100==0 or step==steps-1:
                    log.append({'step':step+1,'loss':float(loss.detach())})
                    print(regime,seed,step+1,log[-1]['loss'],flush=True)
            torch.save({'neck':m.neck.state_dict(),'head':m.head.state_dict(),'seed':seed,'regime':regime,'steps':steps,'trainable_parameters':sum(x.numel() for x in pars),'protocol_sha256':sha(O/'protocol.json')},dest)
            dump(out/f'{regime}_{seed}_log.json',{'loss':log,'draws':draws,'elapsed_seconds':time.monotonic()-start})
            del m,opt;torch.cuda.empty_cache()
    dump(O/'training_complete.json',{'checkpoints':{p.name:sha(p) for p in out.glob('*.pt')}})

def inference():
    check();torch.set_num_threads(4);m=model().cuda().eval()
    for seed in SEEDS:
        for regime in REGIMES:
            state=torch.load(O/'trained_models'/f'{regime}_{seed}.pt',map_location='cpu',weights_only=True)
            assert state['protocol_sha256']==sha(O/'protocol.json')
            m.neck.load_state_dict(state['neck']);m.head.load_state_dict(state['head'])
            out=O/'control_predictions'/f'{regime}_{seed}';out.mkdir(parents=True,exist_ok=True)
            for r in frames():
                dest=out/(r['id']+'.npz')
                if dest.exists():continue
                v=torch.load(OLD/'test_features'/(r['id']+'_identity.pt'),weights_only=True)
                with torch.inference_mode():log=m.head(m.neck([x.cuda() for x in v['features']],16,20),16,20)
                pred=torch.exp(log.clamp(-10,10)).squeeze().cpu().numpy()
                assert np.isfinite(pred).all();np.savez_compressed(dest,prediction=pred)
            print('CONTROL INFERENCE',regime,seed,flush=True)

def factors():
    check();torch.set_num_threads(4)
    m=model().cuda().eval();proc=AutoImageProcessor.from_pretrained(M,local_files_only=True,use_fast=False)
    states={(regime,seed):torch.load(OLD/'trained_models'/f'{regime}_{seed}.pt',map_location='cpu',weights_only=True) for regime in ['source_only','adapted'] for seed in SEEDS}
    for r in frames():
        rgb=cv2.cvtColor(cv2.imread(str(R/r['rgb'])),cv2.COLOR_BGR2RGB)
        # Resize before perturbation: sigma is explicitly in model-input pixels.
        inp=proc(images=rgb,return_tensors='np',size={'height':224,'width':280},keep_aspect_ratio=False,do_rescale=False,do_normalize=False).pixel_values[0]
        resized=np.moveaxis(inp,0,-1).astype(np.float32)/255.
        for factor,(gamma,sigma) in FACTORS.items():
            paths={(regime,seed):O/'factor_predictions'/f'{regime}_{seed}'/(r['id']+'_'+factor+'.npz') for regime,seed in states}
            if all(x.exists() for x in paths.values()):continue
            im=np.clip(resized**gamma,0,1)
            if sigma:im=cv2.GaussianBlur(im,(0,0),sigma)
            pixels=proc(images=im,return_tensors='pt',do_resize=False,do_rescale=False).to('cuda')
            assert pixels.pixel_values.shape[-2:]==(224,280)
            with torch.inference_mode():features=m.backbone.forward_with_filtered_kwargs(pixels.pixel_values).feature_maps
            for key,state in states.items():
                dest=paths[key]
                if dest.exists():continue
                m.neck.load_state_dict(state['neck']);m.head.load_state_dict(state['head'])
                with torch.inference_mode():log=m.head(m.neck(features,16,20),16,20)
                pred=torch.exp(log.clamp(-10,10)).squeeze().cpu().numpy()
                assert np.isfinite(pred).all();dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,prediction=pred)
        print('INPUT FACTORS',r['id'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','inference','factors']);a=p.parse_args();globals()[a.phase]()
