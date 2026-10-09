"""Experimental convex, monotone metric-depth calibration with scene robustness.

No test-time reference alignment. This is a calibration head, not a new backbone.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.special import logsumexp


def depth_feature(relative, mode):
    x=np.asarray(relative,dtype=float)
    if x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError('Finite H,W relative-depth prediction required')
    if mode=='raw':return x
    if mode=='frame_quantile':
        low,high=np.quantile(x,[.05,.95])
        return (x-low)/max(high-low,1e-8)
    raise ValueError('Unknown feature mode')


def interpolation_basis(x, knots):
    x=np.asarray(x,float).ravel();knots=np.asarray(knots,float)
    if len(knots)<2 or not (np.diff(knots)>0).all():raise ValueError('Strictly ordered knots required')
    x=np.clip(x,knots[0],knots[-1]);i=np.clip(np.searchsorted(knots,x,side='right')-1,0,len(knots)-2)
    t=(x-knots[i])/(knots[i+1]-knots[i]);b=np.zeros((len(x),len(knots)))
    b[np.arange(len(x)),i]=1-t;b[np.arange(len(x)),i+1]=t
    return b

@dataclass
class RobustMetricSpline:
    knots: np.ndarray
    values: np.ndarray
    mode: str
    smoothness: float
    scene_temperature: float | None
    diagnostics: dict

    @classmethod
    def fit(cls,x,depth_mm,groups,*,mode='raw',smoothness=.02,scene_temperature=None,n_knots=12,delta_mm=.5):
        x=np.asarray(x,float).ravel();y=np.asarray(depth_mm,float).ravel();groups=np.asarray(groups)
        if x.shape!=y.shape or x.shape!=groups.shape or len(x)<n_knots:
            raise ValueError('Matched calibration samples required')
        if not (np.isfinite(x).all() and np.isfinite(y).all() and (y>0).all()):raise ValueError('Invalid calibration data')
        if smoothness<0 or delta_mm<=0 or (scene_temperature is not None and scene_temperature<=0):raise ValueError('Invalid regularization')
        knots=np.unique(np.quantile(x,np.linspace(0,1,n_knots)))
        b=interpolation_basis(x,knots);k=len(knots)
        second=np.diff(np.eye(k),n=2,axis=0)
        group_ids=[np.flatnonzero(groups==g) for g in np.unique(groups)]
        def objective(v):
            e=b@v-y;root=np.sqrt(e*e+delta_mm*delta_mm)
            losses=np.array([np.mean(root[idx]-delta_mm) for idx in group_ids])
            gradients=np.array([b[idx].T@(e[idx]/root[idx])/len(idx) for idx in group_ids])
            if scene_temperature is None:
                loss=losses.mean();gradient=gradients.mean(axis=0)
            else:
                logits=losses/scene_temperature;weights=np.exp(logits-logsumexp(logits))
                loss=scene_temperature*(logsumexp(logits)-np.log(len(logits)));gradient=weights@gradients
            curvature=second@v
            if len(curvature):
                loss+=smoothness*np.mean(curvature*curvature)
                gradient+=2*smoothness*second.T@curvature/len(curvature)
            return loss,gradient
        initial=np.linspace(np.quantile(y,.9),np.quantile(y,.1),k)
        result=minimize(objective,initial,jac=True,method='SLSQP',bounds=Bounds(1,100),
            constraints=[LinearConstraint(np.diff(np.eye(k),axis=0),-np.inf,0)],
            options={'maxiter':400,'ftol':1e-8})
        if not result.success or np.max(np.diff(result.x))>1e-6:
            raise RuntimeError('Constrained optimization failed: '+str(result.message))
        return cls(knots,result.x,mode,smoothness,scene_temperature,
            {'success':bool(result.success),'iterations':int(result.nit),'objective':float(result.fun),
             'fit_scene_count':len(group_ids),'samples':len(y),'delta_mm':delta_mm})

    def predict(self,relative):
        x=depth_feature(relative,self.mode)
        return np.clip(np.interp(x,self.knots,self.values),1,100)

    def to_dict(self):
        return {'knots':self.knots.tolist(),'values':self.values.tolist(),'mode':self.mode,
                'smoothness':self.smoothness,'scene_temperature':self.scene_temperature,'diagnostics':self.diagnostics}

    @classmethod
    def from_dict(cls,d):
        return cls(np.array(d['knots']),np.array(d['values']),d['mode'],d['smoothness'],d['scene_temperature'],d['diagnostics'])
