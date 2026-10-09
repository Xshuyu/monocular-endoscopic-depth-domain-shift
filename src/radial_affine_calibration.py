"""Small spatial inverse-depth heads with explicit train-only standardization."""
import numpy as np

def radius_squared(shape):
 h,w=shape;y,x=np.ogrid[:h,:w];return ((x-w/2.)**2+(y-h/2.)**2)/((w/2.)**2+(h/2.)**2)

def features(d,r2,radial):
 return np.column_stack([d,d*r2,r2] if radial else [d])

def fit(d,r2,z,groups,radial=False,lam=0.):
 d,r2,z=[np.asarray(a,dtype=float) for a in [d,r2,z]]
 if not (np.isfinite(d).all() and np.isfinite(r2).all() and np.isfinite(z).all() and (z>0).all()):raise ValueError('Invalid calibration samples')
 _,g,c=np.unique(groups,return_inverse=True,return_counts=True);w=1./(len(c)*c[g])
 x=features(d,r2,radial);mu=np.sum(x*w[:,None],axis=0);scale=np.sqrt(np.sum((x-mu)**2*w[:,None],axis=0));scale=np.where(scale>1e-12,scale,1.)
 a=np.column_stack([(x-mu)/scale,np.ones(len(z))]);aw=a*np.sqrt(w[:,None]);yw=np.sqrt(w)/z
 penalty=np.diag([1.]*(a.shape[1]-1)+[0.])
 augmented=np.vstack([aw,np.sqrt(lam)*penalty]);target=np.r_[yw,np.zeros(a.shape[1])]
 beta=np.linalg.lstsq(augmented,target,rcond=None)[0];slopes=beta[:-1]/scale;intercept=beta[-1]-mu@slopes
 coeff=[slopes[0],slopes[1],intercept,slopes[2]] if radial else [slopes[0],0.,intercept,0.]
 return {'radial':radial,'lambda':lam,'feature_mean':mu.tolist(),'feature_scale':scale.tolist(),'standardized_beta':beta.tolist(),'coefficients_a0_a1_b0_b1':list(map(float,coeff)),'n_samples':len(z),'training_scenes':sorted(set(map(str,groups))),'condition_number':float(np.linalg.cond(augmented)),'objective':'scene-mean inverse squared error + lambda * all standardized non-intercept slopes squared'}

def predict(head,d,r2=None):
 d=np.asarray(d,dtype=float)
 if r2 is None:r2=radius_squared(d.shape)
 a0,a1,b0,b1=head['coefficients_a0_a1_b0_b1'];inv=(a0+a1*r2)*d+b0+b1*r2
 return np.clip(np.divide(1.,inv,out=np.full_like(inv,100.),where=inv>0),1,100)
