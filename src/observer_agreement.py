"""Absolute-agreement ICC for a complete object-by-rater measurement matrix.
Statistical software only; no clinical observations are synthesized.
"""
import numpy as np

def absolute_agreement_icc(values):
 x=np.asarray(values,dtype=float)
 if x.ndim!=2 or min(x.shape)<2 or not np.isfinite(x).all():raise ValueError('Need a finite complete object-by-rater matrix, at least 2x2')
 n,k=x.shape;grand=x.mean();obj=x.mean(axis=1);rat=x.mean(axis=0)
 ms_obj=k*np.sum((obj-grand)**2)/(n-1)
 ms_rat=n*np.sum((rat-grand)**2)/(k-1)
 residual=x-obj[:,None]-rat[None,:]+grand
 ms_error=np.sum(residual**2)/((n-1)*(k-1))
 den_single=ms_obj+(k-1)*ms_error+k*(ms_rat-ms_error)/n
 den_mean=ms_obj+(ms_rat-ms_error)/n
 if den_single<=0 or den_mean<=0:raise ValueError('ICC undefined for degenerate measurements')
 return {'icc_a1':float((ms_obj-ms_error)/den_single),'icc_ak':float((ms_obj-ms_error)/den_mean),'objects':n,'raters':k,'ms_object':float(ms_obj),'ms_rater':float(ms_rat),'ms_error':float(ms_error)}

def cluster_bootstrap_icc(values,cluster_ids,*,draws=5000,seed=20260930):
 x=np.asarray(values,float);groups=np.asarray(cluster_ids)
 if len(groups)!=len(x):raise ValueError('One cluster ID per object required')
 unique=np.unique(groups)
 if len(unique)<3:raise ValueError('At least three independent clusters required for this descriptive bootstrap; this is not a sample-size adequacy rule')
 rng=np.random.default_rng(seed);samples=[]
 for _ in range(draws):
  ids=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
  try:
   v=absolute_agreement_icc(x[ids]);samples.append([v['icc_a1'],v['icc_ak']])
  except ValueError:continue
 if len(samples)<.95*draws:raise ValueError('Too many degenerate bootstrap draws')
 ci=np.quantile(samples,[.025,.975],axis=0)
 return {'icc_a1_descriptive_95ci':ci[:,0].tolist(),'icc_ak_descriptive_95ci':ci[:,1].tolist(),'draws_requested':draws,'draws_valid':len(samples),'clusters':len(unique),'scope':'Cluster percentile bootstrap conditional on observed raters; not random-rater population coverage'}
