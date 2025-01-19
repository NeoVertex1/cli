#!/usr/bin/env python3

import math
import cmath
import random
import time
import json
import os
import sys
import yaml
import click
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image

PSI= 44.8
XI= 3721.8
TAU= 64713.97
EPSILON= 0.28082
PHI= (1 + math.sqrt(5)) / 2
BOLTZMANN= 1.380649e-23
PI= math.pi

SESSION_DATA= {
    "psi_offset":0.0,
    "xi_scale":1.0,
    "coherence_mult":1.0,
    "layer_params":[],
    "extra_data":{}
}

@dataclass
class HPCConfig:
    dimension: int= 3
    protection_level: int= 3
    doping_randomize: bool= False
    doping_strength: float= 0.0
    doping_layers: List[Dict[str,Any]]= field(default_factory=list)
    advanced_ec: bool= False
    noise_model: str= "gaussian"
    temperature: float= 0.1
    concurrency: int= 1
    use_slurm: bool= False
    job_name: str= "qjob"
    slurm_time: str= "00:30:00"
    slurm_nodes: int= 1
    slurm_ntasks: int= 1
    error_correction_threshold: float= 0.75
    ml_algorithm: str= "random"
    ml_episodes: int= 0
    param_dims: List[int]= field(default_factory=lambda:[3,4,5])
    param_prots: List[int]= field(default_factory=lambda:[3,4,5])
    param_trials: int= 100
    param_output: str= ""
    doping_correl_dist: float= 0.0
    verbose: bool= False
    data_dir: str= "research_data"

SESSION_LOG= []
os.makedirs("research_data", exist_ok=True)

def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class HPCAdvancedNoise:
    def __init__(self,cfg:HPCConfig):
        self.cfg= cfg
        self.thermal_energy= BOLTZMANN* cfg.temperature
    def thermal_noise_factor(self, e: float)->float:
        if self.thermal_energy<1e-50:
            return 0.0
        return math.exp(-e/self.thermal_energy)
    def apply_gate_noise(self, gate: np.ndarray, dt: float)-> np.ndarray:
        if self.cfg.noise_model=="gaussian":
            noise= np.random.normal(0,0.01,gate.shape)* dt
            return gate + noise
        elif self.cfg.noise_model=="1/f":
            scale= 0.01/(1.0+dt) if dt>0 else 0.01
            flicker= np.random.normal(0,scale,gate.shape)
            return gate + flicker
        elif self.cfg.noise_model=="random_telegraph":
            s= -1 if random.random()<0.5 else 1
            return gate + s*0.005* dt*np.ones_like(gate)
        return gate

def doping_factor_from_layers(lays:List[Dict[str,Any]])->float:
    fac=1.0
    for l in lays:
        c= l.get("dopant_conc",0.0)
        fac*= (1.0+ c*0.001)
    return fac

def doping_correlation_strength(dist: float)-> float:
    return math.exp(-dist* random.random())

class HPCMFT:
    def __init__(self,cfg:HPCConfig):
        base_psi= PSI+ SESSION_DATA["psi_offset"]
        base_xi= XI* SESSION_DATA["xi_scale"]
        self.psi= base_psi
        self.xi= base_xi
        self.tau= TAU
        self.eps= EPSILON
        self.phi= PHI
        self.cfg= cfg
        self.sqrt_psi_xi= math.sqrt(self.psi*self.xi)
        self.noise= HPCAdvancedNoise(cfg)
        self.protection_levels= self._calc_levels()
    def _calc_levels(self)->List[tuple]:
        arr=[]
        dop= doping_factor_from_layers(self.cfg.doping_layers)
        if self.cfg.doping_randomize:
            dop*= (1.0+ self.cfg.doping_strength* random.random())
        dop*= doping_correlation_strength(self.cfg.doping_correl_dist)
        for n in range(1,6):
            e_n= self.sqrt_psi_xi*(self.phi**n)
            c_n= self.tau* math.exp(-n*self.eps)
            c_n*= dop
            thr= self.noise.thermal_noise_factor(e_n)
            c_n*= (1.0- thr)
            c_n*= SESSION_DATA["coherence_mult"]
            arr.append((n,e_n,c_n))
        return arr

def random_unitary(dim:int)-> np.ndarray:
    A= np.random.normal(size=(dim,dim))+ 1j*np.random.normal(size=(dim,dim))
    Q,R= np.linalg.qr(A)
    diagR= np.diagonal(R)
    phases= diagR/ np.abs(diagR)
    return Q* phases

def surface_code_check(dim:int)->List[np.ndarray]:
    ops=[]
    for _ in range(dim):
        diag=[]
        for _ in range(dim):
            diag.append(1 if random.random()<0.5 else -1)
        M= np.diag(diag).astype(complex)
        ops.append(M)
    return ops

def apply_stabilizer(rho:np.ndarray,M:np.ndarray,t: float)-> np.ndarray:
    val= np.trace(M@rho)
    if val.real< t:
        cM= M.conjugate().transpose()
        nrho= cM@ rho@ cM
        s= np.trace(nrho)
        if abs(s)<1e-15:
            d= rho.shape[0]
            return np.identity(d)/d
        return nrho/s
    return rho

class HPCQudit:
    def __init__(self, dim:int, prot:int, mft:HPCMFT):
        self.dim= dim
        self.protection= prot
        self.mft= mft
        rec= self._find_level()
        self.energy= rec[1]
        self.coherence= rec[2]
        self.rho= np.zeros((dim,dim),dtype=complex)
        self.rho[0,0]=1.0+0j
        self.last_access= time.time()
        self.error_rate= math.exp(-self.energy/self.mft.xi)
        self.op_count=0
        self.err_count=0
    def _find_level(self)->tuple:
        for x in self.mft.protection_levels:
            if x[0]== self.protection:
                return x
        return self.mft.protection_levels[-1]
    def apply_gate(self, G: np.ndarray):
        dt= time.time()- self.last_access
        self._maybe_decohere(dt)
        G2= self.mft.noise.apply_gate_noise(G, dt)
        self.rho= G2@ self.rho@ G2.conjugate().transpose()
        self._apply_errors(dt)
        self.op_count+=1
        self.last_access= time.time()
    def measure(self)->int:
        dt= time.time()- self.last_access
        self._maybe_decohere(dt)
        diag= np.diagonal(self.rho)
        pvals= [max(0,d.real) for d in diag]
        s= sum(pvals)
        if s<1e-15:
            o= random.randint(0,self.dim-1)
        else:
            r= random.random()* s
            c=0.0
            o=0
            for i,v in enumerate(pvals):
                c+=v
                if r<=c:
                    o=i
                    break
            proj= np.zeros((self.dim,self.dim),dtype=complex)
            proj[o,o]=1+0j
            if v<1e-15:
                self.rho= np.identity(self.dim)/ self.dim
            else:
                self.rho= (proj@ self.rho@ proj)/(v)
        self.op_count+=1
        self.last_access= time.time()
        return o
    def _maybe_decohere(self, dt: float):
        if dt> self.coherence:
            self.rho= np.identity(self.dim)/ self.dim
    def _apply_errors(self, dt: float):
        p_est= self.error_rate*(dt/self.coherence)
        p= min(p_est,0.01)
        if p>1e-15:
            dmat= (1-p)*self.rho + p*(np.identity(self.dim)/ self.dim)
            self.rho= dmat
            self.err_count+=1
    def stats(self)->Dict[str,Any]:
        return {
            "op_count": self.op_count,
            "err_count": self.err_count,
            "err_rate":0 if self.op_count==0 else self.err_count/self.op_count,
            "coherence_left": max(0,self.coherence-(time.time()-self.last_access))
        }

class HPCMultiQuditRegister:
    def __init__(self, size:int, dim:int, prot:int, mft:HPCMFT):
        self.size= size
        self.dim= dim
        self.prot= prot
        self.mft= mft
        self.qdts= [HPCQudit(dim, prot, mft) for _ in range(size)]
    def apply_gate_to_all(self):
        for q in self.qdts:
            rG= random_unitary(self.dim)
            q.apply_gate(rG)
    def run_surface_code_checks(self, thr: float):
        sc= surface_code_check(self.dim)
        for q in self.qdts:
            dt= time.time()- q.last_access
            q._maybe_decohere(dt)
            for cM in sc:
                q.rho= apply_stabilizer(q.rho,cM,thr)
                q.op_count+=1
    def measure_all(self)->List[int]:
        outs=[]
        for q in self.qdts:
            outs.append(q.measure())
        return outs

def single_qudit_run(cfg:HPCConfig, trials:int)-> dict:
    success=0
    mft= HPCMFT(cfg)
    for _ in range(trials):
        qu= HPCQudit(cfg.dimension, cfg.protection_level, mft)
        gate= random_unitary(cfg.dimension)
        qu.apply_gate(gate)
        scops= surface_code_check(cfg.dimension)
        for cM in scops:
            qu.rho= apply_stabilizer(qu.rho,cM, cfg.error_correction_threshold)
        if qu.measure()==0:
            success+=1
    ret= {"fidelity": success/trials}
    return ret

def multi_qudit_run(cfg:HPCConfig, reg_size:int,trials:int)-> dict:
    mft= HPCMFT(cfg)
    reg= HPCMultiQuditRegister(reg_size, cfg.dimension, cfg.protection_level, mft)
    success=0
    for _ in range(trials):
        reg.apply_gate_to_all()
        reg.run_surface_code_checks(cfg.error_correction_threshold)
        if reg.qdts[0].measure()==0:
            success+=1
    return {"fidelity": success/trials}

def worker_combo(args):
    (d,p,t,sz,multi,ccfg)= args
    c= HPCConfig(**ccfg)
    c.dimension= d
    c.protection_level= p
    if multi:
        ret= multi_qudit_run(c,sz,t)
    else:
        ret= single_qudit_run(c,t)
    return {"dimension":d,"protection":p,"fidelity": ret["fidelity"]}

def slurm_submit(cfg:HPCConfig):
    scr= f"""#!/bin/bash
#SBATCH --job-name={cfg.job_name}
#SBATCH --time={cfg.slurm_time}
#SBATCH --nodes={cfg.slurm_nodes}
#SBATCH --ntasks={cfg.slurm_ntasks}

python {sys.argv[0]} parallel-sweep --config HPC_slurm_temp.yaml
"""
    fname= f"{cfg.job_name}.slurm"
    with open(fname,"w") as f:
        f.write(scr)
    print("Generated =>",fname)
    print("Submit with: sbatch",fname)

def ml_pipeline(cfg:HPCConfig)->dict:
    best_f=0.0
    best_info={}
    for i in range(cfg.ml_episodes):
        ret= single_qudit_run(cfg,5)
        if ret["fidelity"]> best_f:
            best_f= ret["fidelity"]
            best_info={"episode":i}
            if best_f>=1.0:
                break
    return {"best_fidelity":best_f,"best_info":best_info}

def append_data_to_file(data:dict, base_name:str):
    os.makedirs("research_data",exist_ok=True)
    tstamp= now_str()
    out_name= f"research_data/{base_name}.jsonl"
    row= {"timestamp":tstamp,**data}
    with open(out_name,"a") as f:
        f.write(json.dumps(row)+"\n")

@click.group()
def qcli():
    pass

@qcli.command()
@click.option('--config','-c',default="")
@click.option('--verbose','-v',is_flag=True)
def show_constants(config,verbose):
    if config and os.path.isfile(config):
        with open(config,'r') as ff:
            data= yaml.safe_load(ff)
        for k,v in data.items():
            if k in SESSION_DATA:
                SESSION_DATA[k]= v
    print(f"PSI={PSI} XI={XI} TAU={TAU} EPS={EPSILON} PHI={PHI:.5f}")
    print("Disclaimer: HPC classical simulation only")
    if verbose:
        row={"action":"show_constants","session_data":SESSION_DATA}
        append_data_to_file(row,"verbose_log")

@qcli.command()
@click.option('--file','-f',required=True)
def inject_data(file):
    if not os.path.isfile(file):
        print("No such file",file)
        return
    if file.endswith(".yaml") or file.endswith(".yml"):
        with open(file,'r') as ff:
            dd= yaml.safe_load(ff)
    else:
        with open(file,'r') as ff:
            dd= json.load(ff)
    for k in dd:
        if k in SESSION_DATA:
            SESSION_DATA[k]= dd[k]
        else:
            SESSION_DATA["extra_data"][k]= dd[k]
    print(json.dumps(SESSION_DATA,indent=2))
    row={"action":"inject_data","file":file,"session_data":SESSION_DATA}
    append_data_to_file(row,"verbose_log")

@qcli.command()
@click.option('--config','-c',required=True)
@click.option('--multi','-m',is_flag=True)
@click.option('--size','-s',default=2)
@click.option('--trials','-t',default=10)
@click.option('--verbose','-v',is_flag=True)
def run_sim(config,multi,size,trials,verbose):
    if not os.path.isfile(config):
        print("No config =>",config)
        return
    with open(config,'r') as ff:
        cdata= yaml.safe_load(ff)
    c= HPCConfig(**cdata,verbose=verbose)
    if multi:
        out= multi_qudit_run(c,size,trials)
    else:
        out= single_qudit_run(c,trials)
    print(f"Fidelity => {out['fidelity']:.3f}")
    if c.verbose:
        row={"action":"run_sim","multi":multi,"size":size,"trials":trials,"result":out}
        append_data_to_file(row,"run_sim_log")

@qcli.command(name="parallel-sweep")
@click.option('--config','-c',required=True)
@click.option('--multi','-m',is_flag=True)
@click.option('--size','-s',default=2)
def parallel_sweep(config,multi,size):
    if not os.path.isfile(config):
        print("No config =>",config)
        return
    with open(config,'r') as ff:
        cdata= yaml.safe_load(ff)
    c= HPCConfig(**cdata)
    if c.use_slurm:
        slurm_submit(c)
        return
    combos=[]
    for d in c.param_dims:
        for p in c.param_prots:
            combos.append((d,p,c.param_trials,size,multi,c.__dict__))
    if c.concurrency>1:
        pool= multiprocessing.Pool(c.concurrency)
        results= pool.map(worker_combo, combos)
        pool.close()
        pool.join()
    else:
        results=[worker_combo(x) for x in combos]
    for r in results:
        print(f"Dim={r['dimension']} Prot={r['protection']} Fidelity={r['fidelity']:.3f}")
    if c.param_output:
        outp_file= c.param_output
        if not outp_file.endswith(".json"):
            outp_file+=".json"
        old_data= []
        if os.path.isfile(outp_file):
            with open(outp_file,'r') as f:
                try:
                    old_data= json.load(f)
                except:
                    pass
        new_data= old_data+ results
        with open(outp_file,"w") as f:
            json.dump(new_data,f,indent=2)
        print("Saved =>",outp_file)
        if c.verbose:
            row={"action":"parallel_sweep","results_count":len(results),"saved":outp_file}
            append_data_to_file(row,"parallel_sweep_log")

@qcli.command(name="plot-coherence")
@click.option('--config','-c',required=True)
@click.option('--verbose','-v',is_flag=True)
def plot_coherence(config,verbose):
    if not os.path.isfile(config):
        print("No config =>",config)
        return
    with open(config,'r') as ff:
        cdata= yaml.safe_load(ff)
    c= HPCConfig(**cdata,verbose=verbose)
    mft= HPCMFT(c)
    lv=[x[0] for x in mft.protection_levels]
    ct=[x[2] for x in mft.protection_levels]
    plt.figure(figsize=(8,5))
    plt.plot(lv,ct,'ro-')
    plt.xlabel("Protection Level")
    plt.ylabel("Coherence (s)")
    plt.title("HPC-based Coherence vs. Protection")
    plt.grid(True)
    fn="coherence_plot.png"
    plt.savefig(fn)
    print("Saved =>",fn)
    if c.verbose:
        row={"action":"plot_coherence","file":fn,"levels":lv,"coherences":ct}
        append_data_to_file(row,"plot_coherence_log")

@qcli.command(name="ml-optimize")
@click.option('--config','-c',required=True)
@click.option('--verbose','-v',is_flag=True)
def ml_optimize(config,verbose):
    if not os.path.isfile(config):
        print("No config =>",config)
        return
    with open(config,'r') as ff:
        cdata= yaml.safe_load(ff)
    c= HPCConfig(**cdata,verbose=verbose)
    if c.ml_episodes<=0:
        print("No ML episodes >0 found.")
        return
    ret= ml_pipeline(c)
    print(f"ML best_fidelity= {ret['best_fidelity']:.3f}, best_info={ret['best_info']}")
    if c.verbose:
        row={"action":"ml_optimize","episodes":c.ml_episodes,"result":ret}
        append_data_to_file(row,"ml_optimize_log")

@qcli.command(name="distribute-sweep")
@click.option('--clusters','-C',default=2)
@click.option('--dims','-D',default='3,4')
@click.option('--prots','-P',default='3,4,5')
@click.option('--verbose','-v',is_flag=True)
def distribute_sweep(clusters,dims,prots,verbose):
    dd= [int(x.strip()) for x in dims.split(',')]
    pp= [int(x.strip()) for x in prots.split(',')]
    combos=[]
    for d in dd:
        for p in pp:
            combos.append((d,p))
    chunk_size= max(1,len(combos)//clusters)
    idx=0
    node=0
    while idx<len(combos):
        ch= combos[idx: idx+chunk_size]
        print(f"Node {node} => {ch}")
        node+=1
        idx+=chunk_size
    if verbose:
        row={"action":"distribute_sweep","clusters":clusters,"dims":dd,"prots":pp,"combo_count":len(combos)}
        append_data_to_file(row,"distribute_sweep_log")

@qcli.command(name="glitch-image")
@click.option('--input','-i',required=True)
@click.option('--output','-o',default="glitched_output.png")
@click.option('--dimension','-d',default=3)
@click.option('--protection','-p',default=3)
@click.option('--block-size','-b',default=8)
@click.option('--noise-model','-n',default='gaussian')
@click.option('--temperature','-T',default=0.1)
@click.option('--doping','-D',default=0.05)
@click.option('--randomize','-R',is_flag=True)
@click.option('--verbose','-v',is_flag=True)
def glitch_image(input,output,dimension,protection,block_size,noise_model,temperature,doping,randomize,verbose):
    if not os.path.isfile(input):
        print(f"No such file => {input}")
        return
    doping_layer= [{"dopant_conc": doping}]
    c= HPCConfig(dimension=dimension,
                 protection_level=protection,
                 doping_randomize=randomize,
                 doping_strength=0.2,
                 doping_layers=doping_layer,
                 noise_model=noise_model,
                 temperature=temperature,
                 verbose=verbose)
    mft= HPCMFT(c)
    img= Image.open(input).convert("RGB")
    arr= np.array(img,dtype=np.float32)
    h,w,_= arr.shape
    out_arr= arr.copy()
    def glitch_block(block: np.ndarray, qu)-> np.ndarray:
        tot= np.sum(block)
        G= random_unitary(qu.dim)
        qu.apply_gate(G)
        o= qu.measure()
        factor= (1+o)/ qu.dim
        return block*factor
    for yb in range(0,h,block_size):
        for xb in range(0,w,block_size):
            qu= HPCQudit(dimension,protection,mft)
            sub= out_arr[yb:yb+block_size, xb:xb+block_size,:]
            sub2= glitch_block(sub,qu)
            out_arr[yb:yb+block_size, xb:xb+block_size,:]= sub2
    out_img= Image.fromarray(np.clip(out_arr,0,255).astype(np.uint8),"RGB")
    out_img.save(output)
    print(f"Glitched image saved => {output}")
    if verbose:
        row={"action":"glitch_image","input":input,"output":output,"dim":dimension,"prot":protection}
        append_data_to_file(row,"glitch_image_log")

def main():
    qcli()

if __name__=="__main__":
    main()

