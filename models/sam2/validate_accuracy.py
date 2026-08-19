#!/usr/bin/env python3
import argparse,json,os,sys,numpy as np,torch
from transformers import AutoModel
def gen_ref(d):
    os.makedirs(d,exist_ok=True);m=AutoModel.from_pretrained("facebook/sam2-hiera-tiny",dtype=torch.float32).eval()
    ve=m.vision_encoder; pix=torch.zeros((1,3,256,256))
    with torch.no_grad(): out=ve(pix)
    feats=out["last_hidden_state"].detach().cpu().numpy().astype(np.float32)[0,0,0,:]
    ref={"shape":list(out["last_hidden_state"].shape),"sum":float(np.sum(feats))};np.save(os.path.join(d,"ref.npy"),feats)
    with open(os.path.join(d,"manifest.json"),"w") as f:json.dump(ref,f)
def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",required=True,choices=["reference","compare"]);p.add_argument("--output-dir",default="./val");a=p.parse_args()
    if a.mode=="reference":gen_ref(a.output_dir)
if __name__=="__main__":main()
