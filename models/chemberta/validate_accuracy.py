#!/usr/bin/env python3
import argparse,json,os,numpy as np,torch
from transformers import AutoModelForMaskedLM
def gen(d):os.makedirs(d,exist_ok=True);m=AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM",dtype=torch.float32).eval();x=torch.zeros((1,128),dtype=torch.int64);a=torch.ones((1,128),dtype=torch.int64);o=m(input_ids=x,attention_mask=a).logits[0,-1,:].detach().cpu().numpy().astype(np.float32);np.save(os.path.join(d,"ref.npy"),o);print(f"saved {o.shape}")
def main():p=argparse.ArgumentParser();p.add_argument("--mode");p.add_argument("--output-dir",default="./val");a=p.parse_args();gen(a.output_dir) if a.mode=="reference" else None
if __name__=="__main__":main()
