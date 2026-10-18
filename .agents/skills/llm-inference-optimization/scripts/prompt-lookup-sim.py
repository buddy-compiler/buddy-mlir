# ===- prompt-lookup-sim.py ----------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Estimate speculative decoding with prompt lookup before building it.
#
#   python3 prompt-lookup-sim.py generate ids.json --model <HF id or dir> \
#       --prompts prompts.json --new-tokens 1024
#   python3 prompt-lookup-sim.py simulate ids.json --verify-cost 1.2,1.5
#
# generate: greedy outputs of a Hugging Face model (CPU is fine for small
# models) for {name: prompt} in prompts.json, with the chat template; saves
# prompt and output token ids.
# simulate: drafts up to k tokens by the most recent earlier match of the
# last n tokens (n from --ngram max down to min) in prompt + output so far;
# greedy verification accepts the matching prefix plus one token. Prints the
# tokens per verification step and the speedup when a step with a draft
# costs --verify-cost decode steps (steps without a draft cost 1).
#
# Look at the outputs: a model stuck in a repetition loop inflates
# acceptance; exclude such prompts.
#
# ===---------------------------------------------------------------------------

import argparse
import json


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    )
    model.eval()
    with open(args.prompts) as f:
        prompts = json.load(f)
    res = {}
    for name, text in prompts.items():
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        with torch.no_grad():
            out = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=args.new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0, ids.shape[1] :].tolist()
        res[name] = {"prompt": ids[0].tolist(), "output": gen}
        print(
            f"{name}: prompt {ids.shape[1]} tokens, output {len(gen)}",
            flush=True,
        )
        print("   ", repr(tok.decode(gen[-200:]))[:200], flush=True)
    with open(args.ids, "w") as f:
        json.dump(res, f)


def draft(ctx, k, nmax, nmin):
    for n in range(nmax, nmin - 1, -1):
        if len(ctx) <= n:
            continue
        pat = ctx[-n:]
        for s in range(len(ctx) - n - 1, -1, -1):
            if ctx[s : s + n] == pat:
                return ctx[s + n : s + n + k]
    return []


def run(prompt, out, k, nmax, nmin):
    """(steps, steps with a draft) to produce `out` after `prompt`."""
    ctx, i, steps, drafted = list(prompt), 0, 0, 0
    while i < len(out):
        dr = draft(ctx, k, nmax, nmin)
        a = 0
        while a < len(dr) and i + a < len(out) and dr[a] == out[i + a]:
            a += 1
        n = min(a + 1, len(out) - i)
        steps += 1
        drafted += bool(dr)
        ctx += out[i : i + n]
        i += n
    return steps, drafted


def simulate(args):
    with open(args.ids) as f:
        data = json.load(f)
    nmax, nmin = (int(x) for x in args.ngram.split(","))
    costs = [float(c) for c in args.verify_cost.split(",")]
    for k in (int(x) for x in args.drafts.split(",")):
        print(f"== up to {k} drafts, n-gram {nmax}..{nmin}")
        tokens = steps = 0
        cost = [0.0] * len(costs)
        for name, r in data.items():
            s, dr = run(r["prompt"], r["output"], k, nmax, nmin)
            tokens += len(r["output"])
            steps += s
            for j, c in enumerate(costs):
                cost[j] += (s - dr) + dr * c
            print(
                f"  {name:14s} {len(r['output']):5d} tokens, {len(r['output']) / s:.2f}"
                f" per step, draft in {dr / s:.0%} of steps"
            )
        speed = ", ".join(
            f"cost {c}: {tokens / x:.2f}x" for c, x in zip(costs, cost)
        )
        print(f"  all: {tokens / steps:.2f} tokens per step; speedup {speed}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("ids")
    g.add_argument("--model", required=True)
    g.add_argument("--prompts", required=True, help="JSON {name: prompt}")
    g.add_argument("--new-tokens", type=int, default=512)
    s = sub.add_parser("simulate")
    s.add_argument("ids")
    s.add_argument("--drafts", default="3,7", help="draft lengths to try")
    s.add_argument("--ngram", default="3,1", help="max,min n-gram length")
    s.add_argument("--verify-cost", default="1.2,1.5")
    args = p.parse_args()
    generate(args) if args.cmd == "generate" else simulate(args)


if __name__ == "__main__":
    main()
