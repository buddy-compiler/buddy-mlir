import collections
import logging
import torch
import torch._dynamo
import torch._dynamo.config
torch._dynamo.config.log_level = logging.INFO
torch._dynamo.config.output_code = True

import transformers
from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments, BertConfig

def prepare_inputs(**kwargs):
  vocab_size = kwargs['vocab_size']
  batch_size = kwargs['batch_size']
  sequence_length = kwargs['sequence_length']
  input_ids = torch.randint(vocab_size, (batch_size, sequence_length),
                            dtype=torch.long)
  return input_ids.cuda()

def prepare_train_model(config):
  model = transformers.models.bert.modeling_bert.BertForMaskedLM(config)
  model.train()
  model.to('cuda')
  return model

def train(model, inps):
  loss = model(inps, labels=inps)[0]
  loss.backward()

def timed(fn):
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  result = fn()
  end.record()
  torch.cuda.synchronize()
  return result, start.elapsed_time(end) / 1000


N_ITER = 100

def train_bench():
  config_1_lay = BertConfig(num_hidden_layers=1)

  args = {'vocab_size': 30522,
          'batch_size': 8,
          'sequence_length':128
  }
    
  inputs = prepare_inputs(**args)
  model = prepare_train_model(config_1_lay)
  opt = torch.optim.Adam(model.parameters())
    
  # timing eager 
  eager_times = []
  for i in range(N_ITER):
    opt.zero_grad(True)
    _, time = timed(lambda: train(model, inputs))
    opt.step()
    eager_times.append(time)
    print(f"eager train time {i}: {time}")
  print('=' * 20)

  inputs = prepare_inputs(**args)
  model = prepare_train_model(config_1_lay)
  opt = torch.optim.Adam(model.parameters())
  compiled_train = torch.compile(train, backend="inductor")

  # timing dynamo
  dynamo_times = []
  for i in range(N_ITER):
    opt.zero_grad(True)
    _, time = timed(lambda: compiled_train(model, inputs))
    opt.step()
    dynamo_times.append(time)
    print(f"dynamo train time {i}: {time}")
  print('=' * 20)

if __name__ == "__main__":
  train_bench()
