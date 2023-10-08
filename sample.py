"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from model import GPTConfig, GPT
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
regression = True
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
# out_dir = 'out' # ignored if init_from is not 'resume'
dataset = 'sunspot'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
num_start_samples = 50
prediction_length = 10
exec(open('configurator.py').read()) # overrides from command line or config file

data = None
mean = None
std = None
if regression:
    temperature = 1.0
    data = np.memmap(os.path.join('data', dataset, 'test.bin'), dtype=np.float32, mode='r')
    max_new_tokens = len(data) - num_start_samples
    start = data[:num_start_samples]
    with open(os.path.join('data', dataset, 'meta.txt'), 'r') as f:
        mean = float(f.readline().split('=')[1])
        std = float(f.readline().split('=')[1])
# -----------------------------------------------------------------------------

def restore_scale(input):
    return input * std + mean

def mean_absolute_error_naive(actual):
    sum = 0
    for i in range(1, len(actual)):
        sum += abs(actual[i] - actual[i - 1])
    return sum / (len(actual) - 1)

def mean_absolute_scaled_error(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    naive_mae = mean_absolute_error_naive(actual)
    return mae / naive_mae

def symmetric_mean_absolute_percentage_error(acutal, predicted):
    return 100 * np.mean(np.abs(predicted - acutal) / ((np.abs(acutal) + np.abs(predicted)) / 2))

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(f"out-{dataset}", 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    # print("No meta.pkl found, assuming GPT-2 encodings...")
    # enc = tiktoken.get_encoding("gpt2")
    # encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    # decode = lambda l: enc.decode(l)
    print('using custom encodings')
    encode = lambda s: s
    decode = lambda l: l

# encode the beginning of the prompt
# if start.startswith('FILE:'):
#     with open(start[5:], 'r', encoding='utf-8') as f:
#         start = f.read()
start_ids = encode(start)

if regression:
    x = torch.tensor(start_ids, dtype=torch.float32, device=device)[None, ...]
else:
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

data = torch.tensor(data, dtype=torch.float32, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        if regression:
            current = model.generate(x, prediction_length, temperature=temperature, top_k=top_k)
            output = current.clone()
            for i in range(num_start_samples, data.shape[1] - prediction_length):
                current = data[:, :i + 1]
                current = model.generate(current, prediction_length, temperature=temperature, top_k=top_k)
                output = torch.cat((output, current[:, -1:]), 1)
            y_actual = data[0].cpu().numpy()
            y_actual = restore_scale(y_actual)
            y_predicted = output[0].cpu().numpy()
            y_predicted = restore_scale(y_predicted)
            mse = mean_squared_error(y_actual, y_predicted)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_actual, y_predicted)
            smape = symmetric_mean_absolute_percentage_error(y_actual, y_predicted)
            smape2 = symmetric_mean_absolute_percentage_error2(y_actual, y_predicted)
            smape3 = symmetric_mean_absolute_percentage_error3(y_actual, y_predicted)
            mae = mean_absolute_error(y_actual, y_predicted)
            mase = mean_absolute_scaled_error(y_actual, y_predicted)
            mase2 = mean_absolute_scaled_error2(y_actual, y_predicted)
            r2 = r2_score(y_actual, y_predicted)
            print('MSE: ', mse)
            print('RMSE: ', rmse)
            print('MAPE: ', mape)
            print('SMAPE: ', smape)
            print('SMAPE2: ', smape2)
            print('SMAPE3: ', smape3)
            print('MAE: ', mae)
            print('MASE: ', mase)
            print('MASE2: ', mase2)
            print('R2: ', r2)
            plt.plot(y_actual)
            plt.plot(y_predicted, linestyle='dashed')
            plt.show()
        else:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------')
