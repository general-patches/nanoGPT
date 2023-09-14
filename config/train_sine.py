# train a miniature time-series model on a simple sine graph

out_dir = 'out-time-series-sine'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'time-series-sine'
wandb_run_name = 'positionless-run'

dataset = 'time_series_sine'
gradient_accumulation_steps = 1
batch_size = 1
block_size = 64 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 1
n_head = 1
n_embd = 1
dropout = 0
regression = True

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 10
eval_interval = 10
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

compile = False # do not torch compile the model
