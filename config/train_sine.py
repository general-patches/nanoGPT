# train a miniature time-series model on a simple sine graph

out_dir = 'out-time-series-sine'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'time-series-sine'
wandb_run_name = 'positionless-run'

dataset = 'time-series-sine'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 2
n_head = 6
n_embd = 18
dropout = 0
regression = True
position_encoding = 'random'

learning_rate = 5e-4 # with baby networks can afford to go a bit higher
max_iters = 1000
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 5e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

compile = False # do not torch compile the model
