# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-eng-fr-distill-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# validation code currently meaningless for distillation, so always save a checkpoint regardless of validation loss
always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'multilingual-distillation-nanogpt'
wandb_run_name = 'english-french-same-arch-as-teachers'

dataset = 'eng_fr_plays_char/joint_data' # this is the path to the train.bin and val.bin files
# dataset = 'english/shakespeare' # this is the path to the train.bin and val.bin files
# dataset = 'french/TheatreClassique' # this is the path to the train.bin and val.bin files
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 15000
lr_decay_iters = 1e6 # don't want LR decay, so make this very large
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# warmup_iters = 100 # not super necessary potentially
warmup_iters = 0 # no warmup for distillation, just start with the learning rate

eng_teacher_path = "/nobackup/users/scarv/multi-teacher-distillation/nanoGPT-distill/out-shakespeare-char"
fr_teacher_path = "/nobackup/users/scarv/multi-teacher-distillation/nanoGPT-distill/out-theatre-classique-char"

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
