from pathlib import Path

DATA_DIR = Path('/path/to/')

# encoder model
img_h = 32
img_w = 32
kernel_size_1 = 5
out_ch_1 = 20
kernel_size_2 = 5
out_ch_2 = 50

px = 0.9
py = 0.9

lr_discriminator = 1e-3
lr_target_encoder = 1e-5

#lr_discriminator = 1e-2
#lr_target_encoder = 1e-6

