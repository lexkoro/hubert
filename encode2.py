import numpy as np
import torch
import torchaudio
from torchaudio import transforms
from tqdm import tqdm

hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()

downsample = transforms.Resample(
    orig_freq=22050,
    new_freq=16000,
    resampling_method="kaiser_window",
    dtype=torch.float32,
)

filelist = [
    "/home/akorolev/master/projects/vits-emotts/filelists/emotional_train.txt",
    "/home/akorolev/master/projects/vits-emotts/filelists/deepph_train_de.txt_cleaned",
    "/home/akorolev/master/projects/vits-emotts/filelists/deepph_train_en.txt_cleaned",
    "/home/akorolev/master/projects/vits-emotts/filelists/deepph_train_pl.txt_cleaned",
    "/home/akorolev/master/projects/vits-emotts/filelists/deepph_train_de_kcd.txt_cleaned",
    "/home/akorolev/master/projects/vits-emotts/filelists/deepph_test.txt_cleaned",
]

all_paths = []

for f in filelist:
    with open(f, "r") as rf:
        for line in rf:
            audiofile = line.split("|")[0]
            all_paths.append(audiofile)


for in_path in tqdm(all_paths):
    wav, sr = torchaudio.load(in_path)
    wav = downsample(wav)
    wav = wav.unsqueeze(0).cuda()

    with torch.inference_mode():
        units = hubert.units(wav)

    unit_path = in_path.replace(".wav", ".hubert.npy")
    print("Saving to", unit_path)
    np.save(unit_path, units.squeeze().cpu().numpy())
