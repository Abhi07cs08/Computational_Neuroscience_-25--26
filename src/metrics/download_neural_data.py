import brainscore_vision
import numpy as np
import torch
import os
from matplotlib import image
from torchvision import transforms
import pandas as pd


benchmark = brainscore_vision.load_benchmark('MajajHong2015public.IT-pls')

benchmark_assembly = benchmark._assembly
stimulus_set = benchmark_assembly.stimulus_set
groups = stimulus_set["category_name"].to_list()
pres_index = benchmark_assembly.indexes['presentation']
pres_vals = pres_index.get_level_values('stimulus_id')

stimulus_ids = stimulus_set['stimulus_id'].values
paths = []
neural_rows = []
for stim_id in stimulus_ids:
    paths.append(stimulus_set.get_stimulus(stim_id))
    pos = np.where(pres_vals == stim_id)[0]
    neural_data = np.nanmean(benchmark_assembly.values[pos, :], axis=0)
    neural_data = torch.tensor(np.asarray(neural_data), dtype=torch.float32)
    neural_rows.append(neural_data)

neural_master = torch.stack(neural_rows)
os.makedirs('src/metrics/neural_data', exist_ok=True)
torch.save(neural_master, 'src/metrics/neural_data/averaged_neural_data.pt')
torch.save(groups, 'src/metrics/neural_data/stimulus_categories.pt')

os.makedirs('src/metrics/neural_data/images', exist_ok=True)
for i, path in enumerate(paths):
    print(i)
    img = image.imread(path)
    img = transforms.ToPILImage()(img)
    img.save(f'src/metrics/neural_data/images/{i}.png')

benchmark_dataset = brainscore_vision.load_dataset('MajajHong2015.public')

neural_data = benchmark_dataset
neural_data = neural_data.transpose('presentation', 'neuroid', 'time_bin')
benchmark_data_full = neural_data.sel(region='IT')
da = benchmark_data_full.squeeze('time_bin')
old_index = da.indexes["presentation"]
image_ids = old_index.get_level_values("image_id")
reps      = old_index.get_level_values("repetition")

new_index = pd.MultiIndex.from_arrays(
    [image_ids, reps],
    names=["image_id", "repetition"],
)

da2 = da.copy()
da2 = da2.assign_coords(presentation=("presentation", new_index))

da_u = da2.unstack("presentation")

da_u = da_u.transpose("image_id", "repetition", "neuroid")
matrix = da_u.values

matrix = torch.tensor(np.asarray(matrix), dtype=torch.float32)
torch.save(matrix, 'src/metrics/neural_data/unordered_neural_repetitions.pt')