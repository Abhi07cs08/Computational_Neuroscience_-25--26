import numpy as np
import brainscore_vision
from pathlib import Path
from matplotlib import image
from torchvision import transforms


def save_subject_cache(subject_data, subject_label, base_cache_dir, benchmark, skip_stimuli=False):
    subject_data = subject_data.squeeze('time_bin') if 'time_bin' in subject_data.dims else subject_data

    neural_data_full_it = subject_data.values.T  # (n_presentations, n_neuroids)
    stimulus_ids_it = subject_data.stimulus_id.values
    n_neuroids = subject_data.shape[0]

    unique_stimuli, _ = np.unique(stimulus_ids_it, return_inverse=True)

    reps_per_image = np.array([(stimulus_ids_it == stim_id).sum() for stim_id in unique_stimuli])
    max_reps = int(np.max(reps_per_image))
    n_images = len(unique_stimuli)

    neural_activations = np.full((n_images, n_neuroids, max_reps), np.nan, dtype=np.float32)

    for img_idx, stim_id in enumerate(unique_stimuli):
        mask = stimulus_ids_it == stim_id
        stim_presentations = neural_data_full_it[mask, :]  # (n_reps_this_image, n_neuroids)
        n_reps_this = stim_presentations.shape[0]
        neural_activations[img_idx, :, :n_reps_this] = stim_presentations.T


    subject_cache_dir = base_cache_dir / str(subject_label)
    subject_cache_dir.mkdir(parents=True, exist_ok=True)

    if not skip_stimuli:
        images_cache_dir = subject_cache_dir / "images"
        images_cache_dir.mkdir(parents=True, exist_ok=True)

    np.save(subject_cache_dir / "neural_activations.npy", neural_activations)

    if not skip_stimuli:
        np.save(subject_cache_dir / "stimulus_ids.npy", unique_stimuli, allow_pickle=True)
        for stim_id in unique_stimuli:
            image_dest_path = images_cache_dir / f"{stim_id}.png"
            if image_dest_path.exists():
                continue

            image_source_path = benchmark._assembly.stimulus_set.get_stimulus(stim_id)
            img = image.imread(image_source_path)
            img = transforms.ToPILImage()(img)
            img.save(image_dest_path)

    print(
        f"Saved subject={subject_label} | images={len(unique_stimuli)} | "
        f"neuroids={n_neuroids} | max_reps={max_reps}"
    )


def main():
    benchmark = brainscore_vision.load_benchmark('MajajHong2015public.IT-pls')
    neural_data_full = brainscore_vision.load_dataset("MajajHong2015.public")
    neural_data_it = neural_data_full.sel(region='IT')

    if 'subject' in neural_data_it.coords:
        split_key = 'subject'
        subjects = np.unique(neural_data_it[split_key].values)
        get_subject_data = lambda label: neural_data_it.where(neural_data_it[split_key] == label, drop=True)
    else:
        neuroid_index = neural_data_it.indexes.get('neuroid')
        level_names = list(neuroid_index.names) if hasattr(neuroid_index, 'names') else []

        split_key = None
        for candidate in ('animal', 'subject', 'monkey'):
            if candidate in level_names:
                split_key = candidate
                break

        if split_key is None:
            raise ValueError(
                "Could not find monkey identifier. Expected one of "
                "('subject', 'animal', 'monkey') in coords or neuroid index levels."
            )

        subjects = np.unique(neuroid_index.get_level_values(split_key))

        def get_subject_data(label):
            mask = np.asarray(neuroid_index.get_level_values(split_key) == label)
            return neural_data_it.isel(neuroid=mask)

    base_cache_dir = Path("src/latest_neural_data/majajhong_cache_by_monkey")
    base_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found subjects by '{split_key}': {list(subjects)}")

    for subject_label in subjects:
        subject_data = get_subject_data(subject_label)
        save_subject_cache(subject_data, subject_label, base_cache_dir, benchmark)


if __name__ == "__main__":
    main()
