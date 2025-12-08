# Adapted


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from matplotlib import pyplot, image
from src.utils.model_activations import ModelActivations
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import numpy as np
from scipy import stats
import pandas as pd

def predictivity(x, y, rho_xx, rho_yy):
    assert x.shape == y.shape, "Input and prediction shapes must match"

    n_neurons = x.shape[1]

    raw_corr = np.array([stats.pearsonr(x[:, i], y[:, i])[0] for i in range(n_neurons)])
    denominator = np.sqrt(rho_xx * rho_yy)
    corrected_raw_corr = raw_corr / denominator
    ev = (corrected_raw_corr ** 2) * 100
    return ev

def split_half_reliability(X, n_splits=100, eps=1e-8):
    # X: (S, R, N) may contain NaN
    S, R, N = X.shape
    out = np.zeros((n_splits, N))

    for k in range(n_splits):
        idx = np.random.permutation(R)
        h1 = idx[:R//2]
        h2 = idx[R//2:]

        # nan-safe half-averages (S, N)
        m1 = np.nanmean(X[:, h1, :], axis=1)
        m2 = np.nanmean(X[:, h2, :], axis=1)

        # compute correlation per neuron, stimulus-wise ignoring NaNs
        m1c = m1 - np.nanmean(m1, axis=0, keepdims=True)
        m2c = m2 - np.nanmean(m2, axis=0, keepdims=True)

        num = np.nansum(m1c * m2c, axis=0)
        den = np.sqrt(np.nansum(m1c**2, axis=0) *
                      np.nansum(m2c**2, axis=0)) + eps

        r = num / den

        # Spearman–Brown
        out[k] = (2*r) / (1+r+eps)

    return out.mean(axis=0)

def filter_reliable(benchmark_dataset, reliability_threshold=0.7):
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
    r = split_half_reliability(matrix)
    mask = r>=0.7
    r = np.where(mask, r, np.nan)
    # print(r)
    # bool_mask = np.isfinite(r)
    # # print(bool_mask)
    # # print(mask)
    # r_mean = r[r>=reliability_threshold].mean()
    r_means = r[r>=reliability_threshold]
    return mask, r_means

class NeuralDataStimuli(Dataset):
    def __init__(self, benchmark_assembly, trnsfrms=None):
        self.benchmark_assembly = benchmark_assembly
        self.stimulus_set = benchmark_assembly.stimulus_set
        self.transform = trnsfrms
        self.groups = self.stimulus_set["category_name"].tolist()
        self.pres_index = self.benchmark_assembly.indexes['presentation']
        self.pres_vals = self.pres_index.get_level_values('stimulus_id')

    def __len__(self):
        return len(self.stimulus_set)

    def __getitem__(self, idx):
        stimulus_id = self.stimulus_set['stimulus_id'].values[idx]
        local_path = self.stimulus_set.get_stimulus(stimulus_id)
        img = image.imread(local_path)
        img = transforms.ToPILImage()(img)
        trsfm = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
        if self.transform:
            img = self.transform(img)
        else:
            img = trsfm(img)
########################################################################################
        # --- obtain neural response rows that correspond to this stimulus_id ---
        # The assembly's 'presentation' coord is a MultiIndex (e.g. image_id, repetition, stimulus_id, ...)
        # We find all presentation rows with this stimulus_id and average across repetitions.
        # da = self.benchmark_assembly
        # squeeze time_bin if present and singleton

        # print("DA dims:", da.dims)

        # if 'time_bin' in da.dims and da.sizes['time_bin'] == 1:
        #     print("Squeezing time_bin dimension")
        #     da = da.squeeze('time_bin')

        # # try to get 'stimulus_id' level, fall back to 'image_id'
        # try:
        #     print("Accessing presentation index")
        #     pres_index = da.indexes['presentation']
        #     if 'stimulus_id' in pres_index.names:
        #         print("Using 'stimulus_id' level in presentation index")
        #         pres_vals = pres_index.get_level_values('stimulus_id')
        #     else:
        #         print("Using 'image_id' level in presentation index")
        #         pres_vals = pres_index.get_level_values('image_id')
        # except Exception:
        #     print("Failed to access presentation index levels")
        #     # fallback: assume presentation coords are simple and aligned by position
        #     pres_vals = None

        # pres_index = da.indexes['presentation']
        # pres_vals = pres_index.get_level_values('stimulus_id')

        # if pres_vals is not None:
        #     print(f"Finding positions matching stimulus_id {stimulus_id}")
        #     # find positions matching this stimulus_id
        #     pos = np.where(pres_vals == stimulus_id)[0]
        #     if pos.size == 0:
        #         print(f"No positions found for stimulus_id {stimulus_id}, using idx {idx} as fallback")
        #         # as a last resort, try using the idx as position
        #         neural_rows = da.values[idx]
        #     else:
        #         print(f"Found positions for stimulus_id {stimulus_id}: {pos}")
        #         # average over matching presentation rows (repetitions)
        #         neural_rows = np.nanmean(da.values[pos, :], axis=0)
        # else:
        #     print(f"Using idx {idx} as fallback for neural response")
        #     # fallback: index by position
        #     neural_rows = da.values[idx]

        pos = np.where(self.pres_vals == stimulus_id)[0]
        neural_rows = np.nanmean(self.benchmark_assembly.values[pos, :], axis=0)

        # ensure tensor type for downstream code
        neural_response = torch.tensor(np.asarray(neural_rows), dtype=torch.float32)
##############################################################################################
        return img, neural_response
    
def fetch_activation_matrices(model, dataset,activations_recorder, activation_layer="fc1", transforms=None, num_iterations=None, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("using device:", device)
    model = model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_neural_responses = []
    all_model_hidden_activations = []

    model.eval()
    for i, (images, neural_responses) in enumerate(dataloader):
        if num_iterations is not None and i >= num_iterations:
            break
        # move inputs to device, run model, then immediately detach+move outputs to CPU
        images = images.to(device)
        neural_responses = neural_responses.to(device)
        # disable autograd during inference to reduce memory usage
        with torch.no_grad():
            _ = model(images)

        activations_batch = activations_recorder.fetch_activations(activation_layer)
        if activations_batch is None:
            raise RuntimeError(f"No activations captured for layer '{activation_layer}'. Check the layer name and ModelActivations hooks.")
        # If convolutional feature maps (B, C, H, W) -> global pool to (B, C)
        if activations_batch.ndim == 4:
            activations_batch = torch.nn.functional.adaptive_avg_pool2d(activations_batch, 1)
            activations_batch = activations_batch.view(activations_batch.size(0), -1)

        # detach and move to CPU to avoid accumulating GPU tensors across batches
        try:
            activations_cpu = activations_batch.detach().cpu()
        except Exception:
            # if activations_batch is a tuple/list, move each element
            if isinstance(activations_batch, (list, tuple)):
                activations_cpu = type(activations_batch)(
                    a.detach().cpu() if hasattr(a, 'detach') else a for a in activations_batch)
            else:
                activations_cpu = activations_batch

        neural_cpu = neural_responses.detach().cpu()

        all_neural_responses.append(neural_cpu)
        all_model_hidden_activations.append(activations_cpu)
    all_neural_responses = torch.cat(all_neural_responses, dim=0)
    all_model_hidden_activations = torch.cat(all_model_hidden_activations, dim=0)
    all_model_hidden_activations = all_model_hidden_activations.view(all_model_hidden_activations.size(0), -1)
    return all_neural_responses, all_model_hidden_activations
        

def ridge_regression(X, Y, alpha=1.0):
    clf = Ridge(alpha=alpha)
    if torch.is_tensor(X):
        X = X.cpu().numpy()
    if torch.is_tensor(Y):
        Y = Y.cpu().numpy()
    clf.fit(X, Y)
    # W = torch.tensor(clf.coef_, device=X.device).T
    return clf

def F_R_EV(model, activation_layer="fc1", alpha=1, transforms=None, num_iterations=None, splits=10, reliability_threshold=0.7, batch_size=128):
    benchmark = brainscore_vision.load_benchmark('MajajHong2015public.IT-pls')
    benchmark_dataset = brainscore_vision.load_dataset('MajajHong2015.public')

    benchmark_assembly = benchmark._assembly
    dataset = NeuralDataStimuli(benchmark_assembly, trnsfrms=transforms)
    bool_mask, r_means = filter_reliable(benchmark_dataset, reliability_threshold=reliability_threshold)
    # print(bool_mask)
    groups = dataset.groups
    skf = StratifiedKFold(n_splits=splits)
    idx = range(len(dataset))

    F_EV_list = []
    R_EV_list = []

    # all_true = []
    # all_pred = []

    activations_recorder = ModelActivations(model, layers=[activation_layer])
    activations_recorder.register_hooks()

    for train_index, test_index in skf.split(idx, groups):
        dataset_train = torch.utils.data.Subset(dataset, train_index)
        dataset_test = torch.utils.data.Subset(dataset, test_index)
    
        n_train, m_train = fetch_activation_matrices(model, dataset_train, activations_recorder, activation_layer=activation_layer, transforms=transforms, num_iterations=num_iterations, batch_size=batch_size)
        n_train, m_train = n_train.cpu().numpy(), m_train.cpu().numpy()
        n_train_mean, n_train_std = n_train.mean(axis=0, keepdims=True), n_train.std(axis=0, keepdims=True) + 1e-10
        m_train_mean, m_train_std = m_train.mean(axis=0, keepdims=True), m_train.std(axis=0, keepdims=True) + 1e-10
        n_train = (n_train - n_train_mean) / n_train_std
        m_train = (m_train - m_train_mean) / m_train_std
        n_train_filtered = n_train[:, bool_mask]
        n_train_filtered_mean, n_train_filtered_std = n_train_filtered.mean(axis=0, keepdims=True), n_train_filtered.std(axis=0, keepdims=True) + 1e-10
        n_train_filtered = (n_train_filtered - n_train_filtered_mean) / n_train_filtered_std

        # neural_respones_mean, model_activations_mean = neural_responses_train.mean(axis=0, keepdims=True), model_activations_train.mean(axis=0, keepdims=True)
        # neural_responses_std, model_activations_std = neural_responses_train.std(axis=0, keepdims=True) + 1e-10, model_activations_train.std(axis=0, keepdims=True) + 1e-10
        # neural_responses_train = (neural_responses_train - neural_respones_mean) / neural_responses_std
        # model_activations_train = (model_activations_train - model_activations_mean) / model_activations_std

        clf_F = ridge_regression(m_train, n_train_filtered, alpha)
        clf_R = ridge_regression(n_train_filtered, m_train, alpha)

        n_test, m_test = fetch_activation_matrices(model, dataset_test, activations_recorder, activation_layer=activation_layer, transforms=transforms, num_iterations=num_iterations, batch_size=batch_size)
        n_test, m_test = n_test.cpu().numpy(), m_test.cpu().numpy()
        # neural_responses_test = (neural_responses_test - neural_respones_mean) / neural_responses_std
        # model_activations_test = (model_activations_test - model_activations_mean) / model_activations_std
        n_test = (n_test - n_train_mean) / n_train_std
        m_test = (m_test - m_train_mean) / m_train_std
        n_test_filtered = n_test[:, bool_mask]
        n_test_filtered = (n_test_filtered - n_train_filtered_mean) / n_train_filtered_std
        # print("neural responses test shape:", n_test.shape)
        # print("model activations test shape:", m_test.shape)
        model_pred = clf_R.predict(n_test_filtered)
        neural_pred = clf_F.predict(m_test)

        # if r2_mode:
        #     F_EV = r2(n_test_filtered, neural_pred)
        #     # R_EV = r2(model_activations_test, model_pred)
        # else:
        #     F_EV = ev(n_test_filtered, neural_pred)
        #     # R_EV = ev(model_activations_test, model_pred)
        # F_train_EV = predictivity(n_train_filtered, clf_F.predict(m_train), rho_xx=r_mean, rho_yy=1.0).mean()
        # R_train_EV = predictivity(m_train, clf_R.predict(n_train_filtered), rho_xx=1.0, rho_yy=1.0).mean()
        # F_EV = predictivity(n_test_filtered, neural_pred, rho_xx=r_mean, rho_yy=1.0).mean()
        # R_EV = predictivity(m_test, model_pred, rho_xx=1.0, rho_yy=1.0).mean()
        # print(n_test_filtered.shape, neural_pred.shape, r_means.shape)
        # F_train_EV = predictivity(n_train_filtered, clf_F.predict(m_train), rho_xx=r_means, rho_yy=1.0).mean()
        # R_train_EV = predictivity(m_train, clf_R.predict(n_train_filtered), rho_xx=1.0, rho_yy=1.0).mean()
        F_EV = predictivity(n_test_filtered, neural_pred, rho_xx=r_means, rho_yy=1.0).mean()
        R_EV = predictivity(m_test, model_pred, rho_xx=1.0, rho_yy=1.0).mean()

        F_EV_list.append(F_EV)
        R_EV_list.append(R_EV)
        # print(f"Original Fit Results -- F_train_EV: {F_train_EV:.4f} R_train_EV: {R_train_EV:.4f}")
        print(f"Fold results -- F_EV: {F_EV:.4f} R_EV: {R_EV:.4f}")
        # print("neural-nerual test ev:", ev(n_test, n_test))
        # print('neural-predicted neural train ev', ev(n_train, clf_F.predict(m_train)))

    F_EV = sum(F_EV_list) / len(F_EV_list)
    R_EV = sum(R_EV_list) / len(R_EV_list)

    BPI = (2*F_EV*R_EV)/(F_EV + R_EV + 1e-10)

    scores = {'BPI': BPI, 'F_EV': F_EV, 'R_EV': R_EV}


    # all_true = np.vstack(all_true)
    # all_pred = np.vstack(all_pred)

    # neuronwise_ev = neuronwise_corr_ev(all_true, all_pred)
    # scores['neuronwise_ev'] = neuronwise_ev

    return scores

stimuli_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: transforms.functional.rgb_to_grayscale(x, num_output_channels=1)),
    transforms.Normalize(mean=[0.286], std=[0.352]),
    ])
three_channel_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

if __name__ == "__main__":

    from src.models.basic_cnn import BasicCNN
    import brainscore_vision
    from torchvision import transforms
    from  torchvision import models

    resnet18 = models.resnet18(pretrained=True)
    cnn = BasicCNN(in_channels=1, out_channels=10)
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')


    EV_score = F_R_EV(cnn, activation_layer="fc1", alpha=0.1, transforms=stimuli_transform, reliability_threshold=0.7)
    print(EV_score)
    # EV_score_resnet_50 = F_R_EV(benchmark, model, activation_layer="layer4.0.bn1", alpha=0.5, transforms=three_channel_transform, reliability_threshold=0.7, batch_size=4)
    # print(EV_score_resnet_50)

    # EV_score_resnet = F_R_EV(benchmark, resnet18, activation_layer="layer4.0.bn1", alpha=0.1, transforms=three_channel_transform, reliability_threshold=0.7)
    # print(EV_score_resnet)
