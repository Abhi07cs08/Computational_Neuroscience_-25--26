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
import os

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

def filter_reliable(neural_data_dir="src/metrics/neural_data", reliability_threshold=0.7):
    matrix = torch.load(os.path.join(neural_data_dir, 'unordered_neural_repetitions.pt'))
    print(matrix.shape)
    r = split_half_reliability(matrix)
    mask = r>=0.7
    r = np.where(mask, r, np.nan)
    r_means = r[r>=reliability_threshold]
    return mask, r_means

class NeuralDataStimuli(Dataset):
    def __init__(self, neural_data_dir="src/metrics/neural_data", trnsfrms=None):
        self.neural_data_dir = neural_data_dir
        self.neural_data = torch.load(os.path.join(neural_data_dir, 'averaged_neural_data.pt'))
        self.stimulus = os.listdir(os.path.join(neural_data_dir, 'images'))
        self.stimulus.sort(key=lambda x: int(x.split('.')[0]))
        self.transform = trnsfrms
        self.groups = torch.load(os.path.join(neural_data_dir, 'stimulus_categories.pt'))

    def __len__(self):
        return len(self.stimulus)

    def __getitem__(self, idx):
        stimulus_id = self.stimulus[idx]
        local_path = os.path.join(self.neural_data_dir, 'images', stimulus_id)
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
        neural_rows = self.neural_data[idx, :]
        neural_response = torch.tensor(np.asarray(neural_rows), dtype=torch.float32) if not torch.is_tensor(neural_rows) else neural_rows
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

def F_R_EV(model, activation_layer="fc1", neural_data_dir="src/metrics/neural_data", alpha=1, transforms=None, num_iterations=None, splits=10, reliability_threshold=0.7, batch_size=128):
    dataset = NeuralDataStimuli(neural_data_dir=neural_data_dir, trnsfrms=transforms)
    bool_mask, r_means = filter_reliable(neural_data_dir=neural_data_dir, reliability_threshold=reliability_threshold)
    groups = dataset.groups
    skf = StratifiedKFold(n_splits=splits)
    idx = range(len(dataset))

    F_EV_list = []
    R_EV_list = []

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

        clf_F = ridge_regression(m_train, n_train_filtered, alpha)
        clf_R = ridge_regression(n_train_filtered, m_train, alpha)

        n_test, m_test = fetch_activation_matrices(model, dataset_test, activations_recorder, activation_layer=activation_layer, transforms=transforms, num_iterations=num_iterations, batch_size=batch_size)
        n_test, m_test = n_test.cpu().numpy(), m_test.cpu().numpy()

        n_test = (n_test - n_train_mean) / n_train_std
        m_test = (m_test - m_train_mean) / m_train_std
        n_test_filtered = n_test[:, bool_mask]
        n_test_filtered = (n_test_filtered - n_train_filtered_mean) / n_train_filtered_std

        model_pred = clf_R.predict(n_test_filtered)
        neural_pred = clf_F.predict(m_test)

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
    from torchvision import transforms
    from torchvision import models

    resnet18 = models.resnet18(pretrained=True)
    cnn = BasicCNN(in_channels=1, out_channels=10)
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')


    EV_score = F_R_EV(cnn, activation_layer="fc1", neural_data_dir="src/metrics/neural_data", alpha=0.1, transforms=stimuli_transform, reliability_threshold=0.7)
    print(EV_score)
    # EV_score_resnet_50 = F_R_EV(benchmark, model, activation_layer="layer4.0.bn1", alpha=0.5, transforms=three_channel_transform, reliability_threshold=0.7, batch_size=4)
    # print(EV_score_resnet_50)

    # EV_score_resnet = F_R_EV(benchmark, resnet18, activation_layer="layer4.0.bn1", alpha=0.1, transforms=three_channel_transform, reliability_threshold=0.7)
    # print(EV_score_resnet)
