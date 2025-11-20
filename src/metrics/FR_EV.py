import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from matplotlib import pyplot, image
from src.utils.model_activations import ModelActivations
from sklearn.model_selection import StratifiedKFold

class NeuralDataStimuli(Dataset):
    def __init__(self, benchmark_assembly, trnsfrms=None):
        self.benchmark_assembly = benchmark_assembly
        self.stimulus_set = benchmark_assembly.stimulus_set
        self.transform = trnsfrms
        self.groups = self.stimulus_set["category_name"].tolist()

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
        neural_response = self.benchmark_assembly.data[idx]
        return img, neural_response
    
def fetch_activation_matrices(model, dataset, activation_layer="fc1", transforms=None, num_iterations=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)
    model = model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    all_neural_responses = []
    all_model_hidden_activations = []
    # activations = {}
    # def save_activation(name):
    #     def hook(module, inp, out):
    #         activations[name] = out.detach()
    #     return hook

    # handles = []

    # for name, module in model.named_modules():
    #     if name == activation_layer:
    #         handles.append(module.register_forward_hook(save_activation(name)))

    activations_recorder = ModelActivations(model, layers=[activation_layer])
    activations_recorder.register_hooks()


    model.eval()
    for i, (images, neural_responses) in enumerate(dataloader):
        if num_iterations is not None and i >= num_iterations:
            break
        images = images.to(device)
        neural_responses = neural_responses.to(device)
        _ = model(images)

        activations_batch = activations_recorder.fetch_activations(activation_layer)
        if activations_batch.ndim == 4:
                # global average pool to [B, 512, 1, 1]
            activations_batch = torch.nn.functional.adaptive_avg_pool2d(activations_batch, 1)
        all_neural_responses.append(neural_responses)
        all_model_hidden_activations.append(activations_batch)
    all_neural_responses = torch.cat(all_neural_responses, dim=0)
    all_model_hidden_activations = torch.cat(all_model_hidden_activations, dim=0)
    all_model_hidden_activations = all_model_hidden_activations.view(all_model_hidden_activations.size(0), -1)
    return all_neural_responses, all_model_hidden_activations

def ridge_regression(X, Y, alpha=1.0):
    # """Perform ridge regression to predict Y from X."""
    # n_features = X.shape[1]
    # I = torch.eye(n_features).to(X.device)
    # W = torch.linalg.inv(X.T @ X + alpha * I) @ X.T @ Y
    clf = Ridge(alpha=alpha)
    if torch.is_tensor(X):
        X = X.cpu().numpy()
    if torch.is_tensor(Y):
        Y = Y.cpu().numpy()
    clf.fit(X, Y)
    # W = torch.tensor(clf.coef_, device=X.device).T
    return clf

def ev(Y_true, Y_pred):
    if torch.is_tensor(Y_true):
        Y_true = Y_true.cpu().numpy()
    if torch.is_tensor(Y_pred):
        Y_pred = Y_pred.cpu().numpy()
    # return explained_variance_score(Y_true, Y_pred, multioutput='uniform_average')
    return explained_variance_score(Y_true, Y_pred, multioutput='variance_weighted')

def r2(Y_true, Y_pred):
    if torch.is_tensor(Y_true):
        Y_true = Y_true.cpu().numpy()
    if torch.is_tensor(Y_pred):
        Y_pred = Y_pred.cpu().numpy()
    # return r2_score(Y_true, Y_pred, multioutput='uniform_average')
    return r2_score(Y_true, Y_pred, multioutput='variance_weighted')

def F_R_EV(benchmark, model, activation_layer="fc1", alpha=1, transforms=None, num_iterations=None, r2_mode=True, splits=3):
    benchmark_assembly = benchmark._assembly
    dataset = NeuralDataStimuli(benchmark_assembly, trnsfrms=transforms)
    groups = dataset.groups
    skf = StratifiedKFold(n_splits=splits)
    idx = range(len(dataset))

    F_EV_list = []
    R_EV_list = []

    for train_index, test_index in skf.split(idx, groups):
        dataset_train = torch.utils.data.Subset(dataset, train_index)
        dataset_test = torch.utils.data.Subset(dataset, test_index)
    
        neural_responses_train, model_activations_train = fetch_activation_matrices(model, dataset_train, activation_layer, transforms=transforms, num_iterations=num_iterations)
        neural_responses_train, model_activations_train = neural_responses_train.cpu().numpy(), model_activations_train.cpu().numpy()

        neural_respones_mean, model_activations_mean = neural_responses_train.mean(axis=0, keepdims=True), model_activations_train.mean(axis=0, keepdims=True)
        neural_responses_std, model_activations_std = neural_responses_train.std(axis=0, keepdims=True) + 1e-10, model_activations_train.std(axis=0, keepdims=True) + 1e-10
        neural_responses_train = (neural_responses_train - neural_respones_mean) / neural_responses_std
        model_activations_train = (model_activations_train - model_activations_mean) / model_activations_std

        print("neural responses shape:", neural_responses_train.shape)
        print("model activations shape:", model_activations_train.shape)
        clf_F = ridge_regression(model_activations_train, neural_responses_train, alpha)
        clf_R = ridge_regression(neural_responses_train, model_activations_train, alpha)

        neural_responses_test, model_activations_test = fetch_activation_matrices(model, dataset_test, activation_layer, transforms=transforms, num_iterations=num_iterations)
        neural_responses_test, model_activations_test = neural_responses_test.cpu().numpy(), model_activations_test.cpu().numpy()
        neural_responses_test = (neural_responses_test - neural_respones_mean) / neural_responses_std
        model_activations_test = (model_activations_test - model_activations_mean) / model_activations_std
        print("neural responses test shape:", neural_responses_test.shape)
        print("model activations test shape:", model_activations_test.shape)
        model_pred = clf_R.predict(neural_responses_test)
        neural_pred = clf_F.predict(model_activations_test)

        if r2_mode:
            F_EV = r2(neural_responses_test, neural_pred)
            R_EV = r2(model_activations_test, model_pred)
        else:
            F_EV = ev(neural_responses_test, neural_pred)
            R_EV = ev(model_activations_test, model_pred)
        F_EV_list.append(F_EV)
        R_EV_list.append(R_EV)
        print(f"Fold results -- F_EV: {F_EV:.4f}, R_EV: {R_EV:.4f}")
        print("neural-nerual test ev:", ev(neural_responses_test, neural_responses_test))
        print('neural-predicted neural train ev', ev(neural_responses_train, clf_F.predict(model_activations_train)))

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
    import brainscore_vision
    from torchvision import transforms
    from  torchvision import models

    resnet18 = models.resnet18(pretrained=True)
    cnn = BasicCNN(in_channels=1, out_channels=10)

    benchmark = brainscore_vision.load_benchmark('MajajHong2015public.IT-pls')

    # EV_score = F_R_EV(benchmark, cnn, activation_layer="fc1", alpha=1.0, transforms=stimuli_transform)
    # print(EV_score)

    EV_score_resnet = F_R_EV(benchmark, resnet18, activation_layer="layer4", alpha=0.1, transforms=three_channel_transform, r2_mode=False)
    print(EV_score_resnet)
