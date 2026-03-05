import torch
import torch.nn.functional as F

def group_sparsity(fc, eps=1e-8):

	if torch.is_tensor(fc):
		weight = fc
	else:
		weight = fc.weight
	G, K = weight.shape
	
	device = weight.device
	total = weight.new_tensor(0.0)
	# print(f"Weight shape: {weight.shape}")
	for idx in range(G):
		w_g = weight[idx, :]  # [|group|, in_dim]
		# print(f"Group weights shape: {w_g.shape}")
		norms = torch.sqrt((w_g * w_g).sum(dim=0) + eps)  # per-latent L2
		total = total + norms.sum()

	return total

def KL_loss(fc, kl_target):
	if torch.is_tensor(fc):
		weight = fc
	else:
		weight = fc.weight
	mean_activations = weight.mean(dim=0)  
	probs = torch.softmax(mean_activations, dim=0)
	KL_div_loss = F.kl_div(probs.log(), kl_target, reduction="batchmean")

	return KL_div_loss

def probs(fc):
	mean_activations = fc.weight.mean(dim=0)  
	probs = torch.softmax(mean_activations, dim=0)
	return probs

def aux_loss(fc, lam):
	if torch.is_tensor(fc):
		weight = fc
	else:
		weight = fc.weight
	groups = weight.shape[0]
	kl_target = torch.ones(groups, device=weight.device) / groups 
	return lam * KL_loss(fc, kl_target=kl_target) + group_sparsity(fc)



if __name__ == "__main__":
	# Example usage
	fc = torch.nn.Linear(10, 4)

	print(f"Probabilities: {probs(fc)}")

	init_loss = group_sparsity(fc)
	optimizer = torch.optim.Adam(fc.parameters(), lr=0.1)
	optimizer.zero_grad()
	print("Initial group sparsity penalty:", init_loss.item())
	print("initial kl_loss:", KL_loss(fc, kl_target=torch.ones(10) / 10))
	for _ in range(5):
		loss = group_sparsity(fc)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
	print("Final group sparsity penalty:", loss.item())
	print("Final kl_loss:", KL_loss(fc, kl_target=torch.ones(10) / 10))
	print(f"Probabilities: {probs(fc)}")

# 
	# kl_loss = KL_loss(fc, kl_target=torch.ones(10) / 10)  # Uniform target
	# print("KL loss:", kl_loss)
	
