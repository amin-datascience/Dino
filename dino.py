import torch 
import torch.nn as nn 
from torch.nn import functional as F 




#Dino Module

class PatchEmbedding(nn.Module):

	def __init__(self, img_size, patch_size, embed_dim, in_channels = 3, early_cnn = True):
		super(PatchEmbedding, self).__init__() 
		self.img_size = img_size 
		self.patch_size = patch_size 
		self.n_patches = (img_size // patch_size) ** 2
		self.early_cnn = early_cnn

		self.cnn = nn.Conv2d(in_channels = in_channels, out_channels = 3, kernel_size = 3, padding = 1)
		self.bn = nn.BatchNorm2d(3)
		self.relu = nn.ReLU()
		self.transform = nn.Conv2d(in_channels = in_channels, out_channels = embed_dim, 
								kernel_size = patch_size, stride = patch_size) 


	def forward(self, x):

		#(n, 3, 32, 32) doesn't change the size of the image
		if self.early_cnn :
			x = self.cnn(x)  
			x = self.bn(x)
			x = self.relu(x)

		x = self.transform(x) #(n_samples, embed_dim, width, height)
		x = x.flatten(2)
		x = x.transpose(1, 2) #(n_samples, n_patches, embed_dim)

		return x



class SelfAttention(nn.Module):
	def __init__(self, dim, n_heads = 8, proj_drop = 0.1):

		super(SelfAttention, self).__init__()
		self.dim = dim 
		self.n_heads = n_heads 
		self.head_dim = dim // n_heads 
		self.scale = n_heads ** -0.5

		self.query = nn.Linear(dim, dim)
		self.key = nn.Linear(dim, dim)
		self.value = nn.Linear(dim, dim)
		self.fc_out = nn.Linear(dim, dim)
		self.fc_drop = nn.Dropout(proj_drop)



	def forward(self, x):
		n_samples, n_patches, dim = x.shape

		assert dim == self.dim, 'dim should be equal to the dimension declared in the constructor'

		q = self.query(x)  #Each with dim: (n_samples, n_patches + 1, dim)
		k = self.key(x)
		v = self.value(x)


		q = q.reshape(n_samples, n_patches, self.n_heads, self.head_dim) #(n_samples, n_patches, self.n_heads, self.head_dim)
		k = k.reshape(n_samples, n_patches, self.n_heads, self.head_dim)      
		v = v.reshape(n_samples, n_patches, self.n_heads, self.head_dim)

		q = q.permute(0, 2, 1, 3) #(n_samples, n_heads, n_patches, head_dim)
		k = k.permute(0, 2, 1, 3)
		v = v.permute(0, 2, 1, 3)

		k_t = k.transpose(-1, -2) #(n_samples, n_heads, head_dim, n_patches)

		weights = (torch.matmul(q, k_t)) * self.scale #(n_samples, n_heads, n_patches, n_patches)

		scores = weights.softmax(dim = -1)

		weighted_avg = scores @ v  #(n_smples, n_heads, n_patches, head_dim)
		weighted_avg = weighted_avg.transpose(1, 2) #(n_smples, n_patches, n_heads, head_dim)

		weighted_avg = weighted_avg.flatten(2) #(n_samples, n_patches, n_heads*head_dim)

		x = self.fc_out(weighted_avg)
		x = self.fc_drop(x) 

		return x 



class MLP(nn.Module):

	def __init__(self, in_features, out_features, drop_mlp):
		super(MLP, self).__init__()

		self.fc1 = nn.Linear(in_features, out_features)
		self.dropout = nn.Dropout(drop_mlp)
		self.fc2 = nn.Linear(out_features, out_features)
		self.gelu = nn.GELU()


	def forward(self, x):
		x = self.fc1(x)  #(n_samples, n_patches, hidden_features)
		x = self.gelu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.dropout(x)

		return x 



class Block(nn.Module):

	def __init__(self,  dim, n_heads, p_drop = 0.1):
		super(Block, self).__init__()

		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)
		self.attention = SelfAttention(dim = dim, n_heads = n_heads)	 

		self.mlp = MLP(in_features = dim, out_features = dim, drop_mlp = p_drop)



	def forward(self, x):
		x = x + self.attention(self.norm1(x))
		x = x + self.mlp(self.norm2(x))

		return x   



class Dino(nn.Module):
	'''
	This implementation of Dino uses a Vision Transformer as its backbone model.

	'''

	def __init__(self, img_size, patch_size = 16, in_channels = 3, n_classes = 10, embed_dim = 768, 
				layers = 6, n_heads = 12, p_drop = 0.0, early_cnn = True):
		super(Dino, self).__init__()

		self.patch_embed = PatchEmbedding(img_size = img_size, patch_size = patch_size, embed_dim = embed_dim, early_cnn = early_cnn)
		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)) 
		self.pos_drop = nn.Dropout(p_drop)

		self.blocks = nn.ModuleList([
			Block(dim = embed_dim, n_heads = n_heads, p_drop = p_drop)
				for _ in range(layers)])

		self.norm = nn.LayerNorm(embed_dim)
		self.head = nn.Linear(in_features = embed_dim, out_features = n_classes)

		nn.init.trunc_normal_(self.pos_embed, std = 0.02)
		nn.init.trunc_normal_(self.cls_token, std = 0.02)

		self.apply(self._init_weights)

	
	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			nn.init.trunc_normal_(m.weight, std = 0.02)

			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.weight, 1.0)
			nn.init.constant_(m.bias, 0)


	def forward(self, x):
		n_samples = x.shape[0]

		x = self.patch_embed(x) #(n_samples, n_pathces, embed_dim)
		cls_token = self.cls_token.expand(n_samples, -1, -1) #(n_samples, 1, embed_dim)


		x = torch.cat((cls_token, x), dim = 1) #(n_samples, 1+n_patches, embed_dim)
		x = x + self.pos_embed
		x = self.pos_drop(x)

		for block in self.blocks:
			x = block(x) 

		x = self.norm(x) 
		cls_embed = x[:, 0]

		x_cls = self.head(cls_embed)

		return x_cls 




class DinoHead(nn.Module):
	"""
	The projection head used in DINO.
	It takes the output of the Vision Transformer and then 
	passes it through a MLP to. 

	Parameters
	----------

	in_dim: int
			The dimension of the embeddings output by the backbone.

	hidden_dim: int
			The dimension of the middle layers of MLP.

	out_dim: int
			The dimension of the final outputs, which will be used for 
			classification.

	n_layers: int
			The number of layers of the MLP.

	Attributes
	----------
	mlp: `torch.nn.Module`
		The trainable MLP.

	last_linear_layer: `torch.nn.Module`
				The last layer of the MLP.
	"""

	def __init__(self, in_dim = 384, hidden_dim = 384, out_dim = 10, n_layers = 3, norm_last_layer = False):
		super(DinoHead, self).__init__()
		self.in_dim = in_dim
		self.hidden_dim = hidden_dim
		self.out_dim = out_dim 
		self.n_layers = n_layers 


		if n_layers == 1: 
			self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim))

		else:
			layers = [nn.Linear(in_dim, hidden_dim)]
			for _ in range(n_layers-2):
				layers.append(nn.Linear(hidden_dim, hidden_dim))
				layers.append(nn.GELU())
			layers.append(nn.Linear(hidden_dim, hidden_dim))

			self.mlp = nn.Sequential(*layers)

		self.apply(self._init_weights)
		self.last_linear_layer = nn.utils.weight_norm(nn.Linear(hidden_dim, out_dim, bias = False))
		self.last_linear_layer.weight_g.data.fill_(1)
		
		if norm_last_layer:
			self.last_linear_layer.weight_g.requires_grad = False


	def _init_weights(self, m):
		"""Initialize learnable parameters."""
		if isinstance(m, nn.Linear):
			nn.init.trunc_normal_(m.weight, std = 0.02)

			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)


	def forward(self, x):
		"""Run the forward pass.

		parameters
		----------
		x: `torch.Tensor` 
			of shape `(n_samples, in_dim)`

		Returns
		-------
		`torch.Tensor` 
			of shape `(n_samples, out_dim)`
		"""

		x = self.mlp(x) #(n_samples, hidden_dim)
		x = nn.functional.normalize(x, dim = -1, p = 2) #(n_samples, hidden_dim)
		x = self.last_linear_layer(x) #(n_samples, out_dim)

		return x 


class DinoLoss(nn.Module):
	"""The dino loss function.
	We inherit from the 'nn.Module' because we want to create a buffer for the 
	logit centers of the teacher.

	Parameters
	----------
	out_dim: int
		The dimensionality of the final layer(The number of synthesis classes)

	teacher_temp, student_temp: float
		The temperature parameter for the teacher and the student

	center_momentum: float 
		Hyperparameter for the exponential moving average of the student parameters
		that determines the center logits. The higher, the more running average matters.
	"""

	def __init__(self, out_dim, teacher_temp = 0.04, student_temp = 0.1, center_momentum = 0.9):
		super().__init__()
		self.student_temp = student_temp 
		self.teacher_temp = teacher_temp
		self.center_momentum = center_momentum 
		self.register_buffer('center', torch.zeros(1, out_dim))


	def forward(self, student_output, teacher_output):
		"""Computing the Loss. 

		Parameters 
		----------
		student_output, teache_output: tuple
			Tuple of tensors of shape '(n_samples, out_dim)' representing
			logits. The length is equal to the number of crops.
			Note that student processes all crops and that the first two crops are 
			the global crops.
		"""

		student_logits = [s / self.student_temp for s in student_output]
		teacher_logits = [(t - self.center) / self.teacher_temp for t in teacher_output]

		student_probs = [F.log_softmax(s, dim = -1) for s in student_logits] #list: 10, (n_samples, out_dim)
		teacher_probs = [F.softmax(t, dim = -1).detach() for t in teacher_logits] #list: 2, (n_samples, out_dim)---we detach because we don't train the teacher

		total_loss = 0
		n_loss_terms = 0 

		for t_ix, t in enumerate(teacher_probs):
			for s_ix , s in enumerate(student_probs):
				if t_ix == s_ix :
					continue 
						
				loss = torch.sum(-t * s, dim = -1) #(n_samples )
				total_loss += loss.mean()  #scaler  
				n_loss_terms += 1  

		total_loss /= n_loss_terms
		self.update_center(teacher_output)

		return total_loss 



	@torch.no_grad()
	def update_center(self, teacher_output):
		"""Updates the center used for teacher output.

		Compute the exponential moving average.

		Parameters
		----------
		teacher_output: list 
			list of tensors of shape '(n_samples, out_dim)' where each tensor 
			represents a different crop.
		"""
		batch_center = torch.cat(teacher_output).mean(dim = 0, keepdim = True) #(1, out_dim)
		self.center = self.center * self.center_momentum + batch_center * (1-self.center_momentum)












