import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA = torch.cuda.is_available()
PAD_IDX = 1

class PositionalEncoder(nn.Module):
	def __init__(self, embed_size, maxlen=1000):
		super().__init__()

		self.pencoder = Tensor(maxlen, embed_size)

		pos = torch.arange(0, maxlen, 1).unsqueeze(1)
		k = torch.exp(np.log(10000) * -torch.arange(0, embed_size, 2.) / embed_size)

		self.pencoder[:, 0::2] = torch.sin(pos * k)
		self.pencoder[:, 1::2] = torch.cos(pos * k)

	def forward(self, n):
		return self.pencoder[:n]


class EncoderLayer(nn.Module):
	def __init__(self, dim, embed_size, num_heads, batch_size, dropout):
		super().__init__()

		self.self_attention = MultiheadAttention(embed_size, num_heads, batch_size, dropout)
		self.ffn = FeedForwardNetwork(dim, embed_size, dropout)

	def forward(self, x, mask):
		z = self.self_attention(x, x, x, mask)
		z = self.ffn(z)

		return z


class DecoderLayer(nn.Module):
	def __init__(self, dim, embed_size, num_heads, batch_size, dropout):
		super().__init__()

		self.self_attention = MultiheadAttention(embed_size, num_heads, batch_size, dropout)
		self.encoder_attention = self.self_attention
		self.ffn = FeedForwardNetwork(dim, embed_size, dropout)

	def forward(self, enc_out, dec_in, mask1, mask2):
		z = self.self_attention(dec_in, dec_in, dec_in, mask1)
		z = self.encoder_attention(z, enc_out, enc_out, mask2)
		z = self.ffn(z)

		return z


def scaled_dot_product_attention(Q, K, V, mask, d_k):
	attn = torch.matmul(Q, K.transpose(2, 3)) / d_k**0.5

	attn = attn.masked_fill(mask, -10000)
	attn = F.softmax(attn, 3)
	attn = torch.matmul(attn, V)

	return attn


class MultiheadAttention(nn.Module):
	def __init__(self, embed_size, num_heads, batch_size, dropout):
		super().__init__()

		# self.proj = nn.Linear(embed_size, embed_size)
		self.Wq = nn.Linear(embed_size, embed_size) # query
		self.Wk = nn.Linear(embed_size, embed_size) # key for attention distribution
		self.Wv = nn.Linear(embed_size, embed_size) # value for context representation
		self.Wo = nn.Linear(embed_size, embed_size)

		self.batch_size = batch_size
		self.num_heads = num_heads
		self.dropout = dropout
		self.embed_size = embed_size
		self.d_k = embed_size // num_heads

		self.layer_norm = nn.LayerNorm(embed_size)

	def forward(self, q, k, v, mask):
		x = q
		q = self.Wq(q).view(self.batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
		k = self.Wk(k).view(self.batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
		v = self.Wv(v).view(self.batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

		z = scaled_dot_product_attention(q, k, v, mask, self.d_k)
		z = z.transpose(1, 2).contiguous().view(self.batch_size, -1, self.embed_size)

		z = F.dropout(self.Wo(z), p=self.dropout)
		z = self.layer_norm(x + z)	# Residual

		return z


class FeedForwardNetwork(nn.Module):
	def __init__(self, dim, embed_size, dropout):
		super().__init__()

		self.fc1 = nn.Linear(embed_size, dim)
		self.fc2 = nn.Linear(dim, embed_size)

		self.layer_norm = nn.LayerNorm(embed_size)
		self.dropout = dropout

	def forward(self, x):
		z = F.dropout(self.fc2(F.relu(self.fc1(x))), p=self.dropout)
		z = self.layer_norm(x + z)

		return z

def Tensor(*args):
	x = torch.Tensor(*args)
	return x.cuda() if CUDA else x

def mask_triu(x):
	y = Tensor(np.triu(np.ones([x.size(2), x.size(2)]), 1)).byte()
	return torch.gt(x + y, 0)

def LongTensor(*args):
	x = torch.LongTensor(*args)
	return x.cuda() if CUDA else x

def scalar(x):
	return x.view(-1).data.tolist()[0]