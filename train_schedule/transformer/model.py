import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import modules as mm

PAD_IDX = 1
SOS_IDX = 2

CUDA = torch.cuda.is_available()

class TransformerBlock(nn.Module):
	def __init__(self, hp):
		super().__init__()

		self.embeddings = nn.Embedding(hp.vocab_size, hp.embed_size, padding_idx = PAD_IDX)
		self.pos_encoder = mm.PositionalEncoder(hp.embed_size, hp.maxlen)

		if CUDA: self = self.cuda()


class Encoder(TransformerBlock):
	def __init__(self, hp):
		super().__init__(hp)

		self.layers = nn.ModuleList([mm.EncoderLayer(hp.ff_dim, hp.embed_size, hp.num_heads, hp.batch_size, 
			hp.dropout) for _ in range(hp.num_layers)])

	def forward(self, x, mask):
		h = self.embeddings(x)

		h += self.pos_encoder(h.size(1))

		print(x.shape)
		print(h.shape)

		h = F.dropout(h, 0.1)

		for layer in self.layers:
			h = layer(h, mask)

		return h


class Decoder(TransformerBlock):
	def __init__(self, hp):
		super().__init__(hp)

		self.layers = nn.ModuleList([mm.DecoderLayer(hp.ff_dim, hp.embed_size, hp.num_heads, hp.batch_size, 
			hp.dropout) for _ in range(hp.num_layers)])
		self.out = nn.Linear(hp.embed_size, hp.vocab_size)
		self.softmax = nn.LogSoftmax(1)
		self.hp = hp

	def run(self):
		print(self.embeddings)
		print(type(self.embeddings))

	def forward(self, enc_out, dec_in, mask2):
		h = self.embeddings(dec_in)
		h += self.pos_encoder(h.size(1))

		def mask_pad(x):
			return x.data.eq(PAD_IDX).view(self.hp.batch_size, 1, 1, -1)

		mask1 = mm.mask_triu(mask_pad(dec_in))

		for layer in self.layers:
			h = layer(enc_out, h, mask1, mask2)

		h = self.out(h[:, -1])
		y = self.softmax(h)
		
		return y


class Model(nn.Module):
	def __init__(self, hp):
		super().__init__()

		self.hp = hp

		self.encoder = Encoder(self.hp)
		self.decoder = Decoder(self.hp)

		if CUDA: self = self.cuda()

	def train(self):
		pass


if __name__ == '__main__':
	class hp:
		def __init__(self):
			self.vocab_size = 50000
			self.ff_dim = 2048 
			self.batch_size = 16
			self.embed_size = 256
			self.num_layers = 4
			self.num_heads = 8
			self.dropout = 0.5
			self.maxlen = 200

	hparams = hp()
	model = Model(hparams)
	model.decoder.run()
	# print(model)