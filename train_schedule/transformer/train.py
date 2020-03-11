import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from model import Model

import modules as mm

from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from train_util import *

import argparse

CUDA = torch.cuda.is_available()

class hp:
	def __init__(self):
		self.vocab_size = 1500
		self.ff_dim = 2048 
		self.batch_size = 16
		self.embed_size = 512
		self.num_layers = 5
		self.num_heads = 8
		self.dropout = 0.1
		self.maxlen = 500


class TrainInstance:
	def __init__(self, args):
		self.hparams = hp()
		self.model = Model(self.hparams)
		self.vocab = Vocab(config.vocab_path, self.hparams.vocab_size)
		self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
							   batch_size=self.hparams.batch_size, single_pass=False)
		self.args = args
		self.start_id = self.vocab.word2id(data.START_DECODING)
		self.end_id = self.vocab.word2id(data.STOP_DECODING)
		self.pad_id = self.vocab.word2id(data.PAD_TOKEN)
		self.unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
		time.sleep(3)

	def save_model(self, iter):
		save_path = config.save_model_path + "/%07d.tar" % iter
		T.save({
			"iter": iter + 1,
			"model_dict": self.model.state_dict(),
			"trainer_dict": self.trainer.state_dict()
		}, save_path)

	def setup_train(self):
		self.trainer = T.optim.Adam(self.model.parameters(), lr=config.lr)
		start_iter = 0
		if self.args.load_model is not None:
			load_model_path = os.path.join(config.save_model_path, self.args.load_model)
			checkpoint = T.load(load_model_path)
			start_iter = checkpoint["iter"]
			self.model.load_state_dict(checkpoint["model_dict"])
			self.trainer.load_state_dict(checkpoint["trainer_dict"])
			print("Loaded model at " + load_model_path)
		if self.args.new_lr is not None:
			self.trainer = T.optim.Adam(self.model.parameters(), lr=self.args.new_lr)
		return start_iter

	def decoder(self, enc_out, mask, batch):
		SOS_IDX = 2
		dec_batch, max_dec_len, dec_lens, target_batch = get_dec_data(batch)
		dec_batch = torch.cuda.LongTensor(dec_batch)
		dec_in = torch.cuda.LongTensor([SOS_IDX] * self.hparams.batch_size).unsqueeze(1)
		PAD_IDX = 1
		loss = 0
		try:
			# for t in range(min(max_dec_len, config.max_dec_steps)):
			for t in range(dec_batch.size(1)):
				dec_out = self.model.decoder(enc_out, dec_in, mask)
				loss += F.nll_loss(dec_out, dec_batch[:, t], reduction = "sum", ignore_index = PAD_IDX)
				dec_in = torch.cat((dec_in, dec_batch[:, t].unsqueeze(1)), 1)
				# if VERBOSE:
				# 	for i, j in enumerate(dec_out.data.topk(1)[1]):
				# 		pred[i].append(scalar(j))
			loss /= dec_batch.data.gt(0).sum().float() # divide by the number of unpadded tokens
			loss.backward()
			print("Loss: {}".format(mm.scalar(loss)))
			return loss
		except:
			return 0

	def train_one_batch(self, batch):
		enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, context = get_enc_data(batch)

		enc_batch = torch.cuda.LongTensor(enc_batch)

		PAD_IDX = 1
		def mask_pad(x):
			return x.data.eq(PAD_IDX).view(self.hparams.batch_size, 1, 1, -1)

		self.trainer.zero_grad()
		mask = mask_pad(enc_batch)

		enc_out = self.model.encoder(enc_batch, mask)
		dec_out = self.decoder(enc_out, mask, batch)

		if dec_out == True:
			# mle_loss = self.train_batch_MLE(enc_out, enc_hidden, enc_padding_mask, context, extra_zeros, enc_batch_extend_vocab, batch)
			# (self.opt.mle_weight * mle_loss + self.opt.rl_weight * rl_loss).backward()
			self.trainer.step()

		# return mle_loss.item(), batch_reward

	def train_iters(self):
		iter = self.setup_train()
		# count = mle_total = r_total = 0
		count = 0
		print("Training")
		while iter <= config.max_iterations:
			batch = self.batcher.next_batch()
			if count == 1:
				count += 1
				continue
			# try:
				# mle_loss, r = self.train_one_batch(batch, iter)
			print("Batch {}".format(count))
			self.train_one_batch(batch)
			# except KeyboardInterrupt:
				# print("-------------------Keyboard Interrupt------------------")
				# exit(0)

			# mle_total += mle_loss
			# r_total += r
			count += 1
			iter += 1

			if iter % 1000 == 0:
				# mle_avg = mle_total / count
				# r_avg = r_total / count
				# print("iter:", iter, "mle_loss:", "%.3f" % mle_avg, "reward:", "%.4f" % r_avg)
				# count = mle_total = r_total = 0
				count = 0
				print("iter: {}".format(iter))

			if iter % 5000 == 0:
				self.save_model(iter)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# parser.add_argument('--train_mle', type=str, default="yes")
	# parser.add_argument('--train_rl', type=str, default="no")
	# parser.add_argument('--mle_weight', type=float, default=1.0)
	parser.add_argument('--load_model', type=str, default=None)
	parser.add_argument('--new_lr', type=float, default=None)
	args = parser.parse_args()
	# opt.rl_weight = 1 - opt.mle_weight
	# print("Training mle: %s, Training rl: %s, mle weight: %.2f, rl weight: %.2f"%(opt.train_mle, opt.train_rl, opt.mle_weight, opt.rl_weight))
	# print("intra_encoder:", config.intra_encoder, "intra_decoder:", config.intra_decoder)

	trainer = TrainInstance(args)
	trainer.train_iters()