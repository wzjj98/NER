#!/usr/bin/env python3
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
	''' sinusoid编码
		
		:param n_position: int, 位置长度
		:param d_hid: int, 位置编码长度
		:param padding_idx: padding的token_ids
		:return: [seq_len, d_hid]
	'''
	position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
	div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
	embeddings_table = torch.zeros(n_position, d_hid)
	embeddings_table[:, 0::2] = torch.sin(position * div_term)
	embeddings_table[:, 1::2] = torch.cos(position * div_term)
	return embeddings_table

	# 第二种实现
	position_ids = torch.arange(0, n_position).unsqueeze(1)
	position_ids = position_ids.expand(-1, d_hid)
	indices = torch.arange(0, d_hid)
	position_ids = position_ids * torch.pow(10000, -2 * torch.true_divide(torch.floor_divide(indices, 2), d_hid))
	position_ids[:, ::2] = torch.sin(position_ids[:, ::2])
	position_ids[:, 1::2] = torch.cos(position_ids[:, 1::2])
	return position_ids
	
class RoPEPositionEncoding(nn.Module):
	"""旋转式位置编码: https://kexue.fm/archives/8265
	"""
	def __init__(self, embedding_size, rope_rank='adjacent', **kwargs):
		super(RoPEPositionEncoding, self).__init__()
		self.max_seq_len_cache = -1
		self.embedding_size = embedding_size
		# 支持两种方式，一种是奇偶相邻排列，一种是上下排列, 目前只在chatglm中看到updown排列
		assert rope_rank in {'adjacent', 'updown'}, "rank kwarg only support 'adjacent' and 'updown' "
		self.rope_rank = rope_rank
		
	def initialize(self, max_position):
		position_embeddings = get_sinusoid_encoding_table(max_position, self.embedding_size)  # [seq_len, hdsz]
		if self.rope_rank == 'adjacent':
			cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)  # [seq_len, hdsz]
			sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)  # [seq_len, hdsz]
		elif self.rope_rank == 'updown':  # 目前仅chatglm使用
			cos_position = position_embeddings[:, 1::2].repeat(1,2)  # [seq_len, hdsz]
			sin_position = position_embeddings[:, ::2].repeat(1,2)  # [seq_len, hdsz]
		else:
			raise ValueError('Args `rope_rank` only support `adjacent` and `adjacent` mode')
		return cos_position, sin_position
	
	def forward(self, qw, position_ids=None, seq_dim=-2):
		# MultiHeadAttentionLayer中qw是[btz, n_heads, seq_len, head_size]
		# GlobalPointer中*转置*后qw是[btz, n_heads, seq_len, head_size]
		# EfficientGlobalPointer中qw是[btz, seq_len, head_size]
		if self.rope_rank == 'adjacent':
			qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
		elif self.rope_rank == 'updown':  # 目前仅chatglm使用
			qw2 = torch.cat([-qw[..., qw.shape[-1]//2:], qw[..., :qw.shape[-1]//2]], dim=-1)  # cat和stack+reshape是结果不同的
		
		# 超过缓存长度
		seq_len = position_ids.max() + 1 if position_ids is not None else qw.shape[seq_dim]
		if seq_len > self.max_seq_len_cache:
			cos_position, sin_position = self.initialize(seq_len)
			self.cos_position, self.sin_position = cos_position.type_as(qw).to(qw.device), sin_position.type_as(qw).to(qw.device)
			self.max_seq_len_cache = seq_len
			
		# 传入position_ids来获取cos和sin, 主要是在use_states时候能直接取到对应位置的编码
		if position_ids is not None:
			# position_ids: [btz, seq_len]
			cos = F.embedding(position_ids, self.cos_position)  # [btz, seq_len, hdsz]
			sin = F.embedding(position_ids, self.sin_position)
		else:
			cos = self.cos_position[:seq_len]  # [seq_len, hdsz]
			sin = self.sin_position[:seq_len]
			
		if cos.dim() < qw.dim():
			cos = cos.unsqueeze(seq_dim-1)
			sin = sin.unsqueeze(seq_dim-1)
		return qw * cos + qw2 * sin
	
class PromptTable(nn.Module):
	def __init__(self,hidden_size,head_size):
		super().__init__()
		self.head_size = head_size
		self.dense = nn.Linear(hidden_size,head_size*2)
		self.position_embedding = RoPEPositionEncoding(head_size)
	
	def forward(self,inputs,mask=None):
		sequence_output = self.dense(inputs)  # [..., head_size*2]
		qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., head_size]
		qw = self.position_embedding(qw)
		kw = self.position_embedding(kw)
		logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5  # [btz, seq_len, seq_len], 是否是实体的打分
		# 排除padding
		if mask is not None:
			attention_mask1 = 1 - mask.unsqueeze(2)  # [btz, 1, seq_len, 1]
			attention_mask2 = 1 - mask.unsqueeze(1)  # [btz, 1, 1, seq_len]
			logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
			logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))
		return logits
	
