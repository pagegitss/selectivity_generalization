import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

OPS = ['lt', 'eq', 'in', 'like']
NUM_BITMAP_SAMPLE = 1000

ALIAS = {'title': 't',
         'kind_type': 'kt',
         'movie_info': 'mi',
         'movie_info_idx': 'mii',
         'info_type': 'it',
         'cast_info': 'ci',
         'role_type': 'rt',
         'keyword': 'k',
         'name': 'n',
         'movie_companies': 'mc',
         'movie_keyword': 'mk',
         'company_name': 'cn',
         'company_type': 'ct',
         'aka_name': 'an',
         'person_info': 'pi'}

reverse_alias = {value: key for key, value in ALIAS.items()}

JOIN_MAP_IMDB = {}
JOIN_MAP_IMDB["title.id"] = "movie_id"  # pk
JOIN_MAP_IMDB["movie_info.movie_id"] = "movie_id"
JOIN_MAP_IMDB["cast_info.movie_id"] = "movie_id"
JOIN_MAP_IMDB["movie_keyword.movie_id"] = "movie_id"
JOIN_MAP_IMDB["movie_companies.movie_id"] = "movie_id"
# JOIN_MAP_IMDB["movie_link.movie_id"] = "movie_id"
JOIN_MAP_IMDB["movie_info_idx.movie_id"] = "movie_id"
# JOIN_MAP_IMDB["movie_link.linked_movie_id"] = "movie_id"
JOIN_MAP_IMDB["aka_title.movie_id"] = "movie_id"
JOIN_MAP_IMDB["complete_cast.movie_id"] = "movie_id"

JOIN_MAP_IMDB["movie_keyword.keyword_id"] = "keyword"
JOIN_MAP_IMDB["keyword.id"] = "keyword"  # pk

JOIN_MAP_IMDB["name.id"] = "person_id"  # pk
JOIN_MAP_IMDB["person_info.person_id"] = "person_id"
JOIN_MAP_IMDB["cast_info.person_id"] = "person_id"
JOIN_MAP_IMDB["aka_name.person_id"] = "person_id"

JOIN_MAP_IMDB["title.kind_id"] = "kind_id"
JOIN_MAP_IMDB["aka_title.kind_id"] = "kind_id"
JOIN_MAP_IMDB["kind_type.id"] = "kind_id"  # pk

JOIN_MAP_IMDB["cast_info.role_id"] = "role_id"
JOIN_MAP_IMDB["role_type.id"] = "role_id"  # pk

JOIN_MAP_IMDB["cast_info.person_role_id"] = "char_id"
JOIN_MAP_IMDB["char_name.id"] = "char_id"  # pk

JOIN_MAP_IMDB["movie_info.info_type_id"] = "info_id"
JOIN_MAP_IMDB["movie_info_idx.info_type_id"] = "info_id"
JOIN_MAP_IMDB["person_info.info_type_id"] = "info_id"
JOIN_MAP_IMDB["info_type.id"] = "info_id"  # pk

JOIN_MAP_IMDB["movie_companies.company_type_id"] = "company_type"
JOIN_MAP_IMDB["company_type.id"] = "company_type"  # pk

JOIN_MAP_IMDB["movie_companies.company_id"] = "company_id"
JOIN_MAP_IMDB["company_name.id"] = "company_id"  # pk

# JOIN_MAP_IMDB["movie_link.link_type_id"] = "link_id"
# JOIN_MAP_IMDB["link_type.id"] = "link_id" # pk

JOIN_MAP_IMDB["complete_cast.status_id"] = "subject"
JOIN_MAP_IMDB["complete_cast.subject_id"] = "subject"
JOIN_MAP_IMDB["comp_cast_type.id"] = "subject"  # pk

primary_key_dic = {
	'movie_id': "title.id",
	'info_id': "info_type.id",
	'kind_id': "kind_type.id",
	'person_id': "name.id",
	'role_id': "role_type.id"
}

KEY_GROUPS = list(primary_key_dic.keys())

def drop_trailing_number(s):
	return re.sub(r'\d+$', '', s)


def multi_batch_query2reps(batch_reps, OPS, num_cols, max_feat_len):
	# for multiple join queries
	max_num_preds = 0
	qreps_res = []
	for reps in batch_reps:
		feat_set = []
		for per_rep in reps:
			col_id = per_rep[0]
			op = per_rep[1]
			pfeat = per_rep[2]

			col_vec = np.zeros(num_cols, dtype=np.float64)
			col_vec[col_id] = 1.

			op_id = OPS.index(op)
			op_vec = np.zeros(len(OPS), dtype=np.float64)
			op_vec[op_id] = 1.

			if isinstance(pfeat, float):
				pfeat = np.array([pfeat])

			pred_vec = np.concatenate((col_vec, op_vec, pfeat))
			pad_size = max_feat_len - len(pred_vec)
			padding = np.zeros(pad_size, dtype=np.float64)
			pred_vec = np.concatenate((pred_vec, padding))

			feat_set.append(pred_vec)

		qreps_res.append(feat_set)

		if len(feat_set) > max_num_preds:
			max_num_preds = len(feat_set)

	batch_masks = np.zeros((len(qreps_res), max_num_preds), dtype=np.float64)

	if max_num_preds > 0:
		for i, q_reps in enumerate(batch_reps):
			num_preds = len(q_reps)
			if num_preds > 0:
				pad_size = max_num_preds - num_preds
				padding = np.zeros((pad_size, max_feat_len), dtype=np.float64)
				qreps_res[i] = np.vstack((qreps_res[i], padding))
				batch_masks[i, :num_preds] = 1
			else:
				qreps_res[i] = np.zeros((max_num_preds, max_feat_len), dtype=np.float64)
				batch_masks[i, 0] = 1
	else:
		qreps_res = np.zeros((len(qreps_res), 1, max_feat_len), dtype=np.float64)
		batch_masks = np.ones((len(qreps_res), 1), dtype=np.float64)

	qreps_res = torch.from_numpy(np.array(qreps_res))
	batch_masks = torch.from_numpy(batch_masks)
	batch_masks = torch.unsqueeze(batch_masks, dim=-1)
	return qreps_res, batch_masks



class SetConv(nn.Module):
	def __init__(self, sample_feats, predicate_feats, join_feats, hid_units):
		super(SetConv, self).__init__()
		self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
		self.sample_mlp2 = nn.Linear(hid_units, hid_units)
		self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
		self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
		self.join_mlp1 = nn.Linear(join_feats, hid_units)
		self.join_mlp2 = nn.Linear(hid_units, hid_units)
		self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
		self.out_mlp2 = nn.Linear(hid_units, 1)
		self.onehot_mask_truep = 0.8

	def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
		# samples has shape [batch_size x num_joins+1 x sample_feats]
		# predicates has shape [batch_size x num_predicates x predicate_feats]
		# joins has shape [batch_size x num_joins x join_feats]

		hid_sample = F.relu(self.sample_mlp1(samples))
		hid_sample = F.relu(self.sample_mlp2(hid_sample))
		hid_sample = hid_sample * sample_mask  # Mask
		hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
		sample_norm = sample_mask.sum(1, keepdim=False)
		hid_sample = hid_sample / sample_norm  # Calculate average only over non-masked parts

		hid_predicate = F.relu(self.predicate_mlp1(predicates))
		hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
		hid_predicate = hid_predicate * predicate_mask
		hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
		predicate_norm = predicate_mask.sum(1, keepdim=False)
		hid_predicate = hid_predicate / predicate_norm

		hid_join = F.relu(self.join_mlp1(joins))
		hid_join = F.relu(self.join_mlp2(hid_join))
		hid_join = hid_join * join_mask
		hid_join = torch.sum(hid_join, dim=1, keepdim=False)
		join_norm = join_mask.sum(1, keepdim=False)
		hid_join = hid_join / join_norm

		hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
		hid = F.relu(self.out_mlp1(hid))
		out = torch.sigmoid(self.out_mlp2(hid))
		return out

	def load_training_queries(self, qreps_list, tables_list, bitmaps_list, joins_list, training_cards, num_cols,
	                          max_feat_len, max_num_tables, max_num_joins,
	                          all_distict_table, all_distict_joins, bs=64, is_cuda=False, is_shuffle=True):
		training_cards = np.array(training_cards)
		training_cards = torch.from_numpy(training_cards)

		if is_cuda:
			training_cards = training_cards.cuda()

		batch_reps = []
		batch_tables = []
		batch_joins = []

		for table2reps in qreps_list:
			a_qrep = []
			table_list = list(table2reps.keys())
			for table_name in table_list:
				a_qrep_table = table2reps[table_name]
				a_qrep.extend(a_qrep_table)
			batch_reps.append(a_qrep)

		batch_reps, batch_rep_masks = multi_batch_query2reps(batch_reps, OPS, num_cols, max_feat_len)
		if is_cuda:
			batch_reps = batch_reps.cuda()
			batch_rep_masks = batch_rep_masks.cuda()

		#### prepraing table encodings
		for tables, bitmaps in zip(tables_list, bitmaps_list):
			table_set = []
			for table_name in tables:
				t_id = all_distict_table.index(table_name)
				bitmaps_on_table = bitmaps[table_name]

				table_vec = np.zeros(len(all_distict_table), dtype=np.float64)
				table_vec[t_id] = 1.

				bitmap_vec = np.zeros(NUM_BITMAP_SAMPLE, dtype=np.float64)
				for bid in bitmaps_on_table:
					bitmap_vec[bid - 1] = 1.

				table_vec = np.concatenate((table_vec, bitmap_vec))
				table_set.append(table_vec)

			batch_tables.append(table_set)

		#### prepraing join encodings
		for joins in joins_list:
			join_set = []
			for join in joins:
				j_id = all_distict_joins.index(join)
				join_vec = np.zeros(len(all_distict_joins), dtype=np.float64)
				join_vec[j_id] = 1.
				join_set.append(join_vec)

			batch_joins.append(join_set)

		### padding
		table_batch_masks = np.zeros((len(qreps_list), max_num_tables), dtype=np.float64)
		join_batch_masks = np.zeros((len(qreps_list), max_num_joins), dtype=np.float64)

		for i, q_tables in enumerate(batch_tables):
			num_tables = len(q_tables)
			if num_tables > 0:
				pad_size = max_num_tables - num_tables
				padding = np.zeros((pad_size, len(all_distict_table) + NUM_BITMAP_SAMPLE), dtype=np.float64)
				batch_tables[i] = np.vstack((batch_tables[i], padding))
				table_batch_masks[i, :num_tables] = 1
			else:
				batch_tables[i] = np.zeros((max_num_tables, len(all_distict_table) + NUM_BITMAP_SAMPLE),
				                           dtype=np.float64)
				table_batch_masks[i, 0] = 1

		for i, q_joins in enumerate(batch_joins):
			num_joins = len(q_joins)
			if num_joins > 0:
				pad_size = max_num_joins - num_joins
				padding = np.zeros((pad_size, len(all_distict_joins)), dtype=np.float64)
				batch_joins[i] = np.vstack((batch_joins[i], padding))
				join_batch_masks[i, :num_joins] = 1
			else:
				batch_joins[i] = np.zeros((max_num_joins, len(all_distict_joins)), dtype=np.float64)
				join_batch_masks[i, 0] = 1

		batch_tables = np.array(batch_tables)
		batch_joins = np.array(batch_joins)

		batch_tables = torch.from_numpy(batch_tables)
		batch_joins = torch.from_numpy(batch_joins)
		table_batch_masks = torch.from_numpy(table_batch_masks)
		join_batch_masks = torch.from_numpy(join_batch_masks)

		table_batch_masks = table_batch_masks.unsqueeze(-1)
		join_batch_masks = join_batch_masks.unsqueeze(-1)

		if is_cuda:
			batch_tables = batch_tables.cuda()
			batch_joins = batch_joins.cuda()
			table_batch_masks = table_batch_masks.cuda()
			join_batch_masks = join_batch_masks.cuda()

		dataloader_list = [batch_tables, batch_reps, batch_joins, table_batch_masks, batch_rep_masks, join_batch_masks]

		dataloader_list.append(training_cards)

		dataloader_list = TensorDataset(*dataloader_list)
		dataloader = DataLoader(dataloader_list, batch_size=bs, shuffle=is_shuffle)

		return dataloader

	def load_training_queries_w_cdfs(self, qreps_list, left_qreps_list, right_qreps_list, tables_list,
	                                 bitmaps_list, left_bitmaps_list, right_bitmaps_list, joins_list, training_cards,
	                                 signs, num_cols, max_feat_len, max_num_tables, max_num_joins, all_distict_table,
	                                 all_distict_joins,
	                                 bs=64, is_cuda=False, is_shuffle=True, is_unified=True):
		training_cards = np.array(training_cards)
		training_cards = torch.from_numpy(training_cards)

		signs = np.array(signs)
		signs = torch.from_numpy(signs)

		if is_cuda:
			training_cards = training_cards.cuda()
			signs = signs.cuda()

		batch_reps = []
		batch_tables = []
		batch_joins = []

		batch_left_reps = []
		batch_right_reps = []

		batch_left_tables = []
		batch_right_tables = []

		for table2reps in qreps_list:
			a_qrep = []
			table_list = list(table2reps.keys())
			for table_name in table_list:
				a_qrep_table = table2reps[table_name]
				a_qrep.extend(a_qrep_table)
			batch_reps.append(a_qrep)

		for table2reps in left_qreps_list:
			a_qrep = []
			table_list = list(table2reps.keys())
			for table_name in table_list:
				a_qrep_table = table2reps[table_name]
				a_qrep.extend(a_qrep_table)
			batch_left_reps.append(a_qrep)

		for table2reps in right_qreps_list:
			a_qrep = []
			table_list = list(table2reps.keys())
			for table_name in table_list:
				a_qrep_table = table2reps[table_name]
				a_qrep.extend(a_qrep_table)
			batch_right_reps.append(a_qrep)

		batch_reps, batch_rep_masks = multi_batch_query2reps(batch_reps, OPS, num_cols, max_feat_len)
		if is_cuda:
			batch_reps = batch_reps.cuda()
			batch_rep_masks = batch_rep_masks.cuda()

		batch_left_reps, batch_left_rep_masks = multi_batch_query2reps(batch_left_reps, OPS, num_cols, max_feat_len)
		if is_cuda:
			batch_left_reps = batch_left_reps.cuda()
			batch_left_rep_masks = batch_left_rep_masks.cuda()

		batch_right_reps, batch_right_rep_masks = multi_batch_query2reps(batch_right_reps, OPS, num_cols, max_feat_len)
		if is_cuda:
			batch_right_reps = batch_right_reps.cuda()
			batch_right_rep_masks = batch_right_rep_masks.cuda()

		#### prepraing table encodings
		for tables, bitmaps in zip(tables_list, bitmaps_list):
			table_set = []
			for table_name in tables:
				t_id = all_distict_table.index(table_name)
				bitmaps_on_table = bitmaps[table_name]

				table_vec = np.zeros(len(all_distict_table), dtype=np.float64)
				table_vec[t_id] = 1.

				bitmap_vec = np.zeros(NUM_BITMAP_SAMPLE, dtype=np.float64)
				for bid in bitmaps_on_table:
					bitmap_vec[bid - 1] = 1.

				table_vec = np.concatenate((table_vec, bitmap_vec))
				table_set.append(table_vec)

			batch_tables.append(table_set)

		for tables, bitmaps in zip(tables_list, left_bitmaps_list):
			table_set = []
			for table_name in tables:
				t_id = all_distict_table.index(table_name)
				bitmaps_on_table = bitmaps[table_name]

				table_vec = np.zeros(len(all_distict_table), dtype=np.float64)
				table_vec[t_id] = 1.

				bitmap_vec = np.zeros(NUM_BITMAP_SAMPLE, dtype=np.float64)
				for bid in bitmaps_on_table:
					bitmap_vec[bid - 1] = 1.

				table_vec = np.concatenate((table_vec, bitmap_vec))
				table_set.append(table_vec)

			batch_left_tables.append(table_set)

		for tables, bitmaps in zip(tables_list, right_bitmaps_list):
			table_set = []
			for table_name in tables:
				t_id = all_distict_table.index(table_name)
				bitmaps_on_table = bitmaps[table_name]

				table_vec = np.zeros(len(all_distict_table), dtype=np.float64)
				table_vec[t_id] = 1.

				bitmap_vec = np.zeros(NUM_BITMAP_SAMPLE, dtype=np.float64)
				for bid in bitmaps_on_table:
					bitmap_vec[bid - 1] = 1.

				table_vec = np.concatenate((table_vec, bitmap_vec))
				table_set.append(table_vec)

			batch_right_tables.append(table_set)

		#### prepraing join encodings
		for joins in joins_list:
			join_set = []
			for join in joins:
				j_id = all_distict_joins.index(join)
				join_vec = np.zeros(len(all_distict_joins), dtype=np.float64)
				join_vec[j_id] = 1.
				join_set.append(join_vec)

			batch_joins.append(join_set)

		### padding
		table_batch_masks = np.zeros((len(qreps_list), max_num_tables), dtype=np.float64)
		join_batch_masks = np.zeros((len(qreps_list), max_num_joins), dtype=np.float64)

		for i, q_tables in enumerate(batch_tables):
			num_tables = len(q_tables)
			if num_tables > 0:
				pad_size = max_num_tables - num_tables
				padding = np.zeros((pad_size, len(all_distict_table) + NUM_BITMAP_SAMPLE), dtype=np.float64)
				batch_tables[i] = np.vstack((batch_tables[i], padding))
				batch_left_tables[i] = np.vstack((batch_left_tables[i], padding))
				batch_right_tables[i] = np.vstack((batch_right_tables[i], padding))
				table_batch_masks[i, :num_tables] = 1
			else:
				batch_tables[i] = np.zeros((max_num_tables, len(all_distict_table) + NUM_BITMAP_SAMPLE),
				                           dtype=np.float64)
				batch_left_tables[i] = np.zeros((max_num_tables, len(all_distict_table) + NUM_BITMAP_SAMPLE),
				                                dtype=np.float64)
				batch_right_tables[i] = np.zeros((max_num_tables, len(all_distict_table) + NUM_BITMAP_SAMPLE),
				                                 dtype=np.float64)
				table_batch_masks[i, 0] = 1

		for i, q_joins in enumerate(batch_joins):
			num_joins = len(q_joins)
			if num_joins > 0:
				pad_size = max_num_joins - num_joins
				padding = np.zeros((pad_size, len(all_distict_joins)), dtype=np.float64)
				batch_joins[i] = np.vstack((batch_joins[i], padding))
				join_batch_masks[i, :num_joins] = 1
			else:
				batch_joins[i] = np.zeros((max_num_joins, len(all_distict_joins)), dtype=np.float64)
				join_batch_masks[i, 0] = 1

		batch_tables = np.array(batch_tables)
		batch_joins = np.array(batch_joins)

		batch_tables = torch.from_numpy(batch_tables)
		batch_joins = torch.from_numpy(batch_joins)
		table_batch_masks = torch.from_numpy(table_batch_masks)
		join_batch_masks = torch.from_numpy(join_batch_masks)

		batch_left_tables = np.array(batch_left_tables)
		batch_right_tables = np.array(batch_right_tables)
		batch_left_tables = torch.from_numpy(batch_left_tables)
		batch_right_tables = torch.from_numpy(batch_right_tables)

		table_batch_masks = table_batch_masks.unsqueeze(-1)
		join_batch_masks = join_batch_masks.unsqueeze(-1)

		if is_cuda:
			batch_tables = batch_tables.cuda()
			batch_joins = batch_joins.cuda()
			table_batch_masks = table_batch_masks.cuda()
			join_batch_masks = join_batch_masks.cuda()
			batch_left_tables = batch_left_tables.cuda()
			batch_right_tables = batch_right_tables.cuda()

		if is_unified:
			dataloader_list = [batch_tables, batch_reps, batch_joins, table_batch_masks, batch_rep_masks,
			                   join_batch_masks,
			                   batch_left_tables, batch_right_tables, batch_left_reps, batch_right_reps,
			                   batch_left_rep_masks, batch_right_rep_masks, signs]

			dataloader_list.append(training_cards)

			dataloader_list = TensorDataset(*dataloader_list)
			dataloader = DataLoader(dataloader_list, batch_size=bs, shuffle=is_shuffle)

			return dataloader
		else:
			return (batch_tables, batch_reps, batch_joins, table_batch_masks, batch_rep_masks, join_batch_masks,
			        batch_left_tables, batch_right_tables, batch_left_reps, batch_right_reps,
			        batch_left_rep_masks, batch_right_rep_masks, signs)

	def load_training_queries_no_bitmaps(self, qreps_list, tables_list, joins_list, training_cards, num_cols,
	                                     max_feat_len, max_num_tables, max_num_joins,
	                                     all_distict_table, all_distict_joins, bs=64, is_cuda=True, is_shuffle=True):
		dataloader_list = []
		training_cards = np.array(training_cards)
		training_cards = torch.from_numpy(training_cards)

		if is_cuda:
			training_cards = training_cards.cuda()

		batch_reps = []
		batch_tables = []
		batch_joins = []

		for table2reps in qreps_list:
			a_qrep = []
			table_list = list(table2reps.keys())
			for table_name in table_list:
				a_qrep_table = table2reps[table_name]
				a_qrep.extend(a_qrep_table)
			batch_reps.append(a_qrep)

		batch_reps, batch_rep_masks = multi_batch_query2reps(batch_reps, OPS, num_cols, max_feat_len)
		if is_cuda:
			batch_reps = batch_reps.cuda()
			batch_rep_masks = batch_rep_masks.cuda()

		#### prepraing table encodings
		for tables in tables_list:
			table_set = []
			for table_name in tables:
				t_id = all_distict_table.index(table_name)
				table_vec = np.zeros(len(all_distict_table), dtype=np.float64)
				table_vec[t_id] = 1.
				table_set.append(table_vec)

			batch_tables.append(table_set)

		#### prepraing join encodings
		for joins in joins_list:
			join_set = []
			for join in joins:
				j_id = all_distict_joins.index(join)
				join_vec = np.zeros(len(all_distict_joins), dtype=np.float64)
				join_vec[j_id] = 1.
				join_set.append(join_vec)

			batch_joins.append(join_set)

		### padding
		table_batch_masks = np.zeros((len(qreps_list), max_num_tables), dtype=np.float64)
		join_batch_masks = np.zeros((len(qreps_list), max_num_joins), dtype=np.float64)

		for i, q_tables in enumerate(batch_tables):
			num_tables = len(q_tables)
			if num_tables > 0:
				pad_size = max_num_tables - num_tables
				padding = np.zeros((pad_size, len(all_distict_table)), dtype=np.float64)
				batch_tables[i] = np.vstack((batch_tables[i], padding))
				table_batch_masks[i, :num_tables] = 1
			else:
				batch_tables[i] = np.zeros((max_num_tables, len(all_distict_table)), dtype=np.float64)
				table_batch_masks[i, 0] = 1

		for i, q_joins in enumerate(batch_joins):
			num_joins = len(q_joins)
			if num_joins > 0:
				pad_size = max_num_joins - num_joins
				padding = np.zeros((pad_size, len(all_distict_joins)), dtype=np.float64)
				batch_joins[i] = np.vstack((batch_joins[i], padding))
				join_batch_masks[i, :num_joins] = 1
			else:
				batch_joins[i] = np.zeros((max_num_joins, len(all_distict_joins)), dtype=np.float64)
				join_batch_masks[i, 0] = 1

		batch_tables = np.array(batch_tables)
		batch_joins = np.array(batch_joins)

		batch_tables = torch.from_numpy(batch_tables)
		batch_joins = torch.from_numpy(batch_joins)
		table_batch_masks = torch.from_numpy(table_batch_masks)
		join_batch_masks = torch.from_numpy(join_batch_masks)

		table_batch_masks = table_batch_masks.unsqueeze(-1)
		join_batch_masks = join_batch_masks.unsqueeze(-1)

		if is_cuda:
			batch_tables = batch_tables.cuda()
			batch_joins = batch_joins.cuda()
			table_batch_masks = table_batch_masks.cuda()
			join_batch_masks = join_batch_masks.cuda()

		dataloader_list = [batch_tables, batch_reps, batch_joins, table_batch_masks, batch_rep_masks, join_batch_masks]

		dataloader_list.append(training_cards)

		dataloader_list = TensorDataset(*dataloader_list)
		dataloader = DataLoader(dataloader_list, batch_size=bs, shuffle=is_shuffle)

		return dataloader


	def get_key_groups(self, joins):
		key_groups = {}
		for join in joins:
			parts = join.strip().split(' = ')

			alias_table1 = parts[0].split('.')[0]
			alias_table2 = parts[1].split('.')[0]

			col1 = parts[0].split('.')[1]
			col2 = parts[1].split('.')[1]

			full_key_name1 = reverse_alias[drop_trailing_number(alias_table1)] + '.' + col1
			full_key_name2 = reverse_alias[drop_trailing_number(alias_table2)] + '.' + col2

			key_group = JOIN_MAP_IMDB[full_key_name1]
			if alias_table1 == 'it1' or alias_table2 == 'it1':
				key_group = f"{key_group}1"

			if alias_table1 == 'it2' or alias_table2 == 'it2':
				key_group = f"{key_group}2"

			if key_group not in key_groups:
				key_groups[key_group] = []

			if alias_table1 not in key_groups[key_group]:
				key_groups[key_group].append(alias_table1)

			if alias_table2 not in key_groups[key_group]:
				key_groups[key_group].append(alias_table2)

		return key_groups

	def load_training_queries_w_joinbitmaps(self, qreps_list, tables_list, bitmaps_list, join_bitmaps_list, joins_list, training_cards, num_cols,
	                          max_feat_len, max_num_tables, max_num_joins, max_num_key_groups,
	                          all_distict_table, cur, bs=64, is_cuda=False, is_shuffle=True):
		training_cards = np.array(training_cards)
		training_cards = torch.from_numpy(training_cards)

		if is_cuda:
			training_cards = training_cards.cuda()

		batch_reps = []
		batch_tables = []
		batch_joins = []

		all_pk_res = []
		query = "SELECT sid, id FROM jb_movie_id_title;"
		cur.execute(query)
		explain_results = cur.fetchall()

		for id_res in explain_results:
			all_pk_res.append([id_res[0], id_res[1]])

		for table2reps in qreps_list:
			a_qrep = []
			table_list = list(table2reps.keys())
			for table_name in table_list:
				a_qrep_table = table2reps[table_name]
				a_qrep.extend(a_qrep_table)
			batch_reps.append(a_qrep)

		batch_reps, batch_rep_masks = multi_batch_query2reps(batch_reps, OPS, num_cols, max_feat_len)
		if is_cuda:
			batch_reps = batch_reps.cuda()
			batch_rep_masks = batch_rep_masks.cuda()

		#### prepraing table encodings
		for tables, bitmaps in zip(tables_list, bitmaps_list):
			table_set = []
			for table_name in tables:
				t_id = all_distict_table.index(table_name)
				bitmaps_on_table = bitmaps[table_name]

				table_vec = np.zeros(len(all_distict_table), dtype=np.float64)
				table_vec[t_id] = 1.

				bitmap_vec = np.zeros(NUM_BITMAP_SAMPLE, dtype=np.float64)
				for bid in bitmaps_on_table:
					bitmap_vec[bid - 1] = 1.

				table_vec = np.concatenate((table_vec, bitmap_vec))
				table_set.append(table_vec)

			batch_tables.append(table_set)

		#### prepraing join encodings
		for joins, join_bitmaps, tables in zip(joins_list, join_bitmaps_list, tables_list):
			join_set = []
			if len(joins) > 0:
				key_groups = self.get_key_groups(joins)
				for key_group in key_groups:
					t_list = key_groups[key_group]
					original_key_group = drop_trailing_number(key_group)

					num_join_vec = np.zeros(max_num_joins, dtype=np.float64)
					num_join_vec[len(t_list) - 1] = 1.

					table_vec = np.zeros(len(all_distict_table), dtype=np.float64)
					for alias_t in t_list:
						# table_name = reverse_alias[drop_trailing_number(alias_t)]
						t_id = all_distict_table.index(alias_t)
						table_vec[t_id] = 1.

					### get join key group id
					key_group_id = KEY_GROUPS.index(original_key_group)
					key_group_vec = np.zeros(len(KEY_GROUPS), dtype=np.float64)
					key_group_vec[key_group_id] = 1.

					### get join bitmaps

					pk = primary_key_dic[original_key_group]
					pk_table_origin = ALIAS[pk.split('.')[0]]
					pk_table = pk_table_origin
					for t_alias in key_groups[key_group]:
						if drop_trailing_number(t_alias) == pk_table_origin:
							pk_table = t_alias

					if pk not in join_bitmaps[original_key_group]:
						### the case for t
						pk_res = all_pk_res
					else:
						pk_res = join_bitmaps[original_key_group][pk_table]

					join_bitmap = []
					sid2id = {}
					for item in pk_res:
						join_bitmap.append(item[1])
						sid2id[item[1]] = item[0]
					join_bitmap = set(join_bitmap)

					for alias_t in t_list:
						### consider intersection
						if alias_t != pk_table:
							table_join_bitmap = set(join_bitmaps[original_key_group][alias_t])
							# join_bitmap = [idx for idx in join_bitmap if idx in table_join_bitmap]
							join_bitmap.intersection_update(table_join_bitmap)

					jb_idxs = [sid2id[idx] for idx in join_bitmap]
					join_bitmap_vec = np.zeros(NUM_BITMAP_SAMPLE, dtype=np.float64)

					for jb_idx in jb_idxs:
						join_bitmap_vec[jb_idx - 1] = 1.

					join_vec = np.concatenate((num_join_vec, table_vec, key_group_vec, join_bitmap_vec))
					join_set.append(join_vec)
		#
		# ################
		# 	for join in joins:
		# 		j_id = all_distict_joins.index(join)
		# 		join_vec = np.zeros(len(all_distict_joins), dtype=np.float64)
		# 		join_vec[j_id] = 1.
		# 		join_set.append(join_vec)

			batch_joins.append(join_set)

		### padding
		table_batch_masks = np.zeros((len(qreps_list), max_num_tables), dtype=np.float64)
		join_batch_masks = np.zeros((len(qreps_list), max_num_key_groups), dtype=np.float64)

		for i, q_tables in enumerate(batch_tables):
			num_tables = len(q_tables)
			if num_tables > 0:
				pad_size = max_num_tables - num_tables
				padding = np.zeros((pad_size, len(all_distict_table) + NUM_BITMAP_SAMPLE), dtype=np.float64)
				batch_tables[i] = np.vstack((batch_tables[i], padding))
				table_batch_masks[i, :num_tables] = 1
			else:
				batch_tables[i] = np.zeros((max_num_tables, len(all_distict_table) + NUM_BITMAP_SAMPLE),
				                           dtype=np.float64)
				table_batch_masks[i, 0] = 1

		join_vec_len = max_num_joins + len(all_distict_table) + len(KEY_GROUPS) + NUM_BITMAP_SAMPLE
		for i, q_joins in enumerate(batch_joins):
			num_joins = len(q_joins)
			if num_joins > 0:
				pad_size = max_num_key_groups - num_joins
				padding = np.zeros((pad_size, join_vec_len), dtype=np.float64)
				batch_joins[i] = np.vstack((batch_joins[i], padding))
				join_batch_masks[i, :num_joins] = 1
			else:
				batch_joins[i] = np.zeros((max_num_key_groups, join_vec_len), dtype=np.float64)
				join_batch_masks[i, 0] = 1

		batch_tables = np.array(batch_tables)
		batch_joins = np.array(batch_joins)

		batch_tables = torch.from_numpy(batch_tables)
		batch_joins = torch.from_numpy(batch_joins)
		table_batch_masks = torch.from_numpy(table_batch_masks)
		join_batch_masks = torch.from_numpy(join_batch_masks)

		table_batch_masks = table_batch_masks.unsqueeze(-1)
		join_batch_masks = join_batch_masks.unsqueeze(-1)

		if is_cuda:
			batch_tables = batch_tables.cuda()
			batch_joins = batch_joins.cuda()
			table_batch_masks = table_batch_masks.cuda()
			join_batch_masks = join_batch_masks.cuda()

		dataloader_list = [batch_tables, batch_reps, batch_joins, table_batch_masks, batch_rep_masks, join_batch_masks]

		dataloader_list.append(training_cards)

		dataloader_list = TensorDataset(*dataloader_list)
		dataloader = DataLoader(dataloader_list, batch_size=bs, shuffle=is_shuffle)

		return dataloader

	def get_onehot_mask(self, vec):
		tmask = ~np.array(vec, dtype="bool")
		ptrue = self.onehot_mask_truep
		pfalse = 1 - self.onehot_mask_truep

		# probabilities are switched
		bools = np.random.choice(a=[False, True], size=(len(tmask),),
		                         p=[ptrue, pfalse])
		tmask *= bools
		tmask = ~tmask
		tmask = torch.from_numpy(tmask).float()
		return tmask