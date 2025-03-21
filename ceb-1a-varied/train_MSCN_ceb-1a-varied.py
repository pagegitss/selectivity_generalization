import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy

import numpy as np
import torch

# from mscn.model import SetConv
import random
import json
import time

from mscn.ceb_utils.ceb_utilits import *
from mscn.ceb_utils.mscn_model import *
from torch.nn.utils import clip_grad_norm_

OPS = ['lt', 'eq', 'in', 'like']

random.seed(42)

is_cuda = torch.cuda.is_available()
save_directory = './saved_models/mscn/'


def extract_sublist(original_list, indices):
	return [original_list[i] for i in indices]


def normalize_labels(label_list, min_val, max_val):
	new_label_list = []
	for qid, l in enumerate(label_list):
		new_l = (np.log(l) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))
		if new_l < 0.:
			new_l = 0.
		if new_l > 1.:
			new_l = 1.
		new_label_list.append(new_l)
	return new_label_list


def unnormalize_labels(label_list, log_min_val, log_max_val):
	new_label_list = []
	for qid, l in enumerate(label_list):
		new_l = l * (log_max_val - log_min_val) + log_min_val
		new_l = np.exp(new_l)
		new_label_list.append(new_l)
	return new_label_list

epoch = 100
feature_dim = 256

bs = 128
lr = 1e-3

from mscn.data_ceb import *

new_directory_list = ["./ceb-1a-varied-queries"]


(table_list, table_dim_list, table_like_dim_list, table_sizes, table_key_groups,
	table_join_keys, table_text_cols, table_normal_cols, col_type, col2minmax) = get_table_info()

template2queries, template2cards, test_template2queries, test_template2cards, colid2featlen, all_table_alias, all_joins, all_cols, max_num_tables, max_num_joins, \
col2indicator, col2candidatevals, col2samplerange, eq_col_combos, temp_q_rep,  ts_to_joins, ts_to_keygroups = read_query_file_for_mscn_w_bitmaps(col2minmax, num_q=30000, test_size=1000,
   shifting_type='granularity', directory_list=new_directory_list,saved_ditectory="./mscn/")

trues = get_true_cardinalities(temp_q_rep)
temp_joins = get_joins(temp_q_rep)
temp_subq_ts_list = list(trues.keys())


max_pfeat_length = 0
for colid in colid2featlen:
	if colid2featlen[colid] > max_pfeat_length:
		max_pfeat_length = colid2featlen[colid]

max_pfeat_length = max_pfeat_length + len(OPS) + len(PLAIN_FILTER_COLS)

mscn_model = SetConv(len(all_table_alias)+ NUM_BITMAP_SAMPLE, max_pfeat_length, len(all_joins), feature_dim)

training_qs = []
training_qreps = []
training_tables = []
training_bitmaps = []
training_joins = []
training_cards = []

in_test_qs = []
in_test_qreps = []
in_test_tables = []
in_test_bitmaps = []
in_test_joins = []
in_test_cards = []

test_qs = []
test_qreps = []
test_tables = []
test_bitmaps = []
test_joins = []
test_cards = []

num_qs = 0
test_num_qs = 0
num_seen_templates = 0
min_val = 0
max_val = 0

### load seen join templates
for template in template2queries:
	num_seen_templates += 1
	training_queries = template2queries[template]
	all_training_cards = template2cards[template]

	if len(all_training_cards) > 50:
		training_cards.extend(all_training_cards[:-50])
		in_test_cards.extend(all_training_cards[-50:])
		num_qs += len(all_training_cards[:-50])

		# for training
		for (q, q_context, q_reps, left_q_reps, right_q_reps, q_bitmaps, left_bitmap, right_bitmap, join_bitmap,
	     pg_est, rs_est, key_groups_per_q, table_tuple, q_joins, _) in training_queries[:-50]:
			q_tables = list(q_reps.keys())

			training_qs.append(q)
			training_qreps.append(q_reps)
			training_bitmaps.append(q_bitmaps)
			training_tables.append(q_tables)
			training_joins.append(q_joins)

		# for test
		for (q, q_context, q_reps, left_q_reps, right_q_reps, q_bitmaps, left_bitmap, right_bitmap, join_bitmap,
	     pg_est, rs_est, key_groups_per_q, table_tuple, q_joins, _) in training_queries[-50:]:
			q_tables = list(q_reps.keys())

			in_test_qs.append(q)
			in_test_qreps.append(q_reps)
			in_test_bitmaps.append(q_bitmaps)
			in_test_tables.append(q_tables)
			in_test_joins.append(q_joins)
	else:
		training_cards.extend(all_training_cards)
		num_qs += len(all_training_cards)

		# for training
		for (q, q_context, q_reps, left_q_reps, right_q_reps, q_bitmaps, left_bitmap, right_bitmap, join_bitmap,
	     pg_est, rs_est, key_groups_per_q, table_tuple, q_joins, _) in training_queries:
			q_tables = list(q_reps.keys())

			training_qs.append(q)
			training_qreps.append(q_reps)
			training_bitmaps.append(q_bitmaps)
			training_tables.append(q_tables)
			training_joins.append(q_joins)


### load unseen queries
random_templates = list(test_template2queries.keys())[:50]
for template in random_templates:
	test_num_qs += len(test_template2queries[template])

	test_queries = test_template2queries[template]
	temp_test_cards = test_template2cards[template]

	combined = list(zip(test_queries, temp_test_cards))
	random.shuffle(combined)
	test_queries, temp_test_cards = zip(*combined)

	test_queries = list(test_queries)[:50]
	temp_test_cards = list(temp_test_cards)[:50]
	test_cards.extend(temp_test_cards)

	for (q, q_context, q_reps, left_q_reps, right_q_reps, q_bitmaps, left_bitmap, right_bitmap, join_bitmap,
	     pg_est, rs_est, key_groups_per_q, table_tuple, q_joins, _) in test_queries:
		q_tables = list(q_reps.keys())

		test_qs.append(q)
		test_qreps.append(q_reps)
		test_bitmaps.append(q_bitmaps)
		test_tables.append(q_tables)
		test_joins.append(q_joins)


max_val = max(max(training_cards), max(in_test_cards), max(test_cards))
min_val = min(min(training_cards), min(in_test_cards), min(test_cards))
norm_training_cards = normalize_labels(training_cards, min_val, max_val)


mscn_model.double()
if is_cuda:
	mscn_model.cuda()
optimizer = torch.optim.Adam(mscn_model.parameters(), lr=lr)

num_qs = len(training_qreps)

print("total number of all queries: {}".format(num_qs + len(in_test_cards) + len(test_cards)))
print("total number of queries: {}".format(num_qs))
print("total number of seen join templates: {}".format(num_seen_templates))
print("total number of join templates: {}".format(len(test_template2queries) + len(template2queries)))

max_card = max(max(training_cards), max(in_test_cards), max(test_cards))

num_qs = len(training_qreps)
num_batches = math.ceil(num_qs / bs)

for epoch_id in range(epoch):
	mscn_model.train()

	qids = list(range(len(training_qreps)))
	random.shuffle(qids)

	accu_loss_total = 0.
	cdf_valid_loss = 0.

	for batch_id in range(num_batches):
		batch_ids = qids[bs*batch_id: bs*batch_id+bs]

		batch_training_qreps = [training_qreps[qid] for qid in batch_ids]
		batch_training_tables = [training_tables[qid] for qid in batch_ids]
		batch_training_bitmaps = [training_bitmaps[qid] for qid in batch_ids]
		batch_training_joins = [training_joins[qid] for qid in batch_ids]
		batch_norm_training_cards = [norm_training_cards[qid] for qid in batch_ids]

		train_loader = mscn_model.load_training_queries(batch_training_qreps, batch_training_tables, batch_training_bitmaps,
		                                                batch_training_joins,
		                                                batch_norm_training_cards, len(PLAIN_FILTER_COLS),
		                                                max_pfeat_length,
		                                                len(all_table_alias),
		                                                len(all_joins), all_table_alias, all_joins, bs)

		for batch in train_loader:
			est_cards = mscn_model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5])
			batch_cards = batch[-1]
			batch_cards = batch_cards.to(torch.float64)
			optimizer.zero_grad()

			se_loss = torch.square(torch.squeeze(est_cards) - batch_cards)

			total_loss = torch.sqrt(torch.mean(se_loss))
			accu_loss_total += total_loss.item()

			total_loss.backward()
			clip_grad_norm_(mscn_model.parameters(), 10.)
			optimizer.step()

	print("epoch: {}; loss: {}".format(epoch_id, accu_loss_total / num_batches))

	if epoch_id >= 20:
		mscn_model.eval()


		all_est_cards = []
		all_true_cards = []

		num_in_test_qs = len(in_test_qreps)
		num_in_test_batches = math.ceil(num_in_test_qs / bs)
		qids = list(range(len(in_test_qreps)))
		for batch_id in range(num_in_test_batches):
			batch_ids = qids[bs * batch_id: bs * batch_id + bs]

			batch_training_qreps = [in_test_qreps[qid] for qid in batch_ids]
			batch_training_tables = [in_test_tables[qid] for qid in batch_ids]
			batch_training_bitmaps = [in_test_bitmaps[qid] for qid in batch_ids]
			batch_training_joins = [in_test_joins[qid] for qid in batch_ids]
			batch_norm_training_cards = [in_test_cards[qid] for qid in batch_ids]

			in_test_loader = mscn_model.load_training_queries(batch_training_qreps, batch_training_tables, batch_training_bitmaps,
			                                                  batch_training_joins,
			                                                  batch_norm_training_cards, len(PLAIN_FILTER_COLS), max_pfeat_length,
			                                                  len(all_table_alias),
			                                                  len(all_joins), all_table_alias, all_joins, bs)

			for batch in in_test_loader:
				est_cards = mscn_model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5])
				if est_cards is not None:
					batch_cards = batch[-1]
					est_cards = torch.squeeze(est_cards)
					batch_cards = torch.squeeze(batch_cards)

					# batch_cards = batch_cards.to(torch.float64)
					est_cards = est_cards.cpu().detach()
					est_cards = unnormalize_labels(est_cards, np.log(min_val), np.log(max_val))
					all_est_cards.extend(est_cards)
					all_true_cards.extend(batch_cards.cpu().detach())

		q_errors = get_join_qerror(all_est_cards, all_true_cards, max_card, "seen", None, epoch_id)

		### ood

		all_est_cards = []
		all_true_cards = []
		num_test_qs = len(test_qreps)
		num_test_batches = math.ceil(num_test_qs / bs)
		qids = list(range(len(test_qreps)))
		for batch_id in range(num_test_batches):
			batch_ids = qids[bs * batch_id: bs * batch_id + bs]

			batch_training_qreps = [test_qreps[qid] for qid in batch_ids]
			batch_training_tables = [test_tables[qid] for qid in batch_ids]
			batch_training_bitmaps = [test_bitmaps[qid] for qid in batch_ids]
			batch_training_joins = [test_joins[qid] for qid in batch_ids]
			batch_norm_training_cards = [test_cards[qid] for qid in batch_ids]

			test_loader = mscn_model.load_training_queries(batch_training_qreps, batch_training_tables,
			                                                  batch_training_bitmaps,
			                                                  batch_training_joins,
			                                                  batch_norm_training_cards, len(PLAIN_FILTER_COLS),
			                                                  max_pfeat_length,
			                                                  len(all_table_alias),
			                                                  len(all_joins), all_table_alias, all_joins, bs)

			for batch in test_loader:
				est_cards = mscn_model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5])
				if est_cards is not None:
					batch_cards = batch[-1]
					est_cards = torch.squeeze(est_cards)
					batch_cards = torch.squeeze(batch_cards)

					# batch_cards = batch_cards.to(torch.float64)
					est_cards = est_cards.cpu().detach()
					est_cards = unnormalize_labels(est_cards, np.log(min_val), np.log(max_val))
					all_est_cards.extend(est_cards)
					all_true_cards.extend(batch_cards.cpu().detach())
		q_errors = get_join_qerror(all_est_cards, all_true_cards, max_card, "unseen", None, epoch_id)