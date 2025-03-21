import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy

from mscn.ceb_utils.ceb_utilits import *
from mscn.ceb_utils.mscn_model import *
from torch.nn.utils import clip_grad_norm_
from mscn.data_ceb import *
from mscn.secon_utils import *
import torch.multiprocessing as mp
import time

cur = connect_pg()
OPS = ['lt', 'eq', 'in', 'like']

is_cuda = torch.cuda.is_available()
print(is_cuda)


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


def unnormalize_torch(vals, min_val, max_val, is_log_scale=False):
	vals = (vals * (max_val - min_val)) + min_val
	if not is_log_scale:
		return torch.exp(vals) / np.exp(max_val)
	else:
		return vals - max_val


def get_consistency_loss(batch, mscn_model, min_val, max_val, is_cuda=True):
	if is_cuda:
		est_cards = mscn_model(batch[0].squeeze(0).cuda(), batch[1].squeeze(0).cuda(), batch[2].squeeze(0).cuda(),
		                       batch[3].squeeze(0).cuda(), batch[4].squeeze(0).cuda(), batch[5].squeeze(0).cuda())
		left_cdf_ests = mscn_model(batch[6].squeeze(0).cuda(), batch[8].squeeze(0).cuda(), batch[2].squeeze(0).cuda(),
		                           batch[3].squeeze(0).cuda(), batch[10].squeeze(0).cuda(), batch[5].squeeze(0).cuda())
		right_cdf_ests = mscn_model(batch[7].squeeze(0).cuda(), batch[9].squeeze(0).cuda(), batch[2].squeeze(0).cuda(),
		                            batch[3].squeeze(0).cuda(), batch[11].squeeze(0).cuda(), batch[5].squeeze(0).cuda())
	else:
		est_cards = mscn_model(batch[0].squeeze(0), batch[1].squeeze(0), batch[2].squeeze(0),
		                       batch[3].squeeze(0), batch[4].squeeze(0), batch[5].squeeze(0))
		left_cdf_ests = mscn_model(batch[6].squeeze(0), batch[8].squeeze(0), batch[2].squeeze(0),
		                           batch[3].squeeze(0), batch[10].squeeze(0), batch[5].squeeze(0))
		right_cdf_ests = mscn_model(batch[7].squeeze(0), batch[9].squeeze(0), batch[2].squeeze(0),
		                            batch[3].squeeze(0), batch[11].squeeze(0), batch[5].squeeze(0))

	unnormalized_est_cards = unnormalize_torch(torch.squeeze(est_cards), min_val, max_val)

	unnormalized_left_cdf_ests = unnormalize_torch(torch.squeeze(left_cdf_ests), min_val, max_val)
	unnormalized_right_cdf_ests = unnormalize_torch(torch.squeeze(right_cdf_ests), min_val, max_val)
	aug_cdf_preds = unnormalized_right_cdf_ests - unnormalized_left_cdf_ests

	log_aug_cdf_preds = torch.where(aug_cdf_preds <= 0, 1., torch.log(aug_cdf_preds))

	sle_loss = torch.square(log_aug_cdf_preds - torch.log(unnormalized_est_cards))
	se_loss = torch.square(aug_cdf_preds - unnormalized_est_cards)

	ultimate_loss = torch.where(torch.squeeze(aug_cdf_preds) > 0, sle_loss, se_loss)
	consistency_loss = torch.sqrt(torch.mean(ultimate_loss))

	return consistency_loss


def get_cdf_loss(batch, mscn_model, min_val, max_val):
	targets = batch[-1]
	signs = batch[-2]

	left_cdf_ests = mscn_model(batch[6], batch[8], batch[2], batch[3], batch[10], batch[5])
	right_cdf_ests = mscn_model(batch[7], batch[9], batch[2], batch[3], batch[11], batch[5])

	unnormalized_left_cdf_ests = unnormalize_torch(torch.squeeze(left_cdf_ests), min_val, max_val)
	unnormalized_right_cdf_ests = unnormalize_torch(torch.squeeze(right_cdf_ests), min_val, max_val)

	unnormalized_cdf_ests = torch.cat([torch.unsqueeze(unnormalized_left_cdf_ests, dim=-1),
	                                   torch.unsqueeze(unnormalized_right_cdf_ests, dim=-1)], dim=-1)

	train_cdf_preds = torch.sum(unnormalized_cdf_ests * signs, dim=-1)
	train_cdf_preds = torch.squeeze(train_cdf_preds)

	unnormalized_targets = unnormalize_torch(torch.squeeze(targets.float()), min_val, max_val)

	log_aug_cdf_preds = torch.where(train_cdf_preds <= 0, 1., torch.log(train_cdf_preds))

	sle_loss = torch.square(log_aug_cdf_preds - torch.log(unnormalized_targets))
	se_loss = torch.square(train_cdf_preds - unnormalized_targets)

	ultimate_loss = torch.where(torch.squeeze(train_cdf_preds) > 0, sle_loss, se_loss)
	consistency_loss = torch.sqrt(torch.mean(ultimate_loss))

	# cdf_loss = torch.sqrt(torch.mean(torch.square(train_cdf_preds - unnormalized_targets)))
	return consistency_loss


from torch.utils.data import Dataset, DataLoader


def load_training_queries_w_cdfs(qreps_list, left_qreps_list, right_qreps_list, tables_list,
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
			batch_tables[i] = np.zeros((max_num_tables, len(all_distict_table) + NUM_BITMAP_SAMPLE), dtype=np.float64)
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
		dataloader_list = [batch_tables, batch_reps, batch_joins, table_batch_masks, batch_rep_masks, join_batch_masks,
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

class SampleDataset(Dataset):
	def __init__(self, total_samples, other_params):

		self.total_samples = total_samples
		# Store other necessary parameters for sampling
		self.other_params = other_params

	def __len__(self):
		return self.total_samples

	def __getitem__(self, idx):
		# Call your sample_a_batch function here
		sampled_queries, sampled_left_queries, sampled_right_queries = sample_a_batch(
			self.other_params['sample_size'],  # or other appropriate sample size
			self.other_params['temp_q_rep'],
			self.other_params['col2indicator'],
			self.other_params['col2candidatevals'],
			self.other_params['col2samplerange'],
			self.other_params['eq_col_combos'],
			self.other_params['temp_joins'],
			self.other_params['col2minmax'],
			self.other_params['colid2featlen'],
			self.other_params['all_cols'],
			self.other_params['temp_subq_ts_list'],
			self.other_params['ts_to_joins'],
			self.other_params['ts_to_keygroups'], cur
		)

		secon_qreps = []
		secon_bitmaps = []
		secon_tables = []
		secon_joins = []
		secon_signs = []

		secon_left_qreps = []
		secon_left_bitmaps = []
		secon_left_tables = []
		secon_left_joins = []

		secon_right_qreps = []
		secon_right_bitmaps = []
		secon_right_tables = []
		secon_right_joins = []

		dummy_card_list = []

		for (q_reps, q_bitmaps, key_groups_per_q, table_tuple, q_joins) in sampled_queries:
			q_tables = list(q_reps.keys())

			secon_qreps.append(q_reps)
			secon_bitmaps.append(q_bitmaps)
			secon_tables.append(q_tables)
			secon_joins.append(q_joins)
			secon_signs.append([-1, 1])
			dummy_card_list.append(10)

		for (q_reps, q_bitmaps, key_groups_per_q, table_tuple, q_joins) in sampled_left_queries:
			q_tables = list(q_reps.keys())

			secon_left_qreps.append(q_reps)
			secon_left_bitmaps.append(q_bitmaps)
			secon_left_tables.append(q_tables)
			secon_left_joins.append(q_joins)

		for (q_reps, q_bitmaps, key_groups_per_q, table_tuple, q_joins) in sampled_right_queries:
			q_tables = list(q_reps.keys())

			secon_right_qreps.append(q_reps)
			secon_right_bitmaps.append(q_bitmaps)
			secon_right_tables.append(q_tables)
			secon_right_joins.append(q_joins)

		sampled_batch = load_training_queries_w_cdfs(secon_qreps, secon_left_qreps,
		                                                              secon_right_qreps, secon_tables,
		                                                              secon_bitmaps, secon_left_bitmaps,
		                                                              secon_right_bitmaps, secon_joins,
		                                                              dummy_card_list, secon_signs,
		                                                              len(PLAIN_FILTER_COLS), self.other_params['max_pfeat_length'],
		                                                              len(self.other_params['all_table_alias']), len(self.other_params['all_joins']),
		                                                              self.other_params['all_table_alias'], self.other_params['all_joins'],
		                                                              len(secon_qreps), is_cuda=False, is_unified=False)

		# Return the batch data
		return sampled_batch


def main():

	epoch = 100
	feature_dim = 256

	bs = 128
	lr = 1e-3

	new_directory_list = ["./ceb-1a-varied-queries"]
	shifting_type = 'granularity'

	weight_list = [1e-3, 1e-4]

	(table_list, table_dim_list, table_like_dim_list, table_sizes, table_key_groups,
	 table_join_keys, table_text_cols, table_normal_cols, col_type, col2minmax) = get_table_info()

	template2queries, template2cards, test_template2queries, test_template2cards, colid2featlen, all_table_alias, all_joins, all_cols, max_num_tables, max_num_joins, \
	col2indicator, col2candidatevals, col2samplerange, eq_col_combos, temp_q_rep, ts_to_joins, ts_to_keygroups = read_query_file_for_mscn_w_bitmaps(
		col2minmax, num_q=30000, test_size=1000, shifting_type=shifting_type,
		directory_list=new_directory_list, saved_ditectory="./mscn/")

	print(col2samplerange)

	for weight in weight_list:
		print(weight)
		random.seed(42)
		trues = get_true_cardinalities(temp_q_rep)
		temp_joins = get_joins(temp_q_rep)
		temp_subq_ts_list = list(trues.keys())

		max_pfeat_length = 0
		for colid in colid2featlen:
			if colid2featlen[colid] > max_pfeat_length:
				max_pfeat_length = colid2featlen[colid]

		max_pfeat_length = max_pfeat_length + len(OPS) + len(PLAIN_FILTER_COLS)

		mscn_model = SetConv(len(all_table_alias) + NUM_BITMAP_SAMPLE, max_pfeat_length, len(all_joins),
		                     feature_dim)

		training_qs = []
		training_qreps = []
		training_left_qreps = []
		training_right_qreps = []
		training_tables = []
		training_bitmaps = []
		training_left_bitmaps = []
		training_right_bitmaps = []
		training_joins = []
		training_cards = []
		training_signs = []

		in_test_qs = []
		in_test_qreps = []
		in_test_left_qreps = []
		in_test_right_qreps = []
		in_test_tables = []
		in_test_bitmaps = []
		in_test_left_bitmaps = []
		in_test_right_bitmaps = []
		in_test_joins = []
		in_test_cards = []

		test_qs = []
		test_qreps = []
		test_left_qreps = []
		test_right_qreps = []
		test_tables = []
		test_bitmaps = []
		test_left_bitmaps = []
		test_right_bitmaps = []
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
				for (q, q_context, q_reps, left_q_reps, right_q_reps, q_bitmaps, left_bitmap, right_bitmap,
						table2joinbitmaps, pg_est, rs_est, key_groups_per_q,
						table_tuple, q_joins, cdf_signs) in training_queries[:-50]:
					q_tables = list(q_reps.keys())

					training_qs.append(q)
					training_qreps.append(q_reps)
					training_left_qreps.append(left_q_reps)
					training_right_qreps.append(right_q_reps)
					training_bitmaps.append(q_bitmaps)
					training_left_bitmaps.append(left_bitmap)
					training_right_bitmaps.append(right_bitmap)
					training_tables.append(q_tables)
					training_signs.append(cdf_signs)
					training_joins.append(q_joins)

				# for test
				for (q, q_context, q_reps, left_q_reps, right_q_reps, q_bitmaps, left_bitmap, right_bitmap,
						key_groups_per_q, pg_est, rs_est, key_groups_per_q,
						table_tuple, q_joins, cdf_signs) in training_queries[-50:]:
					q_tables = list(q_reps.keys())

					in_test_qs.append(q)
					in_test_qreps.append(q_reps)
					in_test_left_qreps.append(left_q_reps)
					in_test_right_qreps.append(right_q_reps)
					in_test_bitmaps.append(q_bitmaps)
					in_test_left_bitmaps.append(left_bitmap)
					in_test_right_bitmaps.append(right_bitmap)
					in_test_tables.append(q_tables)
					in_test_joins.append(q_joins)
			else:
				training_cards.extend(all_training_cards)
				num_qs += len(all_training_cards)

				# for training
				for (q, q_context, q_reps, left_q_reps, right_q_reps, q_bitmaps, left_bitmap, right_bitmap,
						key_groups_per_q, pg_est, rs_est, key_groups_per_q,
						table_tuple, q_joins, cdf_signs) in training_queries:
					q_tables = list(q_reps.keys())

					training_qs.append(q)
					training_qreps.append(q_reps)
					training_left_qreps.append(left_q_reps)
					training_right_qreps.append(right_q_reps)
					training_bitmaps.append(q_bitmaps)
					training_left_bitmaps.append(left_bitmap)
					training_right_bitmaps.append(right_bitmap)
					training_tables.append(q_tables)
					training_signs.append(cdf_signs)
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

			for (q, q_context, q_reps, left_q_reps, right_q_reps, q_bitmaps, left_bitmap, right_bitmap,
					key_groups_per_q, pg_est, rs_est, key_groups_per_q,
					table_tuple, q_joins, cdf_signs) in test_queries:
				q_tables = list(q_reps.keys())

				test_qs.append(q)
				test_qreps.append(q_reps)
				test_left_qreps.append(left_q_reps)
				test_right_qreps.append(right_q_reps)
				test_bitmaps.append(q_bitmaps)
				test_left_bitmaps.append(left_bitmap)
				test_right_bitmaps.append(right_bitmap)
				test_tables.append(q_tables)
				test_joins.append(q_joins)

		max_val = max(max(training_cards), max(in_test_cards), max(test_cards))
		min_val = min(min(training_cards), min(in_test_cards), min(test_cards))

		log_max = np.log(max_val)
		log_min = np.log(min_val)

		norm_training_cards = normalize_labels(training_cards, min_val, max_val)

		mscn_model.double()
		if is_cuda:
			mscn_model.cuda()
		optimizer = torch.optim.Adam(mscn_model.parameters(), lr=lr)

		num_qs = len(training_qreps)
		num_batches = math.ceil(num_qs / bs)


		print("total number of queries: {}".format(num_qs))
		print("total number of seen join templates: {}".format(num_seen_templates))
		print("total number of join templates: {}".format(len(test_template2queries) + len(template2queries)))

		# Initialize your sample dataset
		sample_size = 2

		sample_dataset = SampleDataset(total_samples=1024, other_params={
			'sample_size': sample_size,
			'temp_q_rep': temp_q_rep,
			'col2indicator': col2indicator,
			'col2candidatevals': col2candidatevals,
			'col2samplerange': col2samplerange,
			'eq_col_combos': eq_col_combos,
			'temp_joins': temp_joins,
			'col2minmax': col2minmax,
			'colid2featlen': colid2featlen,
			'all_cols': all_cols,
			'temp_subq_ts_list': temp_subq_ts_list,
			'ts_to_joins': ts_to_joins,
			'ts_to_keygroups': ts_to_keygroups,
			'max_pfeat_length': max_pfeat_length,
			'all_table_alias': all_table_alias,
			'all_joins': all_joins
		})

		# Create the DataLoader
		sample_loader = DataLoader(
			sample_dataset,
			batch_size=1,  # Adjust batch size as needed
			shuffle=True,
			num_workers=4,  # Number of worker processes for parallel data loading
			pin_memory=True  # Set to True if using CUDA
		)

		sample_loader_iter = iter(sample_loader)

		for epoch_id in range(epoch):
			mscn_model.train()

			qids = list(range(len(training_qreps)))
			random.shuffle(qids)

			accu_loss_total = 0.

			for batch_id in range(num_batches):
				batch_ids = qids[bs * batch_id: bs * batch_id + bs]

				batch_training_qreps = [training_qreps[qid] for qid in batch_ids]
				batch_training_left_qreps = [training_left_qreps[qid] for qid in batch_ids]
				batch_training_right_qreps = [training_right_qreps[qid] for qid in batch_ids]
				batch_training_tables = [training_tables[qid] for qid in batch_ids]
				batch_training_bitmaps = [training_bitmaps[qid] for qid in batch_ids]
				batch_training_left_bitmaps = [training_left_bitmaps[qid] for qid in batch_ids]
				batch_training_right_bitmaps = [training_right_bitmaps[qid] for qid in batch_ids]
				batch_training_joins = [training_joins[qid] for qid in batch_ids]
				batch_norm_training_cards = [norm_training_cards[qid] for qid in batch_ids]
				batch_training_signs = [training_signs[qid] for qid in batch_ids]

				train_loader = mscn_model.load_training_queries_w_cdfs(batch_training_qreps,
				                                                       batch_training_left_qreps,
				                                                       batch_training_right_qreps,
				                                                       batch_training_tables,
				                                                       batch_training_bitmaps,
				                                                       batch_training_left_bitmaps,
				                                                       batch_training_right_bitmaps,
				                                                       batch_training_joins,
				                                                       batch_norm_training_cards,
				                                                       batch_training_signs,
				                                                       len(PLAIN_FILTER_COLS), max_pfeat_length,
				                                                       len(all_table_alias), len(all_joins),
				                                                       all_table_alias, all_joins, bs,
				                                                       is_cuda=is_cuda)

				for batch in train_loader:
					optimizer.zero_grad()

					try:
						sampled_batch = next(sample_loader_iter)
					# Process the batch
					except StopIteration:
						# Restart the iterator when dataset ends
						sample_loader_iter = iter(sample_loader)
						sampled_batch = next(sample_loader_iter)

					consistency_loss = get_consistency_loss(sampled_batch, mscn_model, log_min, log_max, is_cuda=is_cuda)

					###

					est_cards = mscn_model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5])
					cdf_loss = get_cdf_loss(batch, mscn_model, log_min, log_max)

					batch_cards = batch[-1]
					batch_cards = batch_cards.to(torch.float64)

					se_loss = torch.square(torch.squeeze(est_cards) - batch_cards)
					se_loss = torch.sqrt(torch.mean(se_loss))

					total_loss = se_loss + weight * cdf_loss + weight * consistency_loss

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

					in_test_loader = mscn_model.load_training_queries(batch_training_qreps, batch_training_tables,
					                                                  batch_training_bitmaps,
					                                                  batch_training_joins,
					                                                  batch_norm_training_cards,
					                                                  len(PLAIN_FILTER_COLS),
					                                                  max_pfeat_length,
					                                                  len(all_table_alias),
					                                                  len(all_joins), all_table_alias, all_joins,
					                                                  bs,
					                                                  is_cuda=is_cuda)

					for batch in in_test_loader:
						est_cards = mscn_model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5])
						if est_cards is not None:
							batch_cards = batch[-1]
							est_cards = torch.squeeze(est_cards)
							batch_cards = torch.squeeze(batch_cards)

							# batch_cards = batch_cards.to(torch.float64)
							est_cards = est_cards.cpu().detach()
							est_cards = unnormalize_labels(est_cards, log_min, log_max)
							all_est_cards.extend(est_cards)
							all_true_cards.extend(batch_cards.cpu().detach())

				q_errors = get_join_qerror(all_est_cards, all_true_cards, max_val, "seen", None, epoch_id)

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
					                                               batch_norm_training_cards,
					                                               len(PLAIN_FILTER_COLS),
					                                               max_pfeat_length,
					                                               len(all_table_alias),
					                                               len(all_joins), all_table_alias, all_joins, bs,
					                                               is_cuda=is_cuda)

					for batch in test_loader:
						est_cards = mscn_model(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5])
						if est_cards is not None:
							batch_cards = batch[-1]
							est_cards = torch.squeeze(est_cards)
							batch_cards = torch.squeeze(batch_cards)

							# batch_cards = batch_cards.to(torch.float64)
							est_cards = est_cards.cpu().detach()
							est_cards = unnormalize_labels(est_cards, log_min, log_max)
							all_est_cards.extend(est_cards)
							all_true_cards.extend(batch_cards.cpu().detach())

				q_errors = get_join_qerror(all_est_cards, all_true_cards, max_val, "unseen", None, epoch_id)



if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	# Rest of your code
	main()
