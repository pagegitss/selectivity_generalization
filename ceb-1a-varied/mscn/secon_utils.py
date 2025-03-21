import copy
import hashlib

from mscn.query_representation.query import *
from mscn.query_representation.sql_parser import *
from mscn.query_representation.generate_bitmap import *
from mscn.ceb_globals import *
import numpy as np
import random
# random.seed(42)
import re

def drop_trailing_number(s):
	return re.sub(r'\d+$', '', s)

def deterministic_hash(string):
	return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

def normalize_val(val, min_v, max_v, col_info=None):
	res = float(val - min_v) / (max_v - min_v)

	if res > 1:
		print('error')
		print(val)
		print(min_v)
		print(max_v)
		if col_info is not None:
			print(col_info)
	if res < 0:
		return 0.
	elif res > 1:
		return 1.
	else:
		return res


def sample_from_join_template(qrep, col2indicator, col2candidatevals, col2samplerange,
                              eq_col_combos, shifting_col, cur):

	'''
	col2indicator: if the col be sampled?
	col2candidatevals: candidates of the col vals
	col2samplerange: the range of number of values sampled? only for IN operator
	eq_col_combos: [(col_names, col_val_combos)]
	'''

	col2type = {}
	col2vals = {}
	t2bitmaps = {}

	for col_names in eq_col_combos:
		col_val_combos = eq_col_combos[col_names]
		sampled_val_combo = random.choice(col_val_combos)

		col1 = col_names.split(',')[0]
		col2 = col_names.split(',')[1]

		alias_t1 = col1.split('.')[0]
		col2type[col1] = 'eq'
		col2vals[col1] = sampled_val_combo[0]
		sub_sql = "SELECT COUNT(*) FROM info_type AS it1 WHERE it1.id = \'{}\'".format(sampled_val_combo[0])
		bitmap = get_bitmaps(sub_sql, cur=cur)
		t2bitmaps[alias_t1] = bitmap

		alias_t2 = col2.split('.')[0]
		col2type[col2] = 'eq'
		col2vals[col2] = sampled_val_combo[1]
		sub_sql = "SELECT COUNT(*) FROM info_type AS it1 WHERE it1.id = \'{}\'".format(sampled_val_combo[1])
		bitmap = get_bitmaps(sub_sql, cur=cur)
		t2bitmaps[alias_t2] = bitmap

	for node in qrep["join_graph"].nodes(data=True):
		t_alias = node[0]
		info = node[1]
		if len(info["predicates"]) == 0:
			t2bitmaps[t_alias] = list(range(1, 1001))
			continue

		### TODO, not being used at this time
		if (info["pred_cols"][0] not in col2indicator) or (col2indicator[info["pred_cols"][0]] == False):
			for predicate, col, op, val in zip(info["predicates"], info["pred_cols"],
			                                   info["pred_types"], info["pred_vals"]):
				alias_t = col.split('.')[0]
				full_t = reverse_alias[drop_trailing_number(alias_t)]
				col2type[col] = op
				col2vals[col] = val

				sub_sql = f"SELECT COUNT(*) FROM {full_t} AS {alias_t} WHERE {col} {op} {val}"
				bitmap = get_bitmaps(sub_sql, cur=cur)
				t2bitmaps[alias_t] = bitmap
			continue
		######

		for predicate, col, op, val in zip(info["predicates"], info["pred_cols"],
		                                   info["pred_types"],info["pred_vals"]):
			alias_t = col.split('.')[0]
			full_t = reverse_alias[drop_trailing_number(alias_t)]

			if op == 'in':
				if col == 'it1.id' or col == 'it2.id':
					continue
				lower = col2samplerange[col][0]
				higher = col2samplerange[col][1]
				num_samples = random.randint(lower, higher)
				sampled_vals = random.sample(col2candidatevals[col], num_samples)

				col2type[col] = 'in'
				col2vals[col] = sampled_vals

				if 'NULL' not in sampled_vals:
					value_list = "', '".join(sampled_vals)
					sub_sql = f"SELECT COUNT(*) FROM {full_t} AS {alias_t} WHERE {col} IN ('{value_list}')"
				else:
					sampled_vals.remove('NULL')
					value_list = "', '".join(sampled_vals)
					sub_sql = f"SELECT COUNT(*) FROM {full_t} AS {alias_t} WHERE {col} IN ('{value_list}') OR {col} IS NULL"

				bitmap = get_bitmaps(sub_sql, cur=cur)
				t2bitmaps[alias_t] = bitmap

			elif op == 'lt':
				min_val = col2candidatevals[col][0]
				max_val = col2candidatevals[col][1]
				l = random.randint(min_val, max_val-1)
				h = random.randint(l+1, max_val)

				col2type[col] = 'lt'
				col2vals[col] = [l, h]

				sub_sql = f"SELECT COUNT(*) FROM {full_t} AS {alias_t} WHERE {col} <= {h} AND {col} > {l}"
				bitmap = get_bitmaps(sub_sql, cur=cur)
				t2bitmaps[alias_t] = bitmap

			else: ### TODO
				pass

	col2vals_left = copy.deepcopy(col2vals)
	col2vals_right = copy.deepcopy(col2vals)

	col2vals_left[shifting_col][0] = col2candidatevals[shifting_col][0]
	col2vals_left[shifting_col][1] = col2vals[shifting_col][0]

	col2vals_right[shifting_col][0] = col2candidatevals[shifting_col][0]
	col2vals_right[shifting_col][1] = col2vals[shifting_col][1]

	### compute the bitmaps for CDF queries
	sub_sql = f"SELECT COUNT(*) FROM title AS t WHERE {shifting_col} <= {col2vals[shifting_col][0]}"
	bitmap = get_bitmaps(sub_sql, cur=cur)
	left_bitmap = bitmap

	sub_sql = f"SELECT COUNT(*) FROM title AS t WHERE {shifting_col} <= {col2vals[shifting_col][1]}"
	bitmap = get_bitmaps(sub_sql, cur=cur)
	right_bitmap = bitmap

	return col2type, col2vals, col2vals_left, col2vals_right, \
	       t2bitmaps, left_bitmap, right_bitmap


def generate_sub_queries(qrep, joins, col2minmax, colid2featlen, all_cols, subq_ts_list, ts_to_joins, ts_to_keygroups,
                         col2type, col2vals, t2bitmaps, left_bitmap, right_bitmap, shifting_t='t'):

	queries = []
	left_queries = []
	right_queries = []

	table2qreps = {}
	left_table2qreps = {}
	right_table2qreps = {}

	table2bitmaps = {}
	left_table2bitmaps = {}
	right_table2bitmaps = {}

	tables, aliases = get_tables(qrep)

	for alias_t in aliases:
		table2qreps[alias_t] = []
		left_table2qreps[alias_t] = []
		right_table2qreps[alias_t] = []
		table2bitmaps[alias_t] = t2bitmaps[alias_t]

		if alias_t != shifting_t:
			left_table2bitmaps[alias_t] = t2bitmaps[alias_t]
			right_table2bitmaps[alias_t] = t2bitmaps[alias_t]
		else:
			left_table2bitmaps[alias_t] = left_bitmap
			right_table2bitmaps[alias_t] = right_bitmap

	queried_tables = set([])

	### parse pred info
	for col in col2type:
		op = col2type[col]
		val = col2vals[col]

		alias_t = col.split('.')[0]
		queried_tables.add(alias_t)
		full_t = reverse_alias[drop_trailing_number(alias_t)]

		standard_col_name = ALIAS[full_t] + '.' + col.split('.')[1]
		col_id = all_cols.index(col)
		min_val = col2minmax[standard_col_name][0]
		max_val = col2minmax[standard_col_name][1]

		if op == 'lt':
			table2qreps[alias_t].append(
					[col_id, op, [normalize_val(val[0], min_val, max_val),
					              normalize_val(val[1], min_val, max_val)]])

			left_table2qreps[alias_t].append(
				[col_id, op, [normalize_val(min_val, min_val, max_val),
				              normalize_val(val[0], min_val, max_val)]])

			right_table2qreps[alias_t].append(
				[col_id, op, [normalize_val(min_val, min_val, max_val),
				              normalize_val(val[1], min_val, max_val)]])

			if col_id not in colid2featlen:
				colid2featlen[col_id] = 2
			elif 2 > colid2featlen[col_id]:
				colid2featlen[col_id] = 2

		elif op == 'eq':
			if isinstance(val, dict):
				new_val = int(val['literal'])
			else:
				new_val = int(val)
			normal_val1 = normalize_val(new_val, min_val, max_val)
			table2qreps[alias_t].append([col_id, op, normal_val1])
			left_table2qreps[alias_t].append([col_id, op, normal_val1])
			right_table2qreps[alias_t].append([col_id, op, normal_val1])
			if col_id not in colid2featlen:
				colid2featlen[col_id] = 1

		elif op == 'in':
			normal_val_list = []

			if standard_col_name == 'it.id':
				## unify ops for it.id since they are the same thing
				new_val = int(val[0])
				normal_val1 = normalize_val(new_val, min_val, max_val)
				table2qreps[alias_t].append([col_id, 'eq', normal_val1])
				left_table2qreps[alias_t].append([col_id, 'eq', normal_val1])
				right_table2qreps[alias_t].append([col_id, 'eq', normal_val1])
				if col_id not in colid2featlen:
					colid2featlen[col_id] = 1
				continue

			if standard_col_name in CATEGORICAL_COLS_VALS:
				qrep_val_list = np.zeros(len(CATEGORICAL_COLS_VALS[standard_col_name]))
				for item_val in val:
					idx_val = CATEGORICAL_COLS_VALS[standard_col_name].index(item_val)
					normal_val1 = normalize_val(idx_val, min_val, max_val)
					normal_val2 = normalize_val(idx_val + 1, min_val, max_val)
					normal_val_list.append([normal_val1, normal_val2])

					qrep_val_list[idx_val] = 1.

				if col_id not in colid2featlen:
					colid2featlen[col_id] = len(CATEGORICAL_COLS_VALS[standard_col_name])
				elif len(CATEGORICAL_COLS_VALS[standard_col_name]) > colid2featlen[col_id]:
					colid2featlen[col_id] = len(CATEGORICAL_COLS_VALS[standard_col_name])

			elif standard_col_name in IN_TEXT_COLS:
				# the case of text cols with feature hash
				bucket_list = []
				qrep_val_list = np.zeros(IN_BUCKETS)
				for item_val in val:
					bucket_idx = deterministic_hash(str(item_val)) % IN_BUCKETS
					qrep_val_list[bucket_idx] = 1.
					if bucket_idx not in bucket_list:
						bucket_list.append(bucket_idx)
						normal_val1 = normalize_val(bucket_idx, min_val, max_val)
						normal_val2 = normalize_val(bucket_idx + 1, min_val, max_val)
						normal_val_list.append([normal_val1, normal_val2])

					if col_id not in colid2featlen:
						colid2featlen[col_id] = IN_BUCKETS
					elif IN_BUCKETS > colid2featlen[col_id]:
						colid2featlen[col_id] = IN_BUCKETS
			else:
				qrep_val_list = []

			table2qreps[alias_t].append([col_id, op, qrep_val_list])
			left_table2qreps[alias_t].append([col_id, op, qrep_val_list])
			right_table2qreps[alias_t].append([col_id, op, qrep_val_list])

	### end parsing pred info

	### parse join info
	key_groups = {}

	new_aliases = list(aliases)

	query_info = [table2qreps, table2bitmaps, key_groups, sorted(new_aliases), joins]
	left_query_info = [left_table2qreps, left_table2bitmaps, key_groups, sorted(new_aliases), joins]
	right_left_query_info = [right_table2qreps, right_table2bitmaps, key_groups, sorted(new_aliases), joins]

	queries.append(query_info)
	left_queries.append(left_query_info)
	right_queries.append(right_left_query_info)

	### generate all involved subqueries

	for ts in subq_ts_list:
		if shifting_t in ts:  # only keep subqueries containing the shifting table

			global_new_k = ts

			sub_table2qreps = {t: table2qreps[t] for t in global_new_k}
			sub_table2bitmaps = {t: table2bitmaps[t] for t in global_new_k}

			left_sub_table2qreps = {t: left_table2qreps[t] for t in global_new_k}
			left_sub_table2bitmaps = {t: left_table2bitmaps[t] for t in global_new_k}

			right_sub_table2qreps = {t: right_table2qreps[t] for t in global_new_k}
			right_sub_table2bitmaps = {t: right_table2bitmaps[t] for t in global_new_k}

			sorted_ts =  sorted(list(global_new_k))
			tuple_ts = tuple(sorted_ts)

			sub_query_info = [sub_table2qreps, sub_table2bitmaps, ts_to_keygroups[tuple_ts],
			                  sorted_ts, ts_to_joins[tuple_ts]]
			left_sub_query_info = [left_sub_table2qreps, left_sub_table2bitmaps, ts_to_keygroups[tuple_ts],
			                  sorted_ts, ts_to_joins[tuple_ts]]
			right_sub_query_info = [right_sub_table2qreps, right_sub_table2bitmaps, ts_to_keygroups[tuple_ts],
			                        sorted_ts, ts_to_joins[tuple_ts]]

			queries.append(sub_query_info)
			left_queries.append(left_sub_query_info)
			right_queries.append(right_sub_query_info)

	return queries, left_queries, right_queries


def sample_a_batch(sample_size, temp_q_rep, col2indicator, col2candidatevals, col2samplerange, eq_col_combos,
                   temp_joins, col2minmax, colid2featlen, all_cols, temp_subq_ts_list, ts_to_joins, ts_to_keygroups, cur):
	batch_queries = []
	batch_left_queries = []
	batch_right_queries = []

	for _ in range(sample_size):
		col2type, col2vals, col2vals_left, col2vals_right, t2bitmaps, left_bitmap, right_bitmap = sample_from_join_template(
			temp_q_rep, col2indicator, col2candidatevals, col2samplerange, eq_col_combos, 't.production_year', cur)

		queries, left_queries, right_queries = generate_sub_queries(temp_q_rep, temp_joins, col2minmax, colid2featlen, all_cols,
		                                                            temp_subq_ts_list, ts_to_joins, ts_to_keygroups,
		                                                            col2type, col2vals, t2bitmaps, left_bitmap,
		                                                            right_bitmap)

		batch_queries.extend(queries)
		batch_left_queries.extend(left_queries)
		batch_right_queries.extend(right_queries)

	return batch_queries, batch_left_queries, batch_right_queries