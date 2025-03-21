import copy
import hashlib
import torch
import torch.nn as nn
import numpy as np
import random
import csv
import os

from mscn.secon_utils import *
import pickle

import re
from collections import defaultdict
from datetime import date, datetime


def default_serializer(obj):
	"""JSON serializer for objects not serializable by default json code"""
	if isinstance(obj, (date, datetime)):
		return obj.isoformat()
	elif isinstance(obj, np.ndarray):
		return obj.tolist()
	raise TypeError(f"Type {type(obj)} not serializable")


TEXT_COLS = ['t.title', 'mi.info', 'ci.note', 'n.name',
             'n.name_pcode_nf', 'n.name_pcode_cf', 'n.surname_pcode',
             'k.keyword', 'cn.name', 'pi.info']

IN_TEXT_COLS = ['t.title', 'mi.info', 'ci.note',
                'n.name_pcode_nf', 'n.name_pcode_cf', 'n.surname_pcode',
                'k.keyword', 'cn.name']

CATEGORICAL_COLS = ['kt.kind', 'rt.role', 'n.gender', 'cn.country_code', 'ct.kind']
CATEGORICAL_COLS_IDS = [1, 1, 4, 2, 1]
CATEGORICAL_COLS_VALS = {
	'kt.kind': ['episode', 'movie', 'tv mini series', 'tv movie', 'tv series', 'video game', 'video movie'],
	'rt.role': ['actor', 'actress', 'cinematographer', 'composer', 'costume designer', 'director', 'editor', 'guest',
	            'miscellaneous crew', 'producer', 'production designer', 'writer'],
	'n.gender': ['NULL', 'f', 'm'],
	'cn.country_code': ['NULL', '[ad]', '[ae]', '[af]', '[ag]', '[ai]', '[al]', '[am]', '[an]', '[ao]', '[ar]', '[as]',
	                    '[at]', '[au]', '[aw]', '[az]', '[ba]', '[bb]', '[bd]', '[be]', '[bf]', '[bg]', '[bh]', '[bi]',
	                    '[bj]', '[bl]', '[bm]', '[bn]', '[bo]', '[br]', '[bs]', '[bt]', '[bw]', '[by]', '[bz]', '[ca]',
	                    '[cd]', '[cg]', '[ch]', '[ci]', '[cl]', '[cm]', '[cn]', '[co]', '[cr]', '[cshh]', '[cu]',
	                    '[cv]', '[cy]', '[cz]', '[ddde]', '[de]', '[dk]', '[dm]', '[do]', '[dz]', '[ec]', '[ee]',
	                    '[eg]', '[er]', '[es]', '[et]', '[fi]', '[fj]', '[fo]', '[fr]', '[ga]', '[gb]', '[gd]', '[ge]',
	                    '[gf]', '[gg]', '[gh]', '[gi]', '[gl]', '[gn]', '[gp]', '[gr]', '[gt]', '[gu]', '[gw]', '[gy]',
	                    '[hk]', '[hn]', '[hr]', '[ht]', '[hu]', '[id]', '[ie]', '[il]', '[im]', '[in]', '[iq]', '[ir]',
	                    '[is]', '[it]', '[je]', '[jm]', '[jo]', '[jp]', '[ke]', '[kg]', '[kh]', '[ki]', '[kn]', '[kp]',
	                    '[kr]', '[kw]', '[ky]', '[kz]', '[la]', '[lb]', '[lc]', '[li]', '[lk]', '[lr]', '[ls]', '[lt]',
	                    '[lu]', '[lv]', '[ly]', '[ma]', '[mc]', '[md]', '[me]', '[mg]', '[mh]', '[mk]', '[ml]', '[mm]',
	                    '[mn]', '[mo]', '[mq]', '[mr]', '[mt]', '[mu]', '[mv]', '[mx]', '[my]', '[mz]', '[na]', '[nc]',
	                    '[ne]', '[ng]', '[ni]', '[nl]', '[no]', '[np]', '[nr]', '[nz]', '[om]', '[pa]', '[pe]', '[pf]',
	                    '[pg]', '[ph]', '[pk]', '[pl]', '[pm]', '[pr]', '[ps]', '[pt]', '[py]', '[qa]', '[ro]', '[rs]',
	                    '[ru]', '[rw]', '[sa]', '[sd]', '[se]', '[sg]', '[si]', '[sj]', '[sk]', '[sl]', '[sm]', '[sn]',
	                    '[so]', '[sr]', '[suhh]', '[sv]', '[sy]', '[sz]', '[td]', '[tf]', '[tg]', '[th]', '[tj]',
	                    '[tk]', '[tl]', '[tm]', '[tn]', '[to]', '[tr]', '[tt]', '[tv]', '[tw]', '[tz]', '[ua]', '[ug]',
	                    '[um]', '[us]', '[uy]', '[uz]', '[va]', '[ve]', '[vg]', '[vi]', '[vn]', '[xyu]', '[ye]',
	                    '[yucs]', '[za]', '[zm]', '[zw]'],
	'ct.kind': ['distributors', 'miscellaneous companies', 'production companies', 'special effects companies']}

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

JOIN_MAP_IMDB_UPPER_BOUND = {
	'movie_id': 2528312,
	'keyword': 134170,
	'person_id': 4167491,
	'kind_id': 7,
	'role_id': 12,
	'info_id': 113,
	'company_type': 4,
	'company_id': 234997,
}

FILTER_COLS = {'title': ['t.production_year', 't.title'],
               'kind_type': ['kt.kind'],
               'keyword': ['k.keyword'],
               'movie_info': ['mi.info'],
               'movie_info_idx': ['mii.info'],
               'info_type': ['it.id'],
               'cast_info': ['ci.note'],
               'role_type': ['rt.role'],
               'name': ['n.gender', 'n.name', 'n.name_pcode_nf', 'n.name_pcode_cf', 'n.surname_pcode'],
               'movie_companies': [],
               'movie_keyword': [],
               'company_name': ['cn.country_code', 'cn.name'],
               'company_type': ['ct.kind'],
               'aka_name': [],
               'person_info': ['pi.info']}

LIKE_COLS = {'title': ['t.title'],
             'kind_type': ['kt.kind'],
             'movie_info': ['mi.info'],
             'name': ['n.name', 'n.name_pcode_nf', 'n.name_pcode_cf', 'n.surname_pcode'],
             'company_name': ['cn.name'],
             'person_info': ['pi.info']}

# LIKE_COLS = {'t.title', 'kt.kind', 'mi.info', 'n.name',
# 			 'n.name_pcode_nf', 'n.name_pcode_cf', 'n.surname_pcode', 'cn.name', 'pi.info'}
primary_key_dic = {
	'movie_id': "title.id",
	'info_id': "info_type.id",
	'kind_id': "kind_type.id",
	'person_id': "name.id",
	'role_id': "role_type.id"
}

PLAIN_FILTER_COLS = []
for t in FILTER_COLS:
	PLAIN_FILTER_COLS.extend(FILTER_COLS[t])

TABLE_SIZES = {'title': 2528312,
               'kind_type': 7,
               'keyword': 134170,
               'movie_info': 14835720,
               'movie_info_idx': 1380035,
               #    'movie_link':29997,
               'info_type': 113,
               'cast_info': 36244344,
               'role_type': 12,
               'name': 4167491,
               'movie_companies': 2609129,
               'movie_keyword': 4523930,
               'company_name': 234997,
               'company_type': 4,
               'aka_name': 901343,
               'person_info': 2963664}

PURE_LIKE_COLS = list(set(TEXT_COLS) - set(IN_TEXT_COLS))
IN_BUCKETS = 100

class UnionFind:
	def __init__(self):
		self.parent = {}

	def find(self, item):
		if item not in self.parent:
			self.parent[item] = item
		if self.parent[item] != item:
			self.parent[item] = self.find(self.parent[item])
		return self.parent[item]

	def union(self, item1, item2):
		root1 = self.find(item1)
		root2 = self.find(item2)
		if root1 != root2:
			self.parent[root2] = root1


def find_connected_clusters(list_of_lists):
	uf = UnionFind()

	for sublist in list_of_lists:
		for i in range(len(sublist) - 1):
			uf.union(sublist[i], sublist[i + 1])

	clusters = defaultdict(list)
	for item in uf.parent:
		root = uf.find(item)
		clusters[root].append(item)

	return list(clusters.values())


def get_table_id(table):
	keys = list(FILTER_COLS.keys())
	return keys.index(table)

def deterministic_hash(string):
	return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

def _handle_like(val):
	like_val = val[0]
	pfeats = np.zeros(IN_BUCKETS + 2)
	regex_val = like_val.replace("%", "")
	pred_idx = deterministic_hash(regex_val) % IN_BUCKETS
	pfeats[pred_idx] = 1.00

	for v in regex_val:
		pred_idx = deterministic_hash(str(v)) % IN_BUCKETS
		pfeats[pred_idx] = 1.00

	for i, v in enumerate(regex_val):
		if i != len(regex_val) - 1:
			pred_idx = deterministic_hash(v + regex_val[i + 1]) % IN_BUCKETS
			pfeats[pred_idx] = 1.00

	for i, v in enumerate(regex_val):
		if i < len(regex_val) - 2:
			pred_idx = deterministic_hash(v + regex_val[i + 1] + \
			                              regex_val[i + 2]) % IN_BUCKETS
			pfeats[pred_idx] = 1.00

	pfeats[IN_BUCKETS] = len(regex_val)

	# regex has num or not feature
	if bool(re.search(r'\d', regex_val)):
		pfeats[IN_BUCKETS + 1] = 1

	return pfeats


def drop_trailing_number(s):
	return re.sub(r'\d+$', '', s)


def get_table_info(file_path='./mscn/column_min_max_vals_imdb.csv'):
	lines = open(file_path, 'r').readlines()

	table_join_keys = {}
	table_text_cols = {}
	table_normal_cols = {}
	col_type = {}
	col2minmax = {}
	table_dim_list = []
	table_like_dim_list = []
	table_list = []

	### get min/max for cols
	for line in lines[1:]:
		parts = line.strip().split(',')

		col = parts[0]
		min_v = int(parts[1])
		max_v = int(parts[2])

		col2minmax[col] = [min_v, max_v + 1]
	#########

	### build table info
	for table in FILTER_COLS:
		# build filter cols info
		table_cols = FILTER_COLS[table]
		table_dim_list.append(len(table_cols))

		if table in LIKE_COLS:
			table_like_dim_list.append(len(LIKE_COLS[table]))
		else:
			table_like_dim_list.append(0)

		table_list.append(table)
		table_text_cols[table] = []
		table_normal_cols[table] = []
		for col in table_cols:
			full_col_name = col
			if full_col_name in TEXT_COLS:
				table_text_cols[table].append(full_col_name)
				col_type[full_col_name] = 'text'
				if full_col_name in IN_TEXT_COLS:
					col2minmax[col] = [0, IN_BUCKETS + 1]
			else:
				table_normal_cols[table].append(full_col_name)
				if full_col_name in CATEGORICAL_COLS:
					col_type[full_col_name] = 'categorical'
					num_dist_vals = len(CATEGORICAL_COLS_VALS[full_col_name])
					col2minmax[col] = [0, num_dist_vals]
				else:
					col_type[full_col_name] = 'number'

	table_key_groups = {}
	### build table join keys info
	for col_name in JOIN_MAP_IMDB:
		table = col_name.split('.')[0]
		# table = reverse_alias[alias_table]
		key_group = JOIN_MAP_IMDB[col_name]


		if key_group not in table_key_groups:
			table_key_groups[key_group] = [table]
		else:
			if table not in table_key_groups[key_group]:
				table_key_groups[key_group].append(table)

		if table not in table_join_keys:
			table_join_keys[table] = [key_group]
		else:
			if key_group not in table_join_keys[table]:
				table_join_keys[table].append(key_group)

	table_sizes = TABLE_SIZES

	return (table_list, table_dim_list, table_like_dim_list, table_sizes, table_key_groups,
	        table_join_keys, table_text_cols, table_normal_cols, col_type, col2minmax)

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

def read_query_file_for_mscn_w_bitmaps(col2minmax, num_q=10000, test_size=1000, shifting_type='granularity', directory_list=['/home/imdb/2b/'],
                                       saved_ditectory="/home/processed_workloads/imdb/"):

	shifting_t = 't'
	shifting_col = 't.production_year'

	workload_file_path = "{}processed_ceb_1a_varied_{}.pkl".format(saved_ditectory, shifting_type)
	print(workload_file_path)

	if os.path.exists(workload_file_path):
		print('found')
		with open(workload_file_path, "rb") as pickle_file:
			loaded_dict = pickle.load(pickle_file)
			template2queries = loaded_dict['template2queries']
			template2cards = loaded_dict['template2cards']
			test_template2queries = loaded_dict['test_template2queries']
			test_template2cards = loaded_dict['test_template2cards']
			colid2featlen = loaded_dict['colid2featlen']
			all_table_alias = loaded_dict['all_table_alias']
			all_joins = loaded_dict['all_joins']
			all_cols = loaded_dict['all_cols']
			max_num_tables = loaded_dict['max_num_tables']
			max_num_joins = loaded_dict['max_num_joins']

			col2indicator = loaded_dict['col2indicator']
			col2candidatevals = loaded_dict['col2candidatevals']
			col2samplerange = loaded_dict['col2samplerange']
			eq_col_combos = loaded_dict['eq_col_combos']
			temp_q_rep = loaded_dict['temp_q_rep']

			ts_to_joins = loaded_dict['ts_to_joins']
			ts_to_keygroups = loaded_dict['ts_to_keygroups']

		return template2queries, template2cards, test_template2queries, test_template2cards, colid2featlen, all_table_alias, all_joins, all_cols, max_num_tables, max_num_joins,\
				col2indicator, col2candidatevals, col2samplerange, eq_col_combos, temp_q_rep, ts_to_joins, ts_to_keygroups

	## num_q: max number of queries per JOB template
	training_queries = []
	training_cards = []

	test_queries = []
	test_cards = []

	saved_predicates = {}

	colid2featlen = {}

	all_table_alias = []
	all_joins = []
	all_cols = []
	max_num_tables = 0
	max_num_joins = 0
	random.seed(42)

	cur = connect_pg()

	col2indicator = {}
	col2candidatevals = {}
	col2samplerange = {}
	eq_col_combos = {}
	temp_q_rep = None
	eq_col_combos['it1.id,it2.id'] = []
	ts_to_joins = {}
	ts_to_keygroups = {}

	train_count = 0
	test_count = 0

	years_list = []

	for directory in directory_list:
		print("processing {}".format(directory))
		files = os.listdir(directory)
		# Filter out files that do not have a .csv extension
		files = [file for file in files if file.endswith('.pkl')]
		files.sort()

		for qid, file in enumerate(files):
			# print(file)
			file_path = os.path.join(directory, file)
			qrep = load_qrep(file_path)
			if qid == 0:
				temp_q_rep = qrep

			table2normal_predicates = {}
			table2text_predicates = {}
			table2qreps = {}
			table2bitmaps = {}
			table2joinbitmaps = {}

			### for SeConCDF


			left_table2qreps = {}
			right_table2qreps = {}

			tables, aliases = get_tables(qrep)

			q_is_training = True
			eq_col_to_val = {}

			global_origial_alias_to_aliases = {}
			for alias in aliases:
				original_alias = drop_trailing_number(alias)
				if original_alias not in global_origial_alias_to_aliases:
					global_origial_alias_to_aliases[original_alias] = [alias]
				else:
					global_origial_alias_to_aliases[original_alias].append(alias)

			joins = get_joins(qrep)
			preds, pred_cols, pred_types, pred_vals = get_predicates(qrep)

			trues = get_true_cardinalities(qrep)
			card = 0.
			full_k = []

			# get the true card
			for k, v in trues.items():
				if len(k) == len(tables):
					card = trues[k]
					full_k = k

			for alias_t in aliases:
				# table_id = get_table_id(full_t)
				table2normal_predicates[alias_t] = []
				table2text_predicates[alias_t] = []
				table2qreps[alias_t] = []
				left_table2qreps[alias_t] = []
				right_table2qreps[alias_t] = []

				sub_sql = subplan_to_sql(qrep, (alias_t,))
				bit_map = get_bitmaps(sub_sql, cur=cur)
				## add the bitmaps
				table2bitmaps[alias_t] = bit_map

			## handle join bitmaps
			for join in joins:
				parts = join.strip().split(' = ')

				alias_table1 = parts[0].split('.')[0]
				alias_table2 = parts[1].split('.')[0]

				col1 = parts[0].split('.')[1]
				col2 = parts[1].split('.')[1]

				full_key_name1 = reverse_alias[drop_trailing_number(alias_table1)] + '.' + col1
				full_key_name2 = reverse_alias[drop_trailing_number(alias_table2)] + '.' + col2

				key_group = JOIN_MAP_IMDB[full_key_name1]

				pk = primary_key_dic[key_group]

				if key_group not in table2joinbitmaps:
					table2joinbitmaps[key_group] = {}

				if pk != full_key_name1 and pk != full_key_name2:
					continue

				if full_key_name1 == pk:
					alias_pk = alias_table1
					alias_fk = alias_table2
					fk_name = col2
					pk_table = drop_trailing_number(alias_table1)
					fk_table = drop_trailing_number(alias_table2)
				else:
					alias_pk = alias_table2
					alias_fk = alias_table1
					fk_name = col1
					pk_table = drop_trailing_number(alias_table2)
					fk_table = drop_trailing_number(alias_table1)

				if alias_pk not in table2joinbitmaps[key_group]:
					pk_sub_sql = subplan_to_sql(qrep, (alias_pk,))
					pk_bit_map = get_join_bitmaps(pk_sub_sql, key_group, is_pk=True, cur=cur)
					table2joinbitmaps[key_group][alias_pk] = pk_bit_map

				if alias_fk not in table2joinbitmaps[key_group]:
					fk_sub_sql = subplan_to_sql(qrep, (alias_fk,))
					fk_bit_map = get_join_bitmaps(fk_sub_sql, key_group, is_pk=False, jk_name=fk_name, cur=cur)
					table2joinbitmaps[key_group][alias_fk] = fk_bit_map

			##############

			if len(aliases) > max_num_tables:
				max_num_tables = len(aliases)

			if len(joins) > max_num_joins:
				max_num_joins = len(joins)

			queried_tables = set([])
			query_signs = []

			shifting_col_vals = []
			shifting_col_min = 0
			shifting_col_max = 0
			### parse pred info
			for i in range(len(preds)):
				# per table
				vals_visited = []
				for j, pred in enumerate(preds[i]):
					col = pred_cols[i][j]
					op = pred_types[i][j]
					val = pred_vals[i][j]

					alias_t = col.split('.')[0]
					queried_tables.add(alias_t)
					full_t = reverse_alias[drop_trailing_number(alias_t)]
					# table_id = get_table_id(full_t)

					standard_col_name = ALIAS[full_t] + '.' + col.split('.')[1]

					if col not in all_cols:
						all_cols.append(col)
					col_id = all_cols.index(col)

					if standard_col_name not in PURE_LIKE_COLS:
						if val in vals_visited:
							break
						else:
							vals_visited.append(val)

						min_val = col2minmax[standard_col_name][0]
						max_val = col2minmax[standard_col_name][1]

						if op == 'lt':
							if standard_col_name == shifting_col:
								shifting_col_min = min_val
								shifting_col_max = max_val
							col2indicator[col] = True
							col2candidatevals[col] = [min_val, max_val]
							## handles something like year<2010 and year<2015
							if len(vals_visited) == 1:
								if val[0] is None:
									table2normal_predicates[alias_t].append(
										[col_id, [[0., normalize_val(val[1], min_val, max_val)]]])
									table2qreps[alias_t].append(
										[col_id, op, [0., normalize_val(val[1], min_val, max_val)]])
									left_table2qreps[alias_t].append(
										[col_id, op, [0., 0.]])
									right_table2qreps[alias_t].append(
										[col_id, op, [0., normalize_val(val[1], min_val, max_val)]])
									shifting_col_vals = [min_val, val[1]]
								elif val[1] is None:
									table2normal_predicates[alias_t].append(
										[col_id, [[normalize_val(val[0], min_val, max_val), 1.]]])
									table2qreps[alias_t].append(
										[col_id, op, [normalize_val(val[0], min_val, max_val), 1.]])
									left_table2qreps[alias_t].append(
										[col_id, op, [0, normalize_val(val[0], min_val, max_val)]])
									right_table2qreps[alias_t].append(
										[col_id, op, [0, 1.]])
									shifting_col_vals = [val[0], max_val]
								else:
									table2normal_predicates[alias_t].append(
										[col_id, [[normalize_val(val[0], min_val, max_val),
										           normalize_val(val[1], min_val, max_val)]]])
									table2qreps[alias_t].append(
										[col_id, op, [normalize_val(val[0], min_val, max_val),
										              normalize_val(val[1], min_val, max_val)]])
									left_table2qreps[alias_t].append(
										[col_id, op, [0.,
										              normalize_val(val[0], min_val, max_val)]])
									right_table2qreps[alias_t].append(
										[col_id, op, [0.,
										              normalize_val(val[1], min_val, max_val)]])
									shifting_col_vals = [val[0], val[1]]
							else:
								### len(vals_visited) >= 2
								for p in table2normal_predicates[alias_t]:
									if p[0] == col_id:
										if val[0] is None:
											p[1][0][1] = normalize_val(val[1], min_val, max_val)
										# table2normal_predicates[alias_t].append([col_id, [[0., normalize_val(val[1], min_val, max_val)]]])
										elif val[1] is None:
											p[1][0][0] = normalize_val(val[0], min_val, max_val)
										# table2normal_predicates[alias_t].append([col_id, [[normalize_val(val[0], min_val, max_val), 1.]]])
										else:
											p[1][0] = [normalize_val(val[0], min_val, max_val),
											           normalize_val(val[1], min_val, max_val)]
										break
								for p in table2qreps[alias_t]:
									if p[0] == col_id:
										if val[0] is None:
											p[2][1] = normalize_val(val[1], min_val, max_val)
											shifting_col_vals[1] = val[1]
										# table2normal_predicates[alias_t].append([col_id, [[0., normalize_val(val[1], min_val, max_val)]]])
										elif val[1] is None:
											p[2][0] = normalize_val(val[0], min_val, max_val)
											shifting_col_vals[0] = val[0]
										# table2normal_predicates[alias_t].append([col_id, [[normalize_val(val[0], min_val, max_val), 1.]]])
										else:
											p[2] = [normalize_val(val[0], min_val, max_val),
											        normalize_val(val[1], min_val, max_val)]
											shifting_col_vals = [val[0], val[1]]
										break
								for p in left_table2qreps[alias_t]:
									if p[0] == col_id:
										if val[0] is None:
											pass
										elif val[1] is None:
											p[2][1] = normalize_val(val[0], min_val, max_val)
										else:
											p[2] = [0., normalize_val(val[0], min_val, max_val)]
										break
								for p in right_table2qreps[alias_t]:
									if p[0] == col_id:
										if val[1] is None:
											pass
										elif val[0] is None:
											p[2][1] = normalize_val(val[1], min_val, max_val)
										else:
											p[2] = [0., normalize_val(val[1], min_val, max_val)]
										break

							if col_id not in colid2featlen:
								colid2featlen[col_id] = 2
							elif 2 > colid2featlen[col_id]:
								colid2featlen[col_id] = 2

						elif op == 'eq':
							if isinstance(val, dict):
								new_val = int(val['literal'])
							else:
								new_val = val
							normal_val1 = normalize_val(new_val, min_val, max_val)
							normal_val2 = normalize_val(new_val + 1, min_val, max_val)
							table2normal_predicates[alias_t].append([col_id, [[normal_val1, normal_val2]]])

							table2qreps[alias_t].append([col_id, op, normal_val1])
							left_table2qreps[alias_t].append([col_id, op, normal_val1])
							right_table2qreps[alias_t].append([col_id, op, normal_val1])

							if col_id not in colid2featlen:
								colid2featlen[col_id] = 1

							if isinstance(val, dict):
								eq_col_to_val[col] = val['literal']
							else:
								eq_col_to_val[col] = val


						elif op == 'in':
							normal_val_list = []
							col2indicator[col] = True
							### hack for OR n.gender
							if full_t == 'name' and val == ['NULL']:
								### has been processed before
								continue

							if standard_col_name == 'it.id':
								## unify ops for it.id since they are the same thing
								new_val = int(val[0])
								normal_val1 = normalize_val(new_val, min_val, max_val)
								normal_val2 = normalize_val(new_val + 1, min_val, max_val)
								table2normal_predicates[alias_t].append([col_id, [[normal_val1, normal_val2]]])

								table2qreps[alias_t].append([col_id, 'eq', normal_val1])
								left_table2qreps[alias_t].append([col_id, 'eq', normal_val1])
								right_table2qreps[alias_t].append([col_id, 'eq', normal_val1])

								if col_id not in colid2featlen:
									colid2featlen[col_id] = 1
								eq_col_to_val[col] = val[0]
								continue

							if pred.strip(" ").endswith('OR n.gender IS NULL)'):
								val.append('NULL')

							### processing for SeConCDF
							if col not in col2candidatevals:
								col2candidatevals[col] = []
								col2samplerange[col] = [len(val), len(val)]

							for item_val in val:
								if item_val not in col2candidatevals[col]:
									col2candidatevals[col].append(item_val)

							if len(val) < col2samplerange[col][0]:
								col2samplerange[col][0] = len(val)

							if len(val) > col2samplerange[col][1]:
								col2samplerange[col][1] = len(val)
							### end of processing

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
								# the case of 'it.id'
								qrep_val_list = np.zeros(113)
								if col_id not in colid2featlen:
									colid2featlen[col_id] = 113
								elif 113 > colid2featlen[col_id]:
									colid2featlen[col_id] = 113
								for item_val in val:
									item_val = int(item_val)
									qrep_val_list[item_val - 1] = 1.
									normal_val1 = normalize_val(item_val, min_val, max_val)
									normal_val2 = normalize_val(item_val + 1, min_val, max_val)
									normal_val_list.append([normal_val1, normal_val2])

							table2normal_predicates[alias_t].append([col_id, normal_val_list])
							table2qreps[alias_t].append([col_id, op, qrep_val_list])
							left_table2qreps[alias_t].append([col_id, op, qrep_val_list])
							right_table2qreps[alias_t].append([col_id, op, qrep_val_list])

						elif op == 'like':
							pfeats = _handle_like(val)
							like_col_id = LIKE_COLS[full_t].index(standard_col_name)
							col_vector = np.zeros(len(LIKE_COLS[full_t]))
							col_vector[like_col_id] = 1.
							table2text_predicates[alias_t].append(np.concatenate((col_vector, pfeats)))
							table2qreps[alias_t].append([col_id, op, pfeats])
							left_table2qreps[alias_t].append([col_id, op, pfeats])
							right_table2qreps[alias_t].append([col_id, op, pfeats])
							if col_id not in colid2featlen:
								colid2featlen[col_id] = len(pfeats)
							elif len(pfeats) > colid2featlen[col_id]:
								colid2featlen[col_id] = len(pfeats)
					else:
						# text cols, for LIKE
						pfeats = _handle_like(val)
						like_col_id = LIKE_COLS[full_t].index(standard_col_name)
						col_vector = np.zeros(len(LIKE_COLS[full_t]))
						col_vector[like_col_id] = 1.
						table2text_predicates[alias_t].append(np.concatenate((col_vector, pfeats)))

						table2qreps[alias_t].append([col_id, op, pfeats])
						left_table2qreps[alias_t].append([col_id, op, pfeats])
						right_table2qreps[alias_t].append([col_id, op, pfeats])

						if col_id not in colid2featlen:
							colid2featlen[col_id] = len(pfeats)
						elif len(pfeats) > colid2featlen[col_id]:
							colid2featlen[col_id] = len(pfeats)

			if shifting_col_vals not in years_list:
				years_list.append(shifting_col_vals)

			### get the bitmaps for left and right CDF queries
			if shifting_col_vals[0] is None:
				left_bitmap = None
				left_joinbitmap = [None, None]
				query_signs.append(0.)
			else:
				left_sql = "SELECT COUNT(*) FROM title AS t WHERE t.production_year <= {}".format(
					shifting_col_vals[0])
				left_bitmap = get_bitmaps(left_sql, cur=cur)

				left_joinbitmap1 = get_join_bitmaps(left_sql, 'movie_id', is_pk=True, cur=cur)
				left_joinbitmap2 = get_join_bitmaps(left_sql, 'kind_id', is_pk=False, jk_name='t.kind_id', cur=cur)
				left_joinbitmap = [left_joinbitmap1, left_joinbitmap2]

				query_signs.append(-1)

			if shifting_col_vals[1] is None:
				right_bitmap = list(range(1, 1001))
				right_joinbitmap = [list(range(1, 1001)), list(range(1, 1001))]
				query_signs.append(1)
			else:
				right_sql = "SELECT COUNT(*) FROM title AS t WHERE t.production_year <= {}".format(
					shifting_col_vals[1])
				right_bitmap = get_bitmaps(right_sql, cur=cur)

				right_joinbitmap1 = get_join_bitmaps(right_sql, 'movie_id', is_pk=True, cur=cur)
				right_joinbitmap2 = get_join_bitmaps(right_sql, 'kind_id', is_pk=False, jk_name='t.kind_id', cur=cur)
				right_joinbitmap = [right_joinbitmap1, right_joinbitmap2]

				query_signs.append(1)

			### finish getting bitmaps

			if shifting_col_vals[1] is None:
				right = shifting_col_max
			else:
				right = shifting_col_vals[1]

			if shifting_col_vals[0] is None:
				left = shifting_col_min
			else:
				left = shifting_col_vals[0]

			col_range = (right - left) / (shifting_col_max - shifting_col_min)

			if shifting_type == 'granularity':
				if col_range > 0.6:
					q_is_training = True
				elif col_range < 0.1:
					q_is_training = False
				else:
					q_is_training = None
			else:
				raise Exception("shifting_type is not defined")

			### process for SeConCDF
			combo_vals = [eq_col_to_val['it1.id'], eq_col_to_val['it2.id']]
			if combo_vals not in eq_col_combos['it1.id,it2.id']:
				eq_col_combos['it1.id,it2.id'].append(combo_vals)

			### parse join info
			key_groups = {}
			for join in joins:
				if join not in all_joins:
					all_joins.append(join)

			for alias in aliases:
				if alias not in all_table_alias:
					all_table_alias.append(alias)

			left_table2bitmaps = copy.deepcopy(table2bitmaps)
			right_table2bitmaps = copy.deepcopy(table2bitmaps)
			left_table2bitmaps[shifting_t] = left_bitmap
			right_table2bitmaps[shifting_t] = right_bitmap

			left_table2joinbitmaps = copy.deepcopy(table2joinbitmaps)
			right_table2joinbitmaps = copy.deepcopy(table2joinbitmaps)

			left_table2joinbitmaps['movie_id'][shifting_t] = left_joinbitmap[0]
			left_table2joinbitmaps['kind_id'][shifting_t] = left_joinbitmap[1]

			right_table2joinbitmaps['movie_id'][shifting_t] = right_joinbitmap[0]
			right_table2joinbitmaps['kind_id'][shifting_t] = right_joinbitmap[1]

			aliases.sort()

			### get pg_est
			full_sql = subplan_to_sql(qrep, full_k)
			pg_est = get_pg_est(full_sql, cur=cur)
			if pg_est == 0:
				pg_est = 1
			### finish pg est

			### get rs est
			full_t_list = []
			for alias_t in table2bitmaps:
				full_t_list.append(reverse_alias[drop_trailing_number(alias_t)])
			rs_est = get_sampling_est(full_sql, full_t_list, cur=cur)

			if rs_est == 0:
				rs_est = 1
			### finish rs est

			query_info = [table2normal_predicates, table2text_predicates, table2qreps, left_table2qreps, right_table2qreps,
			              table2bitmaps, left_table2bitmaps, right_table2bitmaps,
			              table2joinbitmaps, pg_est, rs_est, key_groups, aliases, joins, query_signs]

			if q_is_training is None:
				continue

			if qid < num_q:
				if q_is_training:
					training_queries.append(query_info)
					training_cards.append(card)
					train_count += 1
				else:
					test_queries.append(query_info)
					test_cards.append(card)
					test_count += 1

				### add subqueries
				for k, v in trues.items():

					global_new_k = list(k)
					sub_card = trues[k]

					### find the key groups

					sub_key_groups = []
					subquery_joins = subplan_to_joins(qrep, k)
					for join in subquery_joins:
						parts = join.strip().split(' = ')

						alias_table1 = parts[0].split('.')[0]
						alias_table2 = parts[1].split('.')[0]

						col1 = parts[0].split('.')[1]
						col2 = parts[1].split('.')[1]

						full_key_name1 = reverse_alias[drop_trailing_number(alias_table1)] + '.' + col1
						key_group = JOIN_MAP_IMDB[full_key_name1]
						if key_group not in sub_key_groups:
							sub_key_groups.append(key_group)

					##############

					sub_table2normal_predicates = {t: table2normal_predicates[t] for t in global_new_k}
					sub_table2text_predicates = {t: table2text_predicates[t] for t in global_new_k}
					sub_table2qreps = {t: table2qreps[t] for t in global_new_k}
					sub_left_table2qreps = {t: left_table2qreps[t] for t in global_new_k}
					sub_right_table2qreps = {t: right_table2qreps[t] for t in global_new_k}
					sub_table2bitmaps = {t: table2bitmaps[t] for t in global_new_k}
					sub_left_table2bitmaps = {t: left_table2bitmaps[t] for t in global_new_k}
					sub_right_table2bitmaps = {t: right_table2bitmaps[t] for t in global_new_k}
					sub_table2joinbitmaps = {}

					for key_group in sub_key_groups:
						sub_table2joinbitmaps[key_group] = {}
						group_tables = [t for t in global_new_k if t in list(table2joinbitmaps[key_group].keys())]
						sub_table2joinbitmaps[key_group] = {t: table2joinbitmaps[key_group][t] for t in group_tables}


					if shifting_t in list(k):
						sub_query_signs = query_signs
					else:
						sub_query_signs = [0., 1.]

					global_new_k.sort()

					### get pg_est
					sub_sql = subplan_to_sql(qrep, k)
					pg_est = get_pg_est(sub_sql, cur=cur)
					if pg_est == 0:
						pg_est = 1
					### finish pg est

					### get rs est
					sub_t_list = []
					for alias_t in k:
						sub_t_list.append(reverse_alias[drop_trailing_number(alias_t)])
					rs_est = get_sampling_est(sub_sql, sub_t_list, cur=cur)

					if rs_est == 0:
						rs_est = 1
					### finish rs est

					sub_query_info = [sub_table2normal_predicates, sub_table2text_predicates, sub_table2qreps, sub_left_table2qreps, sub_right_table2qreps,
					                  sub_table2bitmaps, sub_left_table2bitmaps, sub_right_table2bitmaps,
					                  sub_table2joinbitmaps, pg_est, rs_est, sub_key_groups, global_new_k]

					q_json_str = (json.dumps(sub_table2normal_predicates, sort_keys=True, default=default_serializer)
							+ json.dumps(sub_table2text_predicates, sort_keys=True, default=default_serializer))

					t_list_key = tuple(global_new_k)
					if t_list_key not in saved_predicates:
						saved_predicates[t_list_key] = set([])

					if q_json_str not in saved_predicates[t_list_key]:
						new_subquery_joins = []

						for join in subquery_joins:
							new_subquery_joins.append(join)
							if join not in all_joins:
								all_joins.append(join)

						ts_to_joins[tuple(global_new_k)] = new_subquery_joins
						ts_to_keygroups[tuple(global_new_k)] = sub_key_groups

						sub_query_info.append(new_subquery_joins)
						sub_query_info.append(sub_query_signs)

						if len(subquery_joins) > max_num_joins:
							max_num_joins = len(subquery_joins)

						saved_predicates[t_list_key].add(q_json_str)

						if (not q_is_training) and (shifting_t in list(k)):
							test_queries.append(sub_query_info)
							test_cards.append(sub_card)
						else:
							training_queries.append(sub_query_info)
							training_cards.append(sub_card)

	template2queries = {}
	template2cards = {}
	print(len(training_queries))

	for query_info, card in zip(training_queries, training_cards):
		table_list = query_info[-3]
		table_list = tuple(table_list)

		if table_list not in template2queries:
			template2queries[table_list] = [query_info]
			template2cards[table_list] = [card]
		else:
			template2queries[table_list].append(query_info)
			template2cards[table_list].append(card)

	### shuffle training sets
	for table_list in template2queries:
		zipped = list(zip(template2queries[table_list], template2cards[table_list]))
		random.shuffle(zipped)

		new_qs, new_cards = zip(*zipped)

		template2queries[table_list] = list(new_qs)
		template2cards[table_list] = list(new_cards)

		print(table_list)
		print(len(new_qs))

	test_template2queries = {}
	test_template2cards = {}

	for query_info, card in zip(test_queries, test_cards):
		table_list = query_info[-3]
		table_list = tuple(table_list)

		if table_list not in test_template2queries:
			test_template2queries[table_list] = [query_info]
			test_template2cards[table_list] = [card]
		else:
			test_template2queries[table_list].append(query_info)
			test_template2cards[table_list].append(card)

	### saved to file
	my_dict = {}
	my_dict['template2queries'] = template2queries
	my_dict['template2cards'] = template2cards
	my_dict['test_template2queries'] = test_template2queries
	my_dict['test_template2cards'] = test_template2cards
	my_dict['colid2featlen'] = colid2featlen
	my_dict['all_table_alias'] = all_table_alias
	my_dict['all_joins'] = all_joins
	my_dict['all_cols'] = all_cols
	my_dict['max_num_tables'] = max_num_tables
	my_dict['max_num_joins'] = max_num_joins

	my_dict['col2indicator'] = col2indicator
	my_dict['col2candidatevals'] = col2candidatevals
	my_dict['col2samplerange'] = col2samplerange
	my_dict['eq_col_combos'] = eq_col_combos
	my_dict['temp_q_rep'] = temp_q_rep

	my_dict['ts_to_joins'] = ts_to_joins
	my_dict['ts_to_keygroups'] = ts_to_keygroups

	with open(workload_file_path, "wb") as pickle_file:
		pickle.dump(my_dict, pickle_file)


	print('numbers')
	print(train_count)
	print(test_count)

	return template2queries, template2cards, test_template2queries, test_template2cards, colid2featlen, all_table_alias, all_joins, all_cols, max_num_tables, max_num_joins,\
			col2indicator, col2candidatevals, col2samplerange, eq_col_combos, temp_q_rep, ts_to_joins, ts_to_keygroups


