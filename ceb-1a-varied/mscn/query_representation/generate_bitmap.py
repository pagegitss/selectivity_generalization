import pandas as pd
import numpy as np
import collections
import sys

table_dtype = {'aka_name': {'name': object, 'imdb_index': object, 'name_pcode_cf': object, 'name_pcode_nf': object,
                            'surname_pcode': object, 'md5sum': object},
               'aka_title': {'title': object, 'imdb_index': object, 'phonetic_code': object, 'note': object,
                             'md5sum': object}, 'cast_info': {'note': object},
               'char_name': {'name': object, 'imdb_index': object, 'name_pcode_nf': object, 'surname_pcode': object,
                             'md5sum': object}, 'comp_cast_type': {'kind': object},
               'company_name': {'name': object, 'country_code': object, 'name_pcode_nf': object,
                                'name_pcode_sf': object, 'md5sum': object}, 'company_type': {'kind': object},
               'complete_cast': {}, 'info_type': {'info': object},
               'keyword': {'keyword': object, 'phonetic_code': object}, 'kind_type': {'kind': object},
               'link_type': {'link': object}, 'movie_companies': {'note': object},
               'movie_info_idx': {'info': object, 'note': object}, 'movie_keyword': {}, 'movie_link': {},
               'name': {'name': object, 'imdb_index': object, 'gender': object, 'name_pcode_cf': object,
                        'name_pcode_nf': object, 'surname_pcode': object, 'md5sum': object},
               'role_type': {'role': object},
               'title': {'title': object, 'imdb_index': object, 'phonetic_code': object, 'series_years': object,
                         'md5sum': object}, 'movie_info': {'info': object, 'note': object},
               'person_info': {'info': object, 'note': object},
               'customer_address': {'ca_address_id': object, 'ca_street_number': object, 'ca_street_name': object,
                                    'ca_street_type': object, 'ca_suite_number': object, 'ca_city': object,
                                    'ca_county': object, 'ca_state': object, 'ca_zip': object, 'ca_country': object,
                                    'ca_location_type': object},
               'customer_demographics': {'cd_gender': object, 'cd_marital_status': object,
                                         'cd_education_status': object, 'cd_credit_rating': object},
               'date_dim': {'d_date_id': object, 'd_day_name': object, 'd_quarter_name': object, 'd_holiday': object,
                            'd_weekend': object, 'd_following_holiday': object, 'd_current_day': object,
                            'd_current_week': object, 'd_current_month': object, 'd_current_quarter': object,
                            'd_current_year': object, 'd_date': object},
               'warehouse': {'w_warehouse_id': object, 'w_warehouse_name': object, 'w_street_number': object,
                             'w_street_name': object, 'w_street_type': object, 'w_suite_number': object,
                             'w_city': object, 'w_county': object, 'w_state': object, 'w_zip': object,
                             'w_country': object},
               'ship_mode': {'sm_ship_mode_id': object, 'sm_type': object, 'sm_code': object, 'sm_carrier': object,
                             'sm_contract': object},
               'time_dim': {'t_time_id': object, 't_am_pm': object, 't_shift': object, 't_sub_shift': object,
                            't_meal_time': object}, 'reason': {'r_reason_id': object, 'r_reason_desc': object},
               'income_band': {},
               'item': {'i_item_id': object, 'i_item_desc': object, 'i_brand': object, 'i_class': object,
                        'i_category': object, 'i_manufact': object, 'i_size': object, 'i_formulation': object,
                        'i_color': object, 'i_units': object, 'i_container': object, 'i_product_name': object,
                        'i_rec_start_date': object, 'i_rec_end_date': object},
               'store': {'s_store_id': object, 's_store_name': object, 's_hours': object, 's_manager': object,
                         's_geography_class': object, 's_market_desc': object, 's_market_manager': object,
                         's_division_name': object, 's_company_name': object, 's_street_number': object,
                         's_street_name': object, 's_street_type': object, 's_suite_number': object, 's_city': object,
                         's_county': object, 's_state': object, 's_zip': object, 's_country': object,
                         's_rec_start_date': object, 's_rec_end_date': object},
               'call_center': {'cc_call_center_id': object, 'cc_name': object, 'cc_class': object, 'cc_hours': object,
                               'cc_manager': object, 'cc_mkt_class': object, 'cc_mkt_desc': object,
                               'cc_market_manager': object, 'cc_division_name': object, 'cc_company_name': object,
                               'cc_street_number': object, 'cc_street_name': object, 'cc_street_type': object,
                               'cc_suite_number': object, 'cc_city': object, 'cc_county': object, 'cc_state': object,
                               'cc_zip': object, 'cc_country': object, 'cc_rec_start_date': object,
                               'cc_rec_end_date': object},
               'customer': {'c_customer_id': object, 'c_salutation': object, 'c_first_name': object,
                            'c_last_name': object, 'c_preferred_cust_flag': object, 'c_birth_country': object,
                            'c_login': object, 'c_email_address': object},
               'web_site': {'web_site_id': object, 'web_name': object, 'web_class': object, 'web_manager': object,
                            'web_mkt_class': object, 'web_mkt_desc': object, 'web_market_manager': object,
                            'web_company_name': object, 'web_street_number': object, 'web_street_name': object,
                            'web_street_type': object, 'web_suite_number': object, 'web_city': object,
                            'web_county': object, 'web_state': object, 'web_zip': object, 'web_country': object,
                            'web_rec_start_date': object, 'web_rec_end_date': object}, 'store_returns': {},
               'household_demographics': {'hd_buy_potential': object},
               'web_page': {'wp_web_page_id': object, 'wp_autogen_flag': object, 'wp_url': object, 'wp_type': object,
                            'wp_rec_start_date': object, 'wp_rec_end_date': object},
               'promotion': {'p_promo_id': object, 'p_promo_name': object, 'p_channel_dmail': object,
                             'p_channel_email': object, 'p_channel_catalog': object, 'p_channel_tv': object,
                             'p_channel_radio': object, 'p_channel_press': object, 'p_channel_event': object,
                             'p_channel_demo': object, 'p_channel_details': object, 'p_purpose': object,
                             'p_discount_active': object},
               'catalog_page': {'cp_catalog_page_id': object, 'cp_department': object, 'cp_description': object,
                                'cp_type': object}, 'inventory': {}, 'catalog_returns': {}, 'web_returns': {},
               'web_sales': {}, 'catalog_sales': {}, 'store_sales': {}}
TPCDS_ALIAS_DICT = {'ss': 'store_sales', 'sr': 'store_returns', 'cs': 'catalog_sales', 'cr': 'catalog_returns',
                    'ws': 'web_sales', 'wr': 'web_returns', 'inv': 'inventory', 's': 'store', 'cc': 'call_center',
                    'cp': 'catalog_page', 'web': 'web_site', 'wp': 'web_page', 'w': 'warehouse', 'c': 'customer',
                    'ca': 'customer_address', 'cd': 'customer_demographics', 'd': 'date_dim',
                    'hd': 'household_demographics', 'i': 'item', 'ib': 'income_band', 'p': 'promotion', 'r': 'reason',
                    'sm': 'ship_mode', 't': 'time_dim'}
JOB_ALIAS_DICT = {'n': 'name', 'mc': 'movie_companies', 'an': 'aka_name', 'mi': 'movie_info', 'mk': 'movie_keyword',
                  'pi': 'person_info', 'cct': 'comp_cast_type', 'cc': 'complete_cast', 'ch_n': 'char_name',
                  'ml': 'movie_link', 'ct': 'company_type', 'ci': 'cast_info', 'it': 'info_type', 'cn': 'company_name',
                  'at': 'aka_title', 'kt': 'kind_type', 'rt': 'role_type', 'mi_idx': 'movie_info_idx', 'k': 'keyword',
                  'lt': 'link_type', 't': 'title'}


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

# !/usr/bin/python
import psycopg2
import sys
import re


# SAMPLE_BIT_SIZE = 1000

def connect_pg():
	DB_NAME = "imdb"
	DB_USER = "postgres"
	DB_PASS = "123456"
	DB_HOST = "localhost"
	DB_PORT = "5432"

	try:
		conn = psycopg2.connect(database=DB_NAME,
		                        user=DB_USER,
		                        password=DB_PASS,
		                        host=DB_HOST,
		                        port=DB_PORT)
		print("Database connected successfully")
	except:
		print("Database not connected successfully")

	conn.set_session('read uncommitted', autocommit=True)
	cur = conn.cursor()
	return cur

def get_bitmaps(query, cur=None):
	query = query.replace("COUNT(*)", "id")
	query = query.replace("FROM ", "FROM sampled_")
	cur.execute(query)
	explain_results = cur.fetchall()

	res = []
	for id_res in explain_results:
		res.append(id_res[0])
	return res

def get_join_bitmaps(query, key_group, is_pk=True, jk_name=None, cur=None):
	if is_pk is True:
		query = query.replace("COUNT(*)", "sid, id")
	else:
		query = query.replace("COUNT(*)", jk_name)

	query = query.replace("FROM ", f"FROM jb_{key_group}_")
	cur.execute(query)
	explain_results = cur.fetchall()

	res = []
	for id_res in explain_results:
		if is_pk is True:
			res.append([id_res[0], id_res[1]])
		else:
			if id_res[0] not in res:
				res.append(id_res[0])
	return res

def get_sampling_est(sql_query, table_list, cur=None):
	# query = query.replace("FROM ", "FROM sampled_")
	if 'WHERE' in sql_query:
		prefix = "rs_"
		# Regular expression to match the segment of the query between FROM and WHERE
		pattern = re.compile(r"\bFROM\b(.*?)\bWHERE\b", re.IGNORECASE | re.DOTALL)

		# Find the portion of the query that contains the tables between FROM and WHERE
		from_where_part = re.search(pattern, sql_query)
		if not from_where_part:
			modified_query = sql_query.replace("FROM ", "FROM rs_")
			return modified_query  # No modification if FROM or WHERE not found

		from_where_section = from_where_part.group(1)

		# Function to add prefix to the table name but not to aliases or other keywords
		def prefix_table_name(match):
			parts = match.group(0).split()
			if len(parts) == 1:
				# No alias, just a table name
				return prefix + parts[0]
			elif len(parts) > 1:
				# Handle 'table AS alias' or 'table alias'
				return prefix + parts[0] + " " + " ".join(parts[1:])
			return match.group(0)

		# Use regex to identify table references and add prefix
		prefixed_from_where_section = re.sub(r"(\b\w+\b(?:\s+AS\s+\w+|\s+\w+)?)", prefix_table_name, from_where_section,
		                                     flags=re.IGNORECASE)

		# Replace the original FROM-WHERE section with the modified one
		modified_query = sql_query[:from_where_part.start(1)] + prefixed_from_where_section + sql_query[
		                                                                                      from_where_part.end(1):]
	else:
		modified_query = sql_query.replace("FROM ", "FROM rs_")

	cur.execute(modified_query)
	explain_results = cur.fetchall()

	scaling_ratio = 1
	for t in table_list:
		if TABLE_SIZES[t] > 1000:
			per_table_scaling = TABLE_SIZES[t] / 1000.
		else:
			per_table_scaling = 1
		scaling_ratio = scaling_ratio * per_table_scaling

	return int(explain_results[0][0] * scaling_ratio)

def get_pg_est(sql_query, cur=None):
	sql_query = sql_query.replace("COUNT(*)", '*')
	# Prepare the query to get the plan and extract the estimated rows
	explain_query = f"EXPLAIN (FORMAT JSON) {sql_query}"
	cur.execute(explain_query)

	# Fetch the JSON output from EXPLAIN
	plans = cur.fetchone()

	# Extract the estimated rows from the plan
	estimated_rows = plans[0][0]['Plan']['Plan Rows']

	return estimated_rows

