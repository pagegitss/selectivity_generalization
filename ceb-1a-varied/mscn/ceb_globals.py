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