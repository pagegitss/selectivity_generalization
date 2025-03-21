import json
import numpy as np
from scipy.stats import gmean

def get_join_qerror(preds, targets, max_card, workload_type='In-Distribution', res_file=None, epoch_id=None):
	qerror = []
	for i in range(len(targets)):
		if preds[i] <= 1.:
			preds[i] = 1.

		if (preds[i] > targets[i]):
			qerror.append(preds[i] / targets[i])
		else:
			qerror.append(targets[i] / preds[i])

	preds = np.array(preds)
	targets = np.array(targets)

	rmse = np.sqrt(np.mean(np.square(preds/max_card - targets/max_card)))

	print("Test workload:{}: RMSE:{}, Mean: {}, GMean: {}, Median: {}, 90: {}; 95: {}; 99: {}; Max:{}".format(
		workload_type, rmse, np.mean(qerror), gmean(qerror), np.median(qerror), np.percentile(qerror, 90),
		np.percentile(qerror, 95), np.percentile(qerror, 99), np.max(qerror)))

	if res_file is not None:
		res_file.write(
			"epoch_id:{}; Test workload:{}: RMSE:{}, Mean: {}, GMean: {}, Median: {}, 90: {}; 95: {}; 99: {}; Max:{} \n".format(
				epoch_id, workload_type, rmse, np.mean(qerror), gmean(qerror), np.median(qerror), np.percentile(qerror, 90),
				np.percentile(qerror, 95), np.percentile(qerror, 99), np.max(qerror)))
		res_file.flush()
	return qerror