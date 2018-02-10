import math
import itertools
from util import *
from scipy.spatial.distance import *
import numpy as np
from sklearn.preprocessing import *


def save_pred_and_target_brain_vectors(model, sess,train_x, train_y,
               		  test_x, test_y,
                      test_words,train_words,FLAGS):
	save_pred_and_target_labesl(model, sess, test_x, test_y,test_words,FLAGS,"test")
	save_pred_and_target_labesl(model, sess, train_x, train_y,train_words,FLAGS,"train")


def save_pred_and_target_labesl(model, sess, test_x, test_y,test_words,FLAGS,label):
	predicted_output = sess.run([model.predicted_output],
                                feed_dict={model.input_states_batch: test_x, model.batch_size: len(test_x),
                                           model.p_keep_input: 1.0, model.p_keep_hidden: 1.0})
	np.save(FLAGS.log_root+"/predicted_output_"+label,predicted_output)
	np.save(FLAGS.log_root+"/target_output_"+label,test_y)
	np.save(FLAGS.log_root+"/words_"+label,test_words)


def extract_plotting_output(model,sess,x,y,words,output_name):
	predicted_outputs = sess.run([model.predicted_output],
                                feed_dict={model.input_states_batch: x, model.batch_size: len(words),
                                           model.p_keep_input: 1.0, model.p_keep_hidden: 1.0}
                                )
	e_dists = cdist(predicted_output, y, 'euclidean')
	c_dists = cdist(predicted_output, y, 'cosine')

	with open(output_name,'w') as fout:
		for i in np.arange(x_len):
			line_str = words[i] +" "+str(c_dist[i][i])+" "+str(e_dist[i][i])
			fout.write(line_str)





def qualitative_eval(model, sess, test_x, test_y, train_step,test_words,FLAGS,mode="test"):
	test_size = 100
	if len(test_x.shape) == 2:
		test = test_x[:test_size]
	elif len(test_x.shape) == 3:
		test = test_x[:,:test_size]
	predicted_output = sess.run([model.predicted_output],
                                feed_dict={model.input_states_batch: test, model.batch_size: test_size,
                                           model.p_keep_input: 1.0, model.p_keep_hidden: 1.0}
                                )
	print(predicted_output[0].shape)
	plot(predicted_output[0], test_y[:test_size], train_step, test_words[:test_size], int(math.sqrt(test_size)),FLAGS,mode)
	#plot_all2one(predicted_output[0], test_y[:test_size], train_step, test_words[:test_size], int(math.sqrt(test_size)),FLAGS)


def compare_brain_vectors(model, sess, test_x, test_y, train_step,test_words,FLAGS):
	test_size = 100
	predicted_output = sess.run([model.predicted_output],
                                feed_dict={model.input_states_batch: test_x[:test_size], model.batch_size: test_size,
                                           model.p_keep_input: 1.0, model.p_keep_hidden: 1.0}
                                )
	print(predicted_output[0].shape)
	plot_all2one(predicted_output[0], test_y[:test_size], train_step, test_words[:test_size], int(math.sqrt(test_size)),FLAGS)


def quantitative_eval(model, sess, test_x, test_y, train_step, test_words, FLAGS):
	predicted_output, mse , sd_error, mean_error = sess.run([ model.predicted_output, model.mean_squared_loss, model.sd_error,model.mean_error],feed_dict={model.input_states_batch: test_x, model.output_states_batch: test_y, model.batch_size: len(test_words), model.p_keep_input: 1.0, model.p_keep_hidden: 1.0})
	print("predicted output shape:",predicted_output.shape)

	e_dists = cdist(predicted_output, test_y, 'euclidean')
	dists = cdist(predicted_output, test_y, 'cosine')
	
	nn_index = np.argmin(dists,axis=1)
	#print(nn_index)
	#print("truth:",np.argmax(np.eye(len(nn_index)),axis=1))
	accuracy_on_test = np.mean(nn_index == np.argmax(np.eye(len(nn_index)),axis=1))
	#print(distances)
	print("accuracy_on_test:",accuracy_on_test)
	print("mse:",mse)
	print("sd_error:",sd_error)
	print("mean erros:",mean_error)

	b_acc = []
	e_b_acc = []
	for i,j in itertools.combinations(np.arange(len(test_y)), 2):
		right_match = dists[i,i] + dists[j,j]
		wrong_match = dists[i,j] + dists[j,i]
		b_acc.append(right_match < wrong_match)

		e_right_match = e_dists[i,i] + e_dists[j,j]
		e_wrong_match = e_dists[i,j] + e_dists[j,i]
		e_b_acc.append(e_right_match < e_wrong_match)

	print("binary accuracy: ", np.mean(b_acc)," ", np.mean(e_b_acc))

	return np.mean(b_acc)





def binary_accuracy_eval(model, sess, test_x, test_y, train_step, test_words, FLAGS):
	test_size =  len(test_x)
	predicted_output_all, mse_all , sd_error_all, mean_error_all,accuracy_on_test_all = [], [], [], [], []
	for i,j in itertools.combinations(np.arange(test_size), 2):
		predicted_output, mse , sd_error, mean_error, nearest_neighbor_on_test = sess.run([model.predicted_output, model.mean_squared_loss, model.sd_error,model.mean_error, model.nearest_neighbor_on_test],feed_dict={model.input_states_batch: test_x[[i,j],:], model.output_states_batch: test_y[[i,j],:], model.batch_size: 2, model.p_keep_input: 1.0, model.p_keep_hidden: 1.0})
		dists = cdist(predicted_output, test_y[:test_size], 'cosine')
		nn_index = np.argmin(dists,axis=1)
		accuracy_on_test = np.mean(nn_index == np.argmax(np.eye(len(nn_index)),axis=1))
		mse_all.append(predicted_output)
		sd_error_all.append(sd_error)
		mean_error_all.append(mean_error)
		accuracy_on_test_all.append(accuracy_on_test)


	print("accuracy_on_test:",np.mean(accuracy_on_test))
	print("mse:",np.mean(mse))
	print("sd_error:",np.mean(sd_error))
	print("mean erros:",np.mean(mean_error))


def nearest_neighbor(predicted, all_targets_tree, true_targets):
    dd, ii = all_targets_tree.query(predicted)
