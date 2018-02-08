import numpy as np


from eval import *


def train(model, sess, sv, train_x, train_y, test_x, test_y, test_words, train_words,FLAGS,best_saver,best_dir):
    # Use the session to train the graph.
    training_step = 0
    current_best = 0
    for i in range(model.hparams.number_of_epochs):

        indexes = np.arange(train_x.shape[0])
        np.random.shuffle(indexes)
        start_index = 0;
        end_index = start_index + model.hparams.batch_size
        while end_index < len(train_x):
            summary, _, mse = sess.run([model.summ_op,model.train_op, model.mean_squared_loss],
                                    feed_dict={model.input_states_batch: train_x[indexes[start_index:end_index]],
                                               model.output_states_batch: train_y[indexes[start_index:end_index]], model.batch_size: len(train_x[indexes[start_index:end_index]]),
                                               model.p_keep_input: FLAGS.p_keep_input, model.p_keep_hidden: FLAGS.p_keep_hidden})
            print("mse %f" % mse)
            start_index = end_index
            end_index = start_index + model.hparams.batch_size
            training_step += 1

        sv.summary_computed(sess, summary)
        print("epoch:",i)
        print(test_x.shape)
        print(test_y.shape)
        qualitative_eval(model, sess, test_x, test_y, training_step,test_words,FLAGS)
        qualitative_eval(model, sess, train_x, train_y, training_step,train_words,FLAGS,"train")

        test_accuracy = quantitative_eval(model, sess, test_x, test_y, training_step,test_words,FLAGS)

        print("evaluation on the training set:")
        quantitative_eval(model, sess, train_x, train_y, training_step,test_words,FLAGS)

        if test_accuracy >= current_best:
            best_saver.save(sess,best_dir+"/best_model_epoch"+str(i)+"_"+str(test_accuracy))
            current_best = test_accuracy
