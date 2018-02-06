import numpy as np


from eval import *


def train(model, sess, sv, train_x, test_x, train_word, test_words, FLAGS,best_saver,best_dir):
    # Use the session to train the graph.
    
    training_step = 0
    current_best = 0
    for i in range(model.hparams.number_of_epochs):

        indexes = np.arange(len(train_x))
        np.random.shuffle(indexes)
        print(i)
        start_index = 0
        end_index = start_index + model.hparams.batch_size
        while end_index < len(indexes):
            summary, _, mse = sess.run([model.summ_op,model.train_op, model.mean_squared_loss],
                                    feed_dict={model.input_states_batch: train_x[indexes[start_index:end_index]
                                               ,model.batch_size: len(indexes[start_index:end_index]),
                                               model.p_keep_input: .6, model.p_keep_hidden: 0.9})
            print("mse %f" % mse)
            start_index = end_index
            end_index = start_index + model.hparams.batch_size
            training_step += 1

        sv.summary_computed(sess, summary)
        print(test_x.shape)
        print("epoch: ",i)
        #qualitative_eval(model, sess, test_x, test_y, training_step,test_words,FLAGS)
        #qualitative_eval(model, sess, train_x, train_y, training_step,train_words,FLAGS,"train")

        test_accuracy = quantitative_eval(model, sess, test_x, test_x, training_step,test_words,FLAGS)
        print("evaluation on the training set:")
        quantitative_eval(model, sess, train_x, train_x, training_step,train_words,FLAGS)
        if test_accuracy >= current_best:
            best_saver.save(sess,best_dir+"/best_model_epoch"+str(i))
            current_best = test_accuracy

