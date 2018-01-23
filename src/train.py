import numpy as np


from eval import *


def train(model, sess, sv, train_x, train_y, test_x, test_y, test_words, train_words,FLAGS):
    # Use the session to train the graph.
    training_step = 0
    while not sv.should_stop():
        for i in range(model.hparams.number_of_epochs):

            XY = list(zip(train_x, train_y))
            np.random.shuffle(XY)
            train_x, train_y = zip(*XY)
            print(i)
            start_index = 0;
            end_index = start_index + model.hparams.batch_size
            while end_index < len(train_x):
                summary, _, cost, mse = sess.run([model.summ_op,model.train_op, model.cost, model.mean_squared_loss],
                                        feed_dict={model.input_states_batch: train_x[start_index:end_index],
                                                   model.output_states_batch: train_y[start_index:end_index],
                                                   model.p_keep_input: 0.8, model.p_keep_hidden: 0.5})
                print("mse %f" % mse)
                start_index = end_index
                end_index = start_index + model.hparams.batch_size
                training_step += 1

            sv.summary_computed(sess, summary)
            qualitative_eval(model, sess, test_x, test_y, training_step,test_words,FLAGS)
