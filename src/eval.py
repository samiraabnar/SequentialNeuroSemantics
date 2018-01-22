import math
from util import *

def qualitative_eval(model, sess, test_x, test_y, train_step,test_words,FLAGS):
    test_size = 100
    predicted_output = sess.run([model.predicted_output],
                                feed_dict={model.input_states_batch: test_x[:test_size],
                                           model.p_keep_input: 1.0, model.p_keep_hidden: 1.0}
                                )

    print(predicted_output[0].shape)
    plot(predicted_output[0], test_y[:test_size], train_step, test_words[:test_size], int(math.sqrt(test_size)),FLAGS)
