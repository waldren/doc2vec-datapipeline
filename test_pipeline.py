import EmbeddingPipeline as ep
import numpy as np
import os
import tensorflow as tf
print(tf.__version__)
# 1.13

# Create Embedding Pipeline with simple stop_words
pl = ep.EmbeddingPipeline(stop_words=["the", "is"])

# Create a simple array of sentences for testing
text = np.array(["My first sentence", "They're is wrong sentence", "The 3rd sentence is", "Quite long sentence for the python script to process, let's hope it can be processed okay"])
labels =np.array(["ONE", "TWO", "THREE", "FOUR"])

# Check that the dictionary can build
print(pl.build_dictionary(text))

# Setup a session to run the pipeline test
sess = tf.Session()
# using sess.run does not matter here
ds = sess.run(pl.generate_dataset(text, labels))

# create iterator for the dataset
it = ds.make_initializable_iterator()
el = it.get_next()

#See how the pipeline handled the 4 sentences
sess.run(it.initializer)
print(sess.run(el))
print(sess.run(el))
print(sess.run(el))
print(sess.run(el))
