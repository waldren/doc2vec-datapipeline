import EmbeddingPipeline as ep
import numpy as np
import os
import tensorflow as tf
print(tf.__version__)

pl = ep.EmbeddingPipeline(stop_words=["the", "is"])
text = np.array(["My first sentence", "They're is wrong sentence", "The 3rd sentence is", "Quite long sentence for the python script to process, let's hope it can be processed okay"])
labels =np.array(["ONE", "TWO", "THERE", "FOUR"])

print(pl.build_dictionary(text))

sess = tf.Session()
ds = sess.run(pl.generate_dataset(text, labels))

it = ds.make_initializable_iterator()
el = it.get_next()

sess.run(it.initializer)
print(sess.run(el))
print(sess.run(el))
print(sess.run(el))
print(sess.run(el))
