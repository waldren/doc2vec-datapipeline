# doc2vec-datapipeline
Working on converting a doc2vec feed-dict based implementation to one that uses tf.data to boost performance. Original implmentation did not max out even one core of a 32 core CPU nor 1 of two 1080Ti GPUs.

## Original Code 
from https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/07_Sentiment_Analysis_With_Doc2Vec

The MIT License (MIT)

Copyright (c) 2016 Nick McClure
Copyright (c) 2016 Packt Publishing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
