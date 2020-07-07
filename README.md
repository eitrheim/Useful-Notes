# Useful-Notes

### Command Line Promps
Update all Python packages with pip  
 - `pip list --outdated --format=freeze | grep -v '^-e' | cut -d = -f 1 | xargs -n1 pip install -U`    
or    
 - `pip freeze --local | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U`    

Install package without installing the dependencies
 - `pip install -U --no-deps mypackage`   
 
`ssh USER@hadoop.rcc.uchicago.edu` login to RCC, then enter password  
`ls` list all directories  
`mkdir newdir` makes new directory named newdir  
`rm -rf newdir`	deletes directory named newdir  
`clear`	clears screen  
`cd newdir`		goes into newdir  
`cd`	goes to home  
`pwd`	to see which file you're in  
`rm file`	to remove 'file'  
`ssh user@host`	connect to host as user  
`ctrl+c` halt current command  
`crtl+z` stop current command  
`ctrl+d` log out of current session  
`ctrl+w` erases one word in current line  
`ctrl+u` erases whole line  
`!!` repeat last command  
`exit` log out of current session  
`wget <url>` download file directly  
`mv orignalFile name newFileName` to change file name  
`tar zxvf instacart.tar.gz` unzip file  
`hadoop fs -put /home/$USER/data/instacart /user/$USER/instacart` to put it to hadoop

### Classes
 - [Harvard Introduction to Computer Science](https://online-learning.harvard.edu/course/cs50-introduction-computer-science?category[]=3&sort_by=date_added)
 - [Notes that accompany the Stanford CS class: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

### Tips, Tutorials, How-tos, Visualizations
 - Plotting and Graphing
     - [Top 5 tricks to make plots look better](https://medium.com/@andykashyap/top-5-tricks-to-make-plots-look-better-9f6e687c1e08) 
     - [Plotting with categorical data with Seaborn](https://seaborn.pydata.org/tutorial/categorical.html?highlight=seaborn%20bar) 
     - [Sankey Diagram in Python](https://plot.ly/python/sankey-diagram/)    
 - Best Practices
     - [The Best of the Best Practices (BOBP) Guide for Python](https://gist.github.com/sloria/7001839)
     - [Best practices for file naming](https://library.stanford.edu/research/data-management-services/data-best-practices/best-practices-file-naming)
 - Principal Component Analysis
     - [Eigenvectors and Eigenvalues explained](http://setosa.io/ev/eigenvectors-and-eigenvalues/)    
     - [Principal Component Analysis explained](http://setosa.io/ev/principal-component-analysis/)
 - Clustering
     - [In Depth: k-Means Clustering](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html)  
 - Gradient Descent Optimisation
    - [10 Gradient Descent Optimisation Algorithms](https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9)
    - [Momentum gradient descent](https://distill.pub/2017/momentum/)
 - Neural Network/Deep Learning Overview and Training
    - [Setting hyperparameters for Neural Networks](http://cs231n.github.io/neural-networks-3/#summary)
    - [Training Deep Learning models entirely in your browser](https://cs.stanford.edu/people/karpathy/convnetjs/)
 - Convolutional Neural Networks and Images
    - [Feature Visualization](https://distill.pub/2017/feature-visualization/) how neural networks build up their understanding of images
    - [Combine the content of one image with the style of another image](https://github.com/jcjohnson/neural-style/) and another [Photo Style Transfer](https://github.com/luanfujun/deep-photo-styletransfer) and [another example](https://github.com/jcjohnson/neural-style/
    - [CNN example with MNIST dataset](https://ml4a.github.io/ml4a/looking_inside_neural_nets/)
    - [Convolution animations](https://github.com/vdumoulin/conv_arithmetic)
    - [Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-2/)
    - [What is wrong with Convolutional neural networks?](https://towardsdatascience.com/what-is-wrong-with-convolutional-neural-networks-75c2ba8fbd6f)
    - [Convolutional Neural Networks](http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/)
    - [Understanding layers in NN](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) from google AI blog
    - Pretrained Image Classifier [here](http://demo.caffe.berkeleyvision.org/) and [here](http://places2.csail.mit.edu/demo.html)
    - [DeepFace: Closing the Gap to Human-Level Performance in Face Verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf)
- Natural Language Processing
    - [Word Embeddings Projected](https://projector.tensorflow.org/)
    - [Word Embeddings](https://www.tensorflow.org/tutorials/text/word_embeddings)
    - [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
    - [Word2Vec For Phrases — Learning Embeddings For More Than One Word](https://towardsdatascience.com/word2vec-for-phrases-learning-embeddings-for-more-than-one-word-727b6cf723cf)
    - [Learning Word Embeddings](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html)
    - [Word2Vec For Phrases](https://towardsdatascience.com/word2vec-for-phrases-learning-embeddings-for-more-than-one-word-727b6cf723cf)
    - [Word embeddings in tensorflow](https://www.tensorflow.org/tutorials/text/word_embeddings)
- Recommender Systems
    - [5 Types of Recommender Systems](https://www.datasciencecentral.com/profiles/blogs/5-types-of-recommenders)
    - [How Shopify Uses Recommender Systems to Empower Entrepreneurs](https://medium.com/data-shopify/how-shopify-uses-recommender-systems-to-empower-entrepreneurs-99553b407944)
    - [Various Implementations of Collaborative Filtering](https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0)
    - [Collaborative filtering for movie recommendations](http://ampcamp.berkeley.edu/big-data-mini-course/movie-recommendation-with-mllib.html)
    - [Deep learning for recommender systems](https://ebaytech.berlin/deep-learning-for-recommender-systems-48c786a20e1a)
    - [Association Rules and the Apriori Algorithm](https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html)
 

### Topics Explained or Visualized
 - ![neural network architectures cheatsheet](https://www.asimovinstitute.org/wp-content/uploads/2019/04/NeuralNetworkZoo20042019.png)
 
### Textbooks
 - [R for Data Science textbook](https://r4ds.had.co.nz)
 - [Intro to Stats](http://onlinestatbook.com/2/index.html)
 - [MIT Deep Learning](http://www.deeplearningbook.org/)

### Datasets
 - [Pre-trained models: Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) for transfer learning
 - [Chicago Data](https://data.cityofchicago.org)    
 - [India Data](https://data.gov.in/catalogs)
 - [Airline Dataset](https://www.stat.purdue.edu/~sguha/rhipe/doc/html/airline.html)    
 - [List of Public Data Sources Fit for Machine Learning](https://blog.bigml.com/list-of-public-data-sources-fit-for-machine-learning/)    
 - [Million Song Dataset](http://millionsongdataset.com/pages/getting-dataset/)    
 - [Classification datasets](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)
 - 2013-2016 Cleaned & Parsed  [10-K Filings with the SEC](https://data.world/jumpyaf/2013-2016-cleaned-parsed-10-k-filings-with-the-sec?utm_campaign=data_digest&utm_source=email&utm_medium=email&utm_content=190808&_hsenc=p2ANqtz-_PS-XjGDrizhTbshW6iqJk29RYnUXcCFmqA5YFeY3sDIxCgWMAw6EUs3ecGV5mPKaRzsGojQxdK83sO7nE3swe9OAA1A&_hsmi=75508835)    
 - [Stanford Large Network Dataset Collection](http://snap.stanford.edu/data/index.html)    
 - [Google Dataset Search](https://www.blog.google/products/search/making-it-easier-discover-datasets/)
 - [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) with 60k train and 10k test images
 - [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) Similar to MNIST but used different fonts
 - [The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) 10 classes with 6k images per class
 - Large Scale Visual Recognition Challenge [2012](http://www.image-net.org/challenges/LSVRC/2012/index) and [2014](http://www.image-net.org/challenges/LSVRC/2014/)
 
### Spatial Datasets/Sources
#### Plotting in R
 - [Useful resources for tmap](https://github.com/mtennekes/tmap/#reference)   
 - [Code for tmap: Thematic Maps in R](https://cran.r-project.org/web/packages/tmap/vignettes/tmap-JSS-code.html)    
 - [Making maps with R](https://geocompr.robinlovelace.net/adv-map.html)    
 
#### Learning
 - [GeoDa Data Portal](https://geodacenter.github.io/data-and-lab/) mostly smaller sample datasets for use in learning spatial analysis
 - [spData R package](https://github.com/Nowosad/spData) accompanies Geocomputation with R
 - [geodaData R package](https://github.com/spatialanalysis/geodaData) currently in development, but includes all the packages in the GeoDa tutorials
 - [Tidycensus R package](https://walkerke.github.io/tidycensus) useful for getting US boundaries, variables

#### Social Science/Planning
 - [OpenStreetMap](https://datahub.cmap.illinois.gov/) can use API or osmdata R package to get data CMAP Data Hub: Chicago Metropolitan Agency for Planning (36 datasets, including community areas)
 - [NYU Spatial Data Repository](https://geo.nyu.edu/) lots of data and links to other institutions
 - [ICPSR data portal](https://www.icpsr.umich.edu/icpsrweb/) data archive for the social sciences

#### Land Use/Ecological
 - [USGS Data Products](https://www.usgs.gov/products/data-and-tools/gis-data)
 - [NEON Ecological data](https://data.neonscience.org/home)

#### Health/Demographics
 - [CDC GIS Data](https://www.cdc.gov/gis/geo-spatial-data.html) also has [ArcGIS training](https://www.cdc.gov/dhdsp/maps/gisx/training/index.html)    
 - [Demographic and Health Survey / USAID Spatial Data Repository](http://spatialdata.dhsprogram.com/home/)

#### Historical
 - [Library of Congress](https://www.loc.gov/maps/)    
 - [UChicago Maps](https://www.lib.uchicago.edu/collex/?view=collections&subject=Maps) and [librarians can help make into shapefiles](http://guides.lib.uchicago.edu/maps)

#### Others
 - [GeoDa Datasets on Github](https://github.com/spatialanalysis/geodaData). In R:    
   ```base
   install.packages('remotes')    
   remotes::install_github("spatialanalysis/geodaData")    
   library(geodaData)    
   data("chicago_comm")   
   ```
 - [Search on StackExchange here](https://gis.stackexchange.com/questions/8929/open-access-repository-of-general-gis-spatial-data) and [here](https://gis.stackexchange.com/questions/495/seeking-administrative-boundaries-for-various-countries)

#### To put in a section
 - [Graph Database example to find insurance fraud](https://neo4j.com/blog/insurance-fraud-detection-graph-database/)
 - [A Practical Introduction to Blockchain with Python](http://adilmoujahid.com/posts/2018/03/intro-blockchain-bitcoin-python/)
 - [Differences between Hive Internal and External Tables](https://blogs.msdn.microsoft.com/cindygross/2013/02/05/hdinsight-hive-internal-and-external-tables-intro/)
 - [Predicting Breast Cancer Using Apache Spark Machine Learning Logistic Regression](https://mapr.com/blog/predicting-breast-cancer-using-apache-spark-machine-learning-logistic-regression/)
 - [How to build a simple artificial neural network](https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go/)
 - [Implementation of a majority voting EnsembleVoteClassifier for classification](https://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/)
 - [An intuitive approach to Backpropagation](https://medium.com/spidernitt/breaking-down-neural-networks-an-intuitive-approach-to-backpropagation-3b2ff958794c)
 - [Priming neural networks with an appropriate initializer](https://becominghuman.ai/priming-neural-networks-with-an-appropriate-initializer-7b163990ead)
 - [How transferable are features in deep neural networks?](https://arxiv.org/pdf/1411.1792.pdf)
 - [A Gentle Introduction to Transfer Learning for Deep Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)
 - [Convolutions and Backpropagations](https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c)
 - [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
 - [ResNet, AlexNet, VGGNet, Inception: Understanding various architectures of Convolutional Networks](https://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/)
 - [AlphaGo Case Study](https://deepmind.com/research/case-studies/alphago-the-story-so-far)
 - [Human pose estimation](http://www.cs.cmu.edu/~vramakri/poseMachines.html)
 - [Paper on how deep neural networks are more difficult to train](https://arxiv.org/pdf/1512.03385.pdf)
 - [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)
 - [Transfer Learning](http://cs231n.github.io/transfer-learning/) stanford notes
 - [Chris Olah Blog on ML/DL topics](http://colah.github.io/)
 - [Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
 - [LSTM by Example using Tensorflow](https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537)
 - [Understanding, Deriving and Extending the LSTM](https://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html)
 - [ARIMA vs. LSTM slides](https://rpubs.com/zkajdan/316135)
 - [Auto-Generating Clickbait With Recurrent Neural Networks](https://larseidnes.com/2015/10/13/auto-generating-clickbait-with-recurrent-neural-networks/)
 - [Image Caption Generation](https://arxiv.org/pdf/1502.03044.pdf)
 - [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe's_quartet)    
 - [Visualizing how a Neural Network works](https://playground.tensorflow.org)    
 - [Bagging and Boosting](https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/)
 - [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
 - [Momentum for Gradient Decent](https://distill.pub/2017/momentum/)
 - [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)
 - [Blockchain summary (devopedia)](https://devopedia.org/blockchain)
 - [Brief overview of Blockchain](https://www.sec.gov/spotlight/investor-advisory-committee-2012/slides-nancy-liao-brief-intro-to-blockchain-iac-101217.pdf)
 - [MapReduce processes explained](https://www.oreilly.com/library/view/hadoop-the-definitive/9781491901687/ch04.html) but also explained [here](https://developer.yahoo.com/hadoop/tutorial/module4.html?guccounter=1%23dataflow)
 - [Apache YARN (Yet Another Resource Negotiator): Hadoop’s cluster resource management system](https://www.oreilly.com/library/view/hadoop-the-definitive/9781491901687/ch04.html)
 - [Wide VS Narrow Dependecies: Representation/DAG of what Spark analyzes to do optimizations](https://github.com/rohgar/scala-spark-4/wiki/Wide-vs-Narrow-Dependencies)
 - [Simple guide to confusion matrix terminology](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
 - [Sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
 - [Understanding ROC curves](http://www.navan.name/roc/)
 - [another with a detailed explaination](http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html)
 - [Hierarchical softmax](https://www.quora.com/What-is-hierarchical-softmax)
 - [Kernel Density Estimation](https://mathisonian.github.io/kde/)  
 - [OpenFace: A general-purpose face recognition library](http://cmusatyalab.github.io/openface/)
