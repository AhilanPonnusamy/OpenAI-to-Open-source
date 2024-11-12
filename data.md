Chapter 1
Introduction
Consider waking up one winter morning in Finland and looking outside the window (see
Figure 1.1). It seems to become a nice sunny day which is ideal for a ski trip. To choose the
right gear (clothing, wax) it is vital to have some idea for the maximum daytime temperature
which is typically reached around early afternoon. If we expect a maximum daytime
temperature of around plus 5 degrees, we might not put on the extra warm jacket but rather
take only some extra shirt for change.
Figure 1.1: Looking outside the window during the morning of a winter day in Finland.
We can use ML to learn a predictor for the maximum daytime temperature for the speci c
day depicted in Figure 1.1. The prediction shall be based solely on the minimum temperature
observed in the morning of that day. ML methods can learn a predictor in a data-driven
fashion using historic weather observations provided by the Finnish Meteorological Institute
(FMI). We can download the recordings of minimum and maximum daytime temperature
for the most recent days and denote the resulting dataset by
D =
 
z(1); : : : ; z(m) 
: (1.1)
15
Each data point z(i) =
􀀀
x(i); y(i)
 
, for i = 1; : : : ;m, represents some previous day for which
the minimum and maximum daytime temperature x(i) and y(i) has been recorded. We depict
the data (1.1) in Figure 1.2. Each dot in Figure 1.2 represents a speci c day with minimum
temperature x and maximum temperature y.
x
y
Figure 1.2: Each dot represents a speci c day that is characterized by its minimum daytime
temperature x as feature and its maximum daytime temperature y as label. These
temperatures are measured at some FMI weather station.
ML methods learn a hypothesis h(x), that reads in the minimum temperature x and
delivers a prediction (forecast or approximation) ^y = h(x) for the maximum daytime temperature
y. Every practical ML method uses a particular hypothesis space out of which the
hypothesis h is chosen. This hypothesis space of candidates for the hypothesis map is an
important design choice and might be based on domain knowledge.
In what follows, we illustrate how to use domain knowledge to motivate a choice for the
hypothesis space. Let us assume that the minimum and maximum daytime temperature of
an arbitrary day are approximately related via
y   w1x + w0 with some weights w1 2 R+;w0 2 R: (1.2)
The assumption (1.2) re
ects the intuition (domain knowledge) that the maximum daytime
temperature y should be higher for days with a higher minimum daytime temperature x. The
assumption (1.2) contains two weights w1 and w0. These weights are tuning parameters that
allow for some 
exibility in our assumption. We require the weight w1 to be non-negative
16
but otherwise leave these weights unspeci ed for the time being. The main subject of this
book are ML methods that can be used to learn suitable values for the weights w1 and w0
in a data-driven fashion.
Before we detail how ML can be used to  nd or learn good values for the weights w0
in w1 in (1.2) let us interpret them. The weight w1 in (1.2) can be interpreted as the
relative increase in the maximum daytime temperature for an increased minimum daytime
temperature. Consider an earlier day with recorded maximum daytime temperature of 10
degrees and minimum daytime temperature of 0 degrees. The assumption (1.2) then means
that the maximum daytime temperature for another day with minimum temperature of +1
degrees would be 10 + w1 degrees. The second weight w0 in our assumption (1.2) can be
interpreted as the maximum daytime temperature that we anticipate for a day with minimum
daytime temperature equal to 0.
Given the assumption (1.2), it seems reasonable to restrict the ML method to only
consider linear maps
h(x) := w1x + w0 with some weights w1 2 R+;w0 2 R: (1.3)
Since we require w1   0, the map (1.3) is monotonically increasing with respect to the
argument x. Therefore, the prediction h(x) for the maximum daytime temperature becomes
higher with higher minimum daytime temperature x.
The expression (1.3) de nes a whole ensemble of hypothesis maps. Each individual map
corresponding to a particular choice for w1   0 and w0. We refer to such an ensemble of
potential predictor maps as the model or hypothesis space that is used by a ML method.
We say that the map (1.3) is parametrized by the vector w =
􀀀
w1;w0
 T
and indicate
this by writing h(w). For a given parameter vector w =
􀀀
w1;w0
 T
, we obtain the map
h(w)(x) = w1x + w0. Figure 1.3 depicts three maps h(w) obtained for three di erent choices
for the parameters w.
ML would be trivial if there is only one single hypothesis. Having only a single hypothesis
means that there is no need to try out di erent hypotheses to  nd the best one. To enable
ML, we need to choose between a whole space of di erent hypotheses. ML methods are
computationally e cient methods to choose (learn) a good hypothesis out of (typically very
large) hypothesis spaces. The hypothesis space constituted by the maps (1.3) for di erent
weights is uncountably in nite.
To  nd, or learn, a good hypothesis out of the in nite set (1.3), we need to somehow
assess the quality of a particular hypothesis map. ML methods use a loss function for this
17
􀀀3 􀀀2 􀀀1 1 2 3
􀀀3
􀀀2
􀀀1
1
2
3
feature x
h(w)(x)
Figure 1.3: Three hypothesis maps of the form (1.3).
purpose. A loss function is used to quantify the di erence between the actual data and the
predictions obtained from a hypothesis map (see Figure 1.4). One widely-used example of
a loss function is the squared error loss (y 􀀀 h(x))2. Using this loss function, ML methods
learn a hypothesis map out of the model (1.3) by tuning w1;w0 to minimize the average loss
(1=m)
Xm
i=1
􀀀
y(i) 􀀀 h
􀀀
x(i)  2
:
The above weather prediction is prototypical for many other ML applications. Figure
1 illustrates the typical work
ow of a ML method. Starting from some initial guess, ML
methods repeatedly improve their current hypothesis based on (new) observed data.
Using the current hypothesis, ML methods make predictions or forecasts about future
observations. The discrepancy between the predictions and the actual observations, as measured
using some loss function, is used to improve the hypothesis. Learning happens during
improving the current hypothesis based on the discrepancy between its predictions and the
actual observations.
ML methods must start with some initial guess or choice for a good hypothesis. This
initial guess can be based on some prior knowledge or domain expertise [96]. While the
initial guess for a hypothesis might not be made explicit in some ML methods, each method
must use such an initial guess. In our weather prediction application discussed above, we
used the linear model (1.2) as the initial hypothesis.
18
x
y
Figure 1.4: Each dot represents a speci c days that is characterized by its minimum daytime
temperature x and its maximum daytime temperature y. We also depict a straight line
representing a linear predictor map. A main principle of ML methods is to learn a predictor
(or hypothesis) map with minimum discrepancy between predictor map and data points.
Di erent ML methods use di erent types of predictor maps (hypothesis space) and loss
functions to quantify the discrepancy between hypothesis and actual data points.
1.1 Relation to Other Fields
ML builds on concepts from several other scienti c  elds. Conversely, ML provides important
tools for many other scienti c  elds.
1.1.1 Linear Algebra
Modern ML methods are computationally e cient methods to  t high-dimensional models
to large amounts of data. The models underlying state-of-the-art ML methods can contain
billions of tunable or learnable parameters. To make ML methods computationally e cient
we need to use suitable representations for data and models.
Maybe the most widely used mathematical structure to represent data is the Euclidean
space Rn with some dimension n 2 N [119]. The rich algebraic and geometric structure of
Rn allows us to design of ML algorithms that can process vast amounts of data to quickly
update a model (parameters). Figure 1.5 depicts the Euclidean space Rn for n = 2, which
is used to construct scatterplots.
The scatterplot in Figure 1.2 depicts data points (representing individual days) as vectors
19
􀀀3 􀀀2 􀀀1 1 2 3
􀀀3
􀀀2
􀀀1
1
2
3
z1
z2
z =
􀀀
z1; z2
 T
z0 =
􀀀
z01
; z02
 T
Figure 1.5: The Euclidean space R2 is constituted by all vectors (or points) z =
􀀀
z1; z2
 T
(with z1; z2 2 R) together with the inner product zT z0 = z1z01
+ z2z02
.
in the Euclidean space R2. For a given data point, we obtain its associated vector z =
(x; y)T in R2 by stacking the minimum daytime temperature x and the maximum daytime
temperature y into the vector z of length two.
We can use the Euclidean space Rn not only to represent data points but also to represent
models for these data points. One such class of models is obtained by linear maps on Rn.
Figure 1.3 depicts some examples for such linear maps. We can then use the geometric
structure of Rn, de ned by the Euclidean norm, to search for the best model. As an example,
we could search for the linear model, represented by a straight line, such that the average
(Euclidean) distance to the data points in Figure 1.2 is as small as possible (see Figure 1.4).
The properties of linear structures are studied within linear algebra [135]. Some important
ML methods, such as linear classi er (see Section 3.1) or PCA (see Section 9.2) are direct
applications of methods from linear algebra.
1.1.2 Optimization
A main design principle for ML methods is the formulation of ML problems as optimization
problems [133]. The weather prediction problem above can be formulated as the problem of
optimizing (minimizing) the prediction error for the maximum daytime temperature. Many
ML methods are obtained by straightforward applications of optimization methods to the
optimization problem arising from a ML problem (or application).
The statistical and computational properties of such ML methods can be studied using
20
optimization
variable
objective
(a)
hypothesis
loss
(b)
Figure 1.6: (a) A simple optimization problem consists of  nding the values of an optimization
variable that results in the minimum objective value. (b) ML methods learn ( nd) a
hypothesis by minimizing a (average) loss. This average loss is a noisy version of the ultimate
objective. This ultimate objective function is often de ned as an expectation whose
underlying probability distribution is unknown (see Section 2.3.4).
tools from the theory of optimization. What sets the optimization problems in ML apart from
\plain vanilla" optimization problems (see Figure 1.6-(a)) is that we rarely have perfect access
to the objective function to be minimized. ML methods learn a hypothesis by minimizing
a noisy (or even incomplete) version (see Figure 1.6-(b)) of the ultimate objective function.
The ultimate objective function for ML methods is often de ned using an expectation over
an unknown probability distribution for data points. Chapter 4 discusses methods that are
based on estimating the objective function by empirical averages that are computed over a
set of data points (which serve as a training set).
1.1.3 Theoretical Computer Science
Practical ML methods form a speci c subclass of computing systems. Indeed, ML methods
apply a sequence of computational operations to input data. The result of these computational
operations are the predictions delivered to the user of the ML method. The interpretation
of ML as computational systems allows to use tools from theoretical computer science
to study the feasibility and intrinsic di culty of ML problems. Even if a ML problem can be
solved in theoretical sense, every practical ML method must  t the available computational
infrastructure [113, 142].
The available computational resources, such as processor time, memory and communication
bandwidth, can vary signi cantly between di erent infrastructures. One example
for such a computational infrastructure is a single desktop computer. Another example
21
for a computational infrastructure is a cloud computing service which distributes data and
computation over large networks of physical computers [95].
The focus of this book is on ML methods that can be understood as numerical optimization
algorithms (see Chapter 4 and 5). Most of these ML methods amount to (a large number
of) matrix operations such as matrix multiplication or matrix inversion [47]. Numerical linear
algebra provides a vast algorithmic toolbox for the design of such ML methods [135, 134].
The recent success of ML methods in several application domains might be attributed to
their e cient use of matrices to represent data and models. Using this representation allows
us to implement the resulting ML methods using highly e cient hard- and software
implementations for numerical linear algebra [49].
1.1.4 Information Theory
Information theory studies the problem of communication via noisy channels [139, 127, 28,
43]. Figure 1.7 depicts the most simple communication problem that consists of an information
source that wishes communicate a message m over an imperfect (or noisy) channel to
a receiver. The receiver tries to reconstruct (or learn) the original message based solely on
the noisy channel output. Two main goals of information theory are (i) the characterization
of conditions that allow reliable, i.e., nearly error-free, communication and (ii) the design
of e cient transmitter (coding and modulation) and receiver (demodulation and decoding)
methods.
It turns out that many concepts from information theory are very useful for the analysis
and design of ML methods. As a point in case, Chapter 10 discusses the application of
information-theoretic concepts to the design of explainable ML methods. On a more fundamental
level, we can identify two core communication problems that arise within ML. These
communication problems correspond, respectively, to the inference (making a prediction)
and the learning (adjusting or improving the current hypothesis) step of a ML method (see
Figure 1).
We can an interpret the inference step of ML as the problem of decoding the true label of
a data point for which we only know its features. This communication problem is depicted
in Figure 1.7-(b). Here the message to be communicated is the true label of a random
data point. This data point is \communicated" over a channel that only passes through its
features. The inference step within a ML method then tries to decode the original message
(true label) from the channel output (features) resulting in the predicted label. A recent line
of work used this communication problem to study deep learning methods [139].
22
information
source
transmitter noisy
channel receiver
data channel ML
source
(b)
? channel ML
(c)
m
(a)
^m
(X; y) X by
h  D ^h
Figure 1.7: (a) A basic communication system involves an information source that emits
a message m. The message is processed by some transmitter and passed through a noisy
channel. The receiver tries to recover the original message m by computing the decoded
message ^m. (b) The inference step of ML (see Figure 1) corresponds to a communication
problem with an information source emitting a data point with features x and label y. The
ML method receives the features x and, in an e ort to recover the true label y, computes
the predicted label ^y. (c) The learning or adaptation step of ML (see Figure 1) solves a
communication problem with some source that selects a true (but unknown) hypothesis h 
as the message. The message is passed through an abstract channel that outputs a set D of
labeled data points which are used as the training set by an ML method. The ML method
tries to decode the true hypothesis resulting in the learnt the hypothesis ^h.
23
A second core communication problem of ML corresponds to the problem of learning (or
adjusting) a hypothesis (see Figure 1.7-(c)). In this problem, the source selects some \true"
hypothesis as message. This message is then communicated to an abstract channel that
models the data generation process. The output of this abstract channel are data points
in a training set D (see Chapter 4). The learning step of a ML method, such as empirical
risk minimization (ERM) of Chapter 4, then amounts to the decoding of the message (true
hypothesis) based on the channel output (training set). There is signi cant line or research
that uses the communication problem in Figure 1.7-(c) to characterize the fundamental limits
of ML problems and methods such as the minimum required number of training data points
that makes learning feasible [154, 151, 122, 140, 71].
The relevance of information theoretic concepts and methods for ML is boosted by the
recent trend towards distributed or federated ML [93, 130, 124, 123]. We can interpret
federated learning (FL) applications as a speci c type of network communication problems
[43]. In particular, we can apply network coding techniques to the design and analysis of FL
methods [43].
1.1.5 Probability Theory and Statistics
Consider the data points z(1); : : : ; z(m) depicted in Figure 1.2. Each data point represents
some previous day that is characterized by its minimum and maximum daytime temperature
as measured at a speci c FMI weather observation station. It might be useful to interpret
these data points as realizations of independent and identically distributed (iid) RVs with
common (but typically unknown) probability distribution p(z). Figure 1.8 extends the scatterplot
in Figure 1.2 by adding a contour line of the underlying probability distribution p(z)
[10].
Probability theory o ers principled methods for estimating a probability distribution
from a set of data points (see Section 3.12). Let us assume we know (an estimate of) the
(joint) probability distribution p(z) of features and label of a data point z =
􀀀
x; y). A
principled approach to predict the label value of a data point with features x is based on
evaluating the conditional probability distribution p(y = ^yjx). The conditional probability
distribution p(^y = yjx) quanti es how likely it is that ^y is the actual label value of a data
point. We can evaluate the quantity p(^y = yjx) for any candidate value ^y as soon as we
know the features x of this data point.
Depending on the performance criterion or loss function, the optimal prediction ^y is either
given by the mode of p(^y = yjx) its mean or some other characteristic value. It is important to
24
note that this probabilistic approach not only provides a speci c prediction (point-estimate)
but an entire distribution p(^y = yjx) over possible predictions. This distribution allows to
construct con dence measures, such as the variance, that can be provided along with the
prediction.
Having a probabilistic model, in the form of a probability distribution p(z), for the
data arising in an ML application not only allows us to compute predictions for labels of
data points. We can also use p(z) to augment the available dataset by randomly drawing
(generating) new data points from p(z) (see Section 7.3). ML methods that rely on a
probabilistic model for data are sometimes referred to as generative methods. A recently
popularized class of generative methods, that uses models obtained from ANN, is known as
generative adversarial networks [50].
x
y
p(z)
Figure 1.8: Each dot represents a data point z =
􀀀
x; y
 
that is characterized by a numeric
feature x and a numeric label y. We also indicate a contour-line of a probability distribution
p(z) that could be used to interpret data points as realizations of iid RVs with common
probability distribution p(z).
1.1.6 Arti cial Intelligence
ML theory and methods are instrumental for the analysis and design of arti cial intelligence
(AI) [121]. An AI system, typically referred to as an agent, interacts with its environment
by executing (choosing between di erent) actions. These actions in
uence the environment
as well as the state of the AI agent. The behaviour of an AI system is determined by how
the perceptions made about the environment are used to form the next action.
25
From an engineering point of view, AI aims at optimizing behaviour to maximize a longterm
return. The optimization of behaviour is based solely on the perceptions made by the
agent. Let us consider some application domains where AI systems can be used:
• a forest  re management system: perceptions given by satellite images and local observations
using sensors or \crowd sensing" via some mobile application which allows
humans to notify about relevant events; actions amount to issuing warnings and bans
of open  re; return is the reduction of number of forest  res.
• a control unit for combustion engines: perceptions given by various measurements such
as temperature, fuel consistency; actions amount to varying fuel feed and timing and
the amount of recycled exhaust gas; return is measured in reduction of emissions.
• a severe weather warning service: perceptions given by weather radar; actions are
preventive measures taken by farmers or power grid operators; return is measured by
savings in damage costs (see https://www.munichre.com/)
• an automated bene t application system for the Finnish social insurance institute
(\Kela"): perceptions given by information about application and applicant; actions
are either to accept or to reject the application along with a justi cation for the decision;
return is measured in reduction of processing time (applicants tend to prefer
getting decisions quickly)
• a personal diet assistant: perceived environment is the food preferences of the app user
and their health condition; actions amount to personalized suggestions for healthy and
tasty food; return is the increase in well-being or the reduction in public spending for
health-care.
• the cleaning robot Rumba (see Figure 1.9) perceives its environment using di erent
sensors (distance sensors, on-board camera); actions amount to choosing di erent
moving directions (\north", \south", \east", \west"); return might be the amount of
cleaned 
oor area within a particular time period.
• personal health assistant: perceptions given by current health condition (blood values,
weight,. . . ), lifestyle (preferred food, exercise plan); actions amount to personalized
suggestions for changing lifestyle habits (less meat, more walking,. . . ); return is measured
via the level of well-being (or the reduction in public spending for health-care).
26
• a government-system for a country: perceived environment is constituted by current
economic and demographic indicators such as unemployment rate, budget de cit, age
distribution,. . . ; actions involve the design of tax and employment laws, public investment
in infrastructure, organization of health-care system; return might be determined
by the gross domestic product, the budget de cit or the gross national happiness (cf.
https://en.wikipedia.org/wiki/Gross_National_Happiness).
Figure 1.9: A cleaning robot chooses actions (moving directions) to maximize a long-term
reward measured by the amount of cleaned 
oor area per day.
ML methods are used on di erent levels by an AI agent. On a lower level, ML methods
help to extract the relevant information from raw data. ML methods are used to classify
images into di erent categories which are then used an input for higher level functions of
the AI agent.
ML methods are also used for higher level tasks of an AI agent. To behave optimally, an
agent is required to learn a good hypothesis for how its behaviour a ects its environment.
We can think of optimal behaviour as a consequent choice of actions that might be predicted
by ML methods.
What sets AI applications apart from more traditional ML application is that there is an
strong interaction between ML method and the data generation process. Indeed, AI agents
use the predictions of an ML method to select its next action which, in turn, in
uences the
environment which generates new data points. The ML sub eld of active learning studies
methods that can in
uence the data generation [26].
Another characteristic of AI applications is that they typically allow ML methods to
evaluate the quality of a hypothesis only in hindsight. Within a basic (supervised) ML
application it is possible for a ML method to try out many di erent hypotheses on the same
data point. These di erent hypotheses are then scored by their discrepancies with a known
27
correct predictions. In contrast to such passive ML applications, AI applications involve
data points for which it is infeasible to determine the correct predictions.
Let us illustrate the above di erences between ML and AI applications with the help
of a self-driving toy car. The toy-car is equipped with a small onboard computer, camera,
sensors and actors that allow to de ne the steering direction. Our goal is to program the
onboard computer such that it implements an AI agent that optimally steers the toy-car.
This AI application involves data points that represent the di erent (temporal) states of the
toy car during its ride. We use a ML method to predict the optimal steering direction for
the current state. The prediction for the optimal steering angle is obtained by a hypothesis
map that reads a snapshot form an on-board camera. Since these predictions are used to
actually steer the car, they in
uence the future data points (states) that will be obtained.
Note that we typically do not know the actual optimal steering direction for each possible
state of the car. It is infeasible to let the toy car roam around along any possible path and
then manually label each on-board camera snapshot with the optimal steering direction (see
Figure 1.12). The usefulness of a prediction can be measured only in an indirect fashion by
using some form of reward signal. Such a reward signal could be obtained from a distance
sensor that allows to determine if the toy car reduced the distance to a target location.
1.2 Flavours of Machine Learning
ML methods read in data points which are generated within some application domain. An
individual data point is characterized by various properties. We  nd it convenient to divide
the properties of data points into two groups: features and labels (see Chapter 2.1). Features
are properties that we measure or compute easily in an automated fashion. Labels are
properties that cannot be measured easily and often represent some higher level fact (or
quantity of interest) whose discovery often requires human experts.
Roughly speaking, ML aims at learning to predict (approximating or guessing) the label
of a data point based solely on the features of this data point. Formally, the prediction is
obtained as the function value of a hypothesis map whose input argument are the features of a
data point. Since any ML method must be implemented with  nite computational resources,
it can only consider a subset of all possible hypothesis maps. This subset is referred to as the
hypothesis space or model underlying a ML method. Based on how ML methods assess the
quality of di erent hypothesis maps we distinguish three main 
avours of ML: supervised,
unsupervised and reinforcement learning.
28
1.2.1 Supervised Learning
The main focus of this book is on supervised ML methods. These methods use a training
set that consists of labeled data points (for which we know the correct label values). We
refer to a data point as labeled if its label value is known. Labeled data points might be
obtained from human experts that annotate (\label") data points with their label values.
There are marketplaces for renting human labelling workforce [132]. Supervised ML searches
for a hypothesis that can imitate the human annotator and allows to predict the label solely
from the features of a data point.
Figure 1.10 illustrates the basic principle of supervised ML methods. These methods
learn a hypothesis with minimum discrepancy between its predictions and the true labels of
the data points in the training set. Loosely speaking, supervised ML  ts a curve (the graph
of the predictor map) to labeled data points in a training set. For the actual implementing
of this curve  tting we need a loss function that quanti es the  tting error. Supervised ML
method di er in their choice for a loss function to measure the discrepancy between predicted
label and true label of a data point.
While the principle behind supervised ML sounds trivial, the challenge of modern ML
applications is the sheer amount of data points and their complexity. ML methods must
process billions of data points with each single data points characterized by a potentially
vast number of features. Consider data points representing social network users, whose
features include all media that has been posted (videos, images, text). Besides the size and
complexity of datasets, another challenge for modern ML methods is that they must be able
to  t highly non-linear predictor maps. Deep learning methods address this challenge by
using a computationally convenient representation of non-linear maps via arti cial neural
networks [49].
1.2.2 Unsupervised Learning
Some ML methods do not require knowing the label value of any data point and are therefore
referred to as unsupervised ML methods. Unsupervised methods must rely solely on the
intrinsic structure of data points to learn a good hypothesis. Thus, unsupervised methods
do not need a teacher or domain expert who provides labels for data points (used to form a
training set). Chapters 8 and 9 discuss two large families of unsupervised methods, referred
to as clustering and feature learning methods.
Clustering methods group data points into few subsets or cluster. The data points within
29
(x(2); y(2))
(x(1); y(1))
feature x
predictor h(x)
label y
Figure 1.10: Supervised ML methods  t a curve to a set of data points (which serve as a
training set). The curve represents a hypothesis out of some hypothesis space or model. The
 tting error (or training error) is measured using a loss function. Di erent ML methods use
di erent combinations of model and loss function.
the same cluster should be more similar with each other than with data points outside the
cluster (see Figure 1.11). Feature learning methods determine numeric features such that
data points can be processed e ciently using these features. Two important applications of
feature learning are dimensionality reduction and data visualization.
1.2.3 Reinforcement Learning
In general, ML methods use a loss function to evaluate and compare di erent hypotheses.
The loss function assigns a (typically non-negative) loss value to a pair of a data point and
a hypothesis. ML methods search for a hypothesis, out of (typically large) hypothesis space,
that incurs minimum loss for any data point. Reinforcement learning (RL) studies applications
where the predictions obtained by a hypothesis in
uences the generation of future data
points. RL applications involve data points that represent the states of a programmable system
(an AI agent) at di erent time instants. The label of such a data point has the meaning
of an optimal action that the agent should take in a given state. Similar to unsupervised
ML, RL methods often must learn a hypothesis without having access to any labeled data
point.
In stark contrast to supervised and unsupervised ML methods, RL methods cannot evaluate
the loss function for di erent choices of a hypothesis. Consider a RL method that has
to predict the optimal steering angle of a car. Naturally, we can only evaluate the usefulness
speci c combination of predicted label (steering angle) and the current state of the car. It
is impossible to try out two di erent hypotheses at the same time as the car cannot follow
30
x(3)
x(4)
x(2)
x(1)
x(5)
x(6)
x(7)
x1
x2
Figure 1.11: Clustering methods learn to predict the cluster (or group) assignments of data
points based solely on their features. Chapter 8 discusses clustering methods that are unsupervised
in the sense of not requiring the knowledge of the true cluster assignment of any
data point.
two di erent steering angles (obtained by the two hypotheses) at the same time.
Mathematically speaking, RL methods can evaluate the loss function only point-wise
for the current hypothesis that has been used to obtain the most recent prediction. These
point-wise evaluations of the loss function are typically implemented by using some reward
signal [136]. Such a reward signal might be obtained from a sensing device and allows to
quantify the usefulness of the current hypothesis.
One important application domain for RL methods is autonomous driving (see Figure
1.12). Consider data points that represent individual time instants t = 0; 1; : : : during a
car ride. The features of the tth data point are the pixel intensities of an on-board camera
snapshot taken at time t. The label of this data point is the optimal steering direction at time
t to maximize the distance between the car and any obstacle. We could use a ML method to
learn hypothesis for predicting the optimal steering direction solely from the pixel intensities
in the on-board camera snapshot. The loss incurred by a particular hypothesis is determined
from the measurement of a distance sensor after the car moved along the predicted direction.
We can evaluate the loss only for the hypothesis that has actually been used to predict the
optimal steering direction. It is impossible to evaluate the loss for other predictions of the
optimal steering direction since the car already moved on.
31
Figure 1.12: Autonomous driving requires to predict the optimal steering direction (label)
based on an on-board camera snapshot (features) in each time instant. RL methods sequentially
adjust a hypothesis for predicting the steering direction from the snapshot. The
quality of the current hypothesis is evaluated by the measurement of a distance sensor (to
avoid collisions with obstacles).
1.3 Organization of this Book
Chapter 2 introduces the notions of data, a model and a loss function as the three main
components of ML. We will also highlight some of the computational and statistical aspects
that might guide the design choices arising for these three components. A guiding theme of
this book is the depiction of ML methods as combinations of speci c design choices for data
representation, the model and the loss function. Put di erently, we aim at mapping out the
vast landscape of ML methods in an abstract three-dimensional space spanned by the three
dimensions: data, model and loss.
Chapter 3 details how several well-known ML methods are obtained by speci c design
choices for data (representation), model and loss function. Examples range from basic linear
regression (see Section 3.1) via support vector machine (SVM) (see Section 3.7) to deep
reinforcement learning (see Section 3.14).
Chapter 4 discusses a principle approach to combine the three components within a
practical ML method. In particular, Chapter 4 explains how a simple probabilistic model
for data lends naturally to the principle of ERM. This principle translates the problem of
learning into an optimization problem. ML methods based on the ERM are therefore a
special class of optimization methods. The ERM principle can be interpreted as a precise
mathematical formulation of the \learning by trial and error" paradigm.
Chapter 5 discusses a family of iterative methods for solving the ERM problem introduced
in Chapter 4. These methods use the concept of a gradient to locally approximate
the objective function used in ERM. Gradient-based methods are widely used within deep
32
learning methods to learn useful weights for large ANN (see Section 3.11 and [49]).
The ERM principle of Chapter 4 requires a hypothesis to accurately predict the labels
of data points in a training set. However, the ultimate goal of ML is to learn a hypothesis
that delivers accurate predications for any data point and not only the training set. Chapter
6 discusses some basic validation techniques that allow to probe a hypothesis outside the
training set that has been used to learn (optimize) this hypothesis via ERM. Validation
techniques are instrumental for model selection, i.e., to choose the best model from a given
set of candidate models. Chapter 7 presents regularization techniques that aim at replacing
the training error of a candidate hypothesis with an estimate (or approximation) of its
average loss incurred for data points outside the training set.
The focus of Chapters 3 - 7 is on supervised ML methods that require a training set of
labeled data points. Chapters 8 and 9 are devoted to unsupervised ML methods which do
not require any labeled data. Chapter 8 discusses clustering methods that partition data
points into coherent groups which are referred to as clusters. Chapter 9 discusses methods
that automatically determine the most relevant characteristics (or features) of a data point.
This chapter also highlights the importance of using only the most relevant features of a data
point, and to avoid irrelevant features, to reduce computational complexity and improve the
accuracy of ML methods (such as those discussed in Chapter 3).
The success of ML methods becomes increasingly dependent on their explainability or
transparency for the user of the ML method [55, 92]. The explainability of a ML method
and its predictions typically depends on the background knowledge of the user which might
vary signi cantly. Chapter 10 discusses two di erent approaches to obtain personalized
explainable ML. These techniques use a feedback signal, which is provided for the data
points in a training set, to either compute personalized explanations for a given ML method
or to choose models that are intrinsically explainable to a speci c user.
Prerequisites. We assume familiarity with basic notions and concepts of linear algebra,
real analysis, and probability theory [135, 119]. For a brief review of those concepts, we
recommend [49, Chapter 2-4] and the references therein. A main goal of this book is to
develop the basic ideas and principles behind ML methods using a minimum of probability
theory. However, some rudimentary knowledge about the concept of expectations, probability
density function of a continuous (real-valued) RV and probability mass function of a
discrete RV is helpful [10, 51].
33
Chapter 2
Three Components of ML
model
loss data
Figure 2.1: ML methods  t a model to data via minimizing a loss function (see Figure 1).
We obtain a variety of ML methods from di erent design choices for the model, loss function
and data representation (see Chapter 3). A principled approach to combine these three
components is ERM (see Chapter 4).
This book portrays ML as combinations of three components as depicted in Figure 2.1.
These components are
• data as collections of individual data points that are characterized by features (see
Section 2.1.1) and labels (see Section 2.1.2)
• a model or hypothesis space that consists of computationally feasible hypothesis maps
from feature space to label space (see Section 2.2)
• a loss function (see Section 2.3) to measure the quality of a hypothesis map.
34
A ML problem involves speci c design choices for data points, its features and labels, the
hypothesis space and the loss function to measure the quality of a particular hypothesis.
Similar to ML problems (or applications), we can also characterize ML methods as combinations
of these three components. This chapter discusses in some depth each of the above
ML components and their combination within some widely-used ML methods [112].
We detail in Chapter 3 how some of the most popular ML methods, including linear
regression (see Section 3.1) as well as deep learning methods (see Section 3.11), are obtained
by speci c design choices for the three components. Chapter 4 will then introduce ERM as
a main principle for how to operationally combine the individual ML components. Within
the ERM principle, ML problems become optimization problems and ML methods become
optimization methods.
2.1 The Data
Data as Collections of Data points. Maybe the most important component of any
ML problem (and method) is data. We consider data as collections of individual data
points which are atomic units of \information containers". Data points can represent text
documents, signal samples of time series generated by sensors, entire time series generated
by collections of sensors, frames within a single video, random variables, videos within a
movie database, cows within a herd, trees within a forest, or forests within a collection of
forests. Mountain hikers might be interested in data points that represent di erent hiking
tours (see Figure 2.2).
Figure 2.2: A snapshot taken at the beginning of a mountain hike.
We use the concept of data points in a very abstract and therefore highly 
exible manner.
35
Data points can represent very di erent types of objects that arise in fundamentally di erent
application domains. For an image processing application it might be useful to de ne data
points as images. When developing a recommendation system we might de ne data points
to represent customers. In the development of new drugs we might use data points to
represent di erent diseases. Ultimately, the choice or de nition of data points is a design
choice. We might refer to the task of  nding a useful de nition of data points as \data point
engineering".
One practical requirement for a useful de nition of data points is that we should have
access to many of them. Many ML methods construct estimates for a quantity of interest
(such as a prediction or forecast) by averaging over a set of reference (or training) data
points. These estimates become more accurate for an increasing number of data points used
for computing the average. A key property of a dataset is the number m of individual
data points it contains. The number of data points within a dataset is also referred to as
the sample size. From a statistical point of view, the larger the sample size m the better.
However, there might be restrictions on computational resources (such as memory size) that
limit the maximum sample size m that can be processed.
For most applications, it is impossible to have full access to every single microscopic
property of a data point. Consider a data point that represents a vaccine. A full characterization
of such a data point would require to specify its chemical composition down to
level of molecules and atoms. Moreover, there are properties of a vaccine that depend on
the patient who received the vaccine.
We  nd it useful to distinguish between two di erent groups of properties of a data
point. The  rst group of properties is referred to as features and the second group of
properties is referred to as labels. Roughly speaking, features are low-level properties of a
data point that can be measured or computed easily in an automated fashion. In contract,
labels are high-level properties of a data points that represent some quantity of interest.
Determining the label value of a data point often requires human labour, e.g., a domain
expert who has to examine the data point. Some widely used synonyms for features are
\covariate",\explanatory variable", \independent variable", \input (variable)", \predictor
(variable)" or \regressor" [53, 31, 39]. Some widely used synonyms for the label of a data
point are "response variable", "output variable" or "target" [53, 31, 39].
We will discuss the concepts of features and labels in somewhat more detail in Sections
2.1.1 and 2.1.2. However, we would like to point out that the distinction between features
and labels is blurry. The same property of a data point might be used as a feature in one
36
application, while it might be used as a label in another application. Let us illustrate this
blurry distinction between features and labels using the problem of missing data.
Assume we have a list of data points each of which is characterized by several properties
that could be measured easily in principles (by sensors). These properties would be  rst
candidates for being used as features of the data points. However, some of these properties
are unknown (missing) for a small set of data points (e.g., due to broken sensors). We
could then de ne the properties which are missing for some data points as labels and try to
predict these labels using the remaining properties (which are known for all data points) as
features. The task of determining missing values of properties that could be measured easily
in principle is referred to as imputation [1].
Missing data might also arise in image processing applications. Consider data points
being images (snapshots) generated by a smartphone camera. Maybe the most intuitive
choice for the features of a (bitmap) image are the colour intensities for each pixel (see
Figure 2.5). Due to hardware failures some of the image pixels might be corrupted or (their
colour intensities) even completely missing. We could then try to use to learn to predict
the colour intensities of a pixel based on the colour intensities of the neighbouring pixels.
To this end, we might de ne new data points as small rectangular regions (patches) of the
image and use the colour intensity of the centre pixel (\target pixel") as the label of such a
patch.
Figure 2.3 illustrates two key properties of a dataset. The  rst property is the sample size
m, i.e., the number of individual data points that constitute the dataset. The second key
property of is the number n of features that are used to characterize an individual data point.
The behaviour of ML methods often depends crucially on the ratio m=n. The performance
of ML methods typically improves with increasing m=n. As a rule of thumb, we should use
datasets for which m=n   1. We will make the informal condition m=n   1 more precise
in Chapter 6.
2.1.1 Features
Similar to the choice (or de nition) of data points with an ML application, also the choice
of which properties to be used as their features is a design choice. In general, features
are (low-level) properties of a data point that can be computed or measured easily. This
is obviously a highly informal characterization since there is no universal criterion for the
di culty of computing of measuring a speci c property of data points. The task of choosing
which properties to use as features of data points might be the most challenging part in the
37
Year
2020
2020
2020
2020
2020
2020
m
1
1
1
1
1
1
d
2
3
4
5
6
7
Time
00:00
00:00
00:00
00:00
00:00
00:00
precp
0,4
1,6
0,1
1,9
0,6
4,1
snow
55
53
51
52
52
52
airtmp
2,5
0,8
-5,8
-13,5
-2,4
0,4
mintmp
-2
-0,8
-11,1
-19,1
-11,4
-2
maxtmp
4,5
4,6
-0,7
-4,6
-1
1,3
n
m
Figure 2.3: Two key properties of a dataset are the number (sample size) m of individual
data points that constitute the dataset and the number n of features used to characterize
individual data points. The behaviour of ML methods typically depends crucially on the
ratio m=n.
application of ML methods. Chapter 9 discusses feature learning methods that automate
(to some extend) the construction of good features.
In some application domains there is a rather natural choice for the features of a data
point. For data points representing audio recording (of a given duration) we might use the
signal amplitudes at regular sampling instants (e.g., using sampling frequency 44 kHz) as
features. For data points representing images it seems natural to use the colour (red, green
and blue) intensity levels of each pixel as a feature (see Figure 2.5).
The feature construction for images depicted in Figure 2.5 can be extended to other types
of data points as long as they can be visualized e ciently [41]. Audio recordings are typically
available a sequence of signal amplitudes at collected regularly at time instants t = 1; : : : ; n
with sampling frequency   44 kHz. From a signal processing perspective, it seems natural
to directly use the signal amplitudes as features, xj = aj for j = 1; : : : ; n. However, another
choice for the features would be the pixel RGB values of some visualization of the audio
recording.
Figure 2.4 depicts two possible visualizations of an audio signal. The  rst visualization
is obtained from a line plot of the signal amplitudes as a function of time t. Another
visualization of an audio recording is obtained from an intensity plot of its spectrogram[14,
90]. We can then use the pixel RGB intensities of these visualizations as the features for an
audio recording. Using this trick we can transform any ML method for image data to an
ML method for audio data. We can use the scatterplot of a data set to use ML methods for
38
Figure 2.4: Two visualizations of a data point that represents an audio recording. The left
 gure shows a line plot of the audio signal amplitudes. The right  gure shows a spectogram
of the audio recording.
image segmentation to cluster the dataset(see Chapter 8).
Many important ML application domains generate data points that are characterized
by several numeric features x1; : : : ; xn. We represent numeric features by real numbers
x1; : : : ; xn 2 R which might seem impractical. Indeed, digital computers cannot store a real
number exactly as this would require an in nite number of bits. However, numeric linear
algebra soft- and hardware allows to approximate real numbers with su cient accuracy. The
majority of ML methods discussed in this book assume that data points are characterized
by real-valued features. Section 9.3 discusses methods for constructing numeric features of
data points whose natural representation is non-numeric.
We assume that data points arising in a given ML application are characterized by the
same number n of individual features x1: : : : ; xn. It is convenient to stack the individual
features of a data point into a single feature vector
x =
􀀀
x1; : : : ; xn
 T
:
Each data point is then characterized by its feature vector x. Note that stacking the features
of a data point into a column vector x is pure convention. We could also arrange the features
as a row vector or even as a matrix, which might be even more natural for features obtained
by the pixels of an image (see Figure 2.5).
We refer to the set of possible feature vectors of data points arising in some ML appli-
39
snapshot z(i)
512   512 pixels
(red-green-blue bitmap)
feature vector x(i)
pixel 1
pixel 2
pixel 512   512
n
255
0
0
0
255
0
0
0
255
......
r[1]
g[1]
b[1]
r[2]
g[2]
b[2]
Figure 2.5: If the snapshot z(i) is stored as a 512 512 RGB bitmap, we could use as features
x(i) 2 Rn the red-, green- and blue component of each pixel in the snapshot. The length of
the feature vector would then be n = 3   512   512   786000.
cation as the feature space and denote it as X. The feature space is a design choice as it
depends on what properties of a data point we use as its features. This design choice should
take into account the statistical properties of the data as well as the available computational
infrastructure. If the computational infrastructure allows for e cient numerical linear
algebra, then using X = Rn might be a good choice.
The Euclidean space Rn is an example of a feature space with a rich geometric and
algebraic structure [119]. The algebraic structure of Rn is de ned by vector addition and
multiplication of vectors with scalars. The geometric structure of Rn is obtained by the
Euclidean norm as a measure for the distance between two elements of Rn. The algebraic
and geometric structure of Rn often enables an e cient search over Rn to  nd elements with
desired properties. Chapter 4.3 discusses examples of such search problems in the context
of learning an optimal hypothesis.
Modern information-technology, including smartphones or wearables, allows us to measure
a huge number of properties about data points in many application domains. Consider
a data point representing the book author \Alex Jung". Alex uses a smartphone to take
roughly  ve snapshots per day (sometimes more, e.g., during a mountain hike) resulting
in more than 1000 snapshots per year. Each snapshot contains around 106 pixels whose
greyscale levels we can use as features of the data point. This results in more than 109
40
features (per year!).
As indicated above, many important ML applications involve data points represented
by very long feature vectors. To process such high-dimensional data, modern ML methods
rely on concepts from high-dimensional statistics [18, 150]. One such concept from highdimensional
statistics is the notion of sparsity. Section 3.4 discusses methods that exploit
the tendency of high-dimensional data points, which are characterized by a large number n
of features, to concentrate near low-dimensional subspaces in the feature space [146].
At  rst sight it might seem that \the more features the better" since using more features
might convey more relevant information to achieve the overall goal. However, as we
discuss in Chapter 7, it can be detrimental for the performance of ML methods to use an
excessive amount of (irrelevant) features. Computationally, using too many features might
result in prohibitive computational resource requirements (such as processing time). Statistically,
each additional feature typically introduces an additional amount of noise (due to
measurement or modelling errors) which is harmful for the accuracy of the ML method.
It is di cult to give a precise and broadly applicable characterization of the maximum
number of features that should be used to characterize the data points. As a rule of thumb,
the number m of (labeled) data points used to train a ML method should be much larger
than the number n of numeric features (see Figure 2.3). The informal condition m=n   1
can be ensured by either collecting a su ciently large number m of data points or by using
a su ciently small number n of features. We next discuss implementations for each of these
two complementary approaches.
The acquisition of (labeled) data points might be costly, requiring human expert labour.
Instead of collecting more raw data, it might be more e cient to generate new arti cial (synthetic)
data via data augmentation techniques. Section 7.3 shows how intrinsic symmetries
in the data can be used to augment the raw data with synthetic data. As an example for
an intrinsic symmetry of data, consider data points being images. We assign each image the
label y = 1 if it shows a cat and y = 􀀀1 otherwise. For each image with known label we
can generate several augmented (additional) images with the same label. These additional
images might be obtained by simple image transformation such as rotations or re-scaling
(zoom-in or zoom-out) that do not change the depicted objects (the meaning of the image).
Chapter 7 shows that some basic regularization techniques can be interpreted as an implicit
form of data augmentation.
The informal condition m=n   1 can also be ensured by reducing the number n of
features used to characterize data points. In some applications, we might use some domain
41
knowledge to choose the most relevant features. For other applications, it might be di cult
to tell which quantities are the best choice for features. Chapter 9 discusses methods that
learn, based on some given dataset, to determine a small number of relevant features of data
points.
Beside the available computational infrastructure, also the statistical properties of datasets
must be taken into account for the choices of the feature space. The linear algebraic structure
of Rn allows us to e ciently represent and approximate datasets that are well aligned along
linear subspaces. Section 9.2 discusses a basic method to optimally approximate datasets by
linear subspaces of a given dimension. The geometric structure of Rn is also used in Chapter
8 to decompose a dataset into few groups or clusters that consist of similar data points.
Throughout this book we will mainly use the feature space Rn with dimension n being
the number of features of a data point. This feature space has proven useful in many ML
applications due to availability of e cient soft- and hardware for numerical linear algebra.
Moreover, the algebraic and geometric structure of Rn re
ect the intrinsic structure of data
arising in many important application domains. This should not be too surprising as the
Euclidean space has evolved as a useful mathematical abstraction of physical phenomena
[77].
In general there is no mathematically correct choice for which properties of a data point
to be used as its features. Most application domains allow for some design freedom in the
choice of features. Let us illustrate this design freedom with a personalized health-care
applications. This application involves data points that represent audio recordings with the
 xed duration of three seconds. These recordings are obtained via smartphone microphones
and used to detect coughing [7].
2.1.2 Labels
Besides its features, a data point might have a di erent kind of properties. These properties
represent a higher-level fact or quantity of interest that is associated with the data point. We
refer to such properties of a data point as its label (or \output" or \target") and typically
denote it by y (if it is a single number) or by y (if it is a vector of di erent label values,
such as in multi-label classi cation). We refer to the set of all possible label values of data
points arising in a ML application is the label space Y. In general, determining the label
of a data point is more di cult (to automate) compared to determining its features. Many
ML methods revolve around  nding e cient ways to predict (estimate or approximate) the
label of a data point based solely on its features.
42
The distinction of data point properties into labels and features is blurry. Roughly
speaking, labels are properties of data points that might only be determined with the help
of human experts. For data points representing humans we could de ne its label y as an
indicator if the person has 
u (y = 1) or not (y = 0). This label value can typically only be
determined by a physician. However, in another application we might have enough resources
to determine the 
u status of any person of interest and could use it as a feature that
characterizes a person.
Consider a data point that represents a hike, at the start of which the snapshot in Figure
2.2 has been taken. The features of this data point could be the red, green and blue (RGB)
intensities of each pixel in the snapshot in Figure 2.2. We stack these RGB values into a
vector x 2 Rn whose length n is three times the number of pixels in the image. The label y
associated with a data point (a hike) could be the expected hiking time to reach the mountain
in the snapshot. Alternatively, we could de ne the label y as the water temperature of the
lake that is depicted in the snapshot.
Numeric Labels - Regression. For a given ML application, the label space Y contains
all possible label values of data points. In general, the label space is not just a set of
di erent elements but also equipped (algebraic or geometric) structure. To obtain e cient
ML methods, we should exploit such structure. Maybe the most prominent example for
such a structured label space are the real numbers Y = R. This label space is useful for ML
applications involving data points with numeric labels that can be modelled by real numbers.
ML methods that aim at predicting a numeric label are referred to as regression methods.
Categorical Labels - Classi cation. Many important ML applications involve data
points whose label indicate the category or class to which data points belongs to. ML methods
that aim at predicting such categorical labels are referred to as classi cation methods.
Examples for classi cation problems include the diagnosis of tumours as benign or male -
cent, the classi cation of persons into age groups or detecting the current 
oor conditions (
\grass", \tiles" or \soil") for a mower robot.
The most simple type of a classi cation problems is a binary classi cation problem.
Within binary classi cation, each data point belongs to exactly one out of two di erent
classes. Thus, the label of a data point takes on values from a set that contains two di erent
elements such as f0; 1g or f􀀀1; 1g or f\shows cat"; \shows no cat"g.
We speak of a multi-class classi cation problem if data points belong to exactly one out
of more than two categories (e.g., image categories \no cat shown" vs. \one cat shown"
and \more than one cat shown"). If there are K di erent categories we might use the label
43
values f1; 2; : : : ;Kg.
There are also applications where data points can belong to several categories simultaneously.
For example, an image can be cat image and a dog image at the same time if it
contains a dog and a cat. Multi-label classi cation problems and methods use several labels
y1; y2; : : : ; for di erent categories to which a data point can belong to. The label yj represents
the jth category and its value is yj = 1 if the data point belongs to the j-th category
and yj = 0 if not.
Ordinal Labels. Ordinal label values are somewhat in between numeric and categorical
labels. Similar to categorical labels, ordinal labels take on values from a  nite set. Moreover,
similar to numeric labels, ordinal labels take on values from an ordered set. For an example
for such an ordered label space, consider data points representing rectangular areas of size 1
km by 1 km. The features x of such a data point can be obtained by stacking the RGB pixel
values of a satellite image depicting that area (see Figure 2.5). Beside the feature vector,
each rectangular area is characterized by a label y 2 f1; 2; 3g where
• y = 1 means that the area contains no trees.
• y = 2 means that the area is partially covered by trees.
• y = 3 means that the are is entirely covered by trees.
Thus we might say that label value y = 2 is \larger" than label value y = 1 and label value
y = 3 is \larger" than label value y = 2.
The distinction between regression and classi cation problems and methods is somewhat
blurry. Consider a binary classi cation problem based on data points whose label y takes
on values 􀀀1 or 1. We could turn this into a regression problem by using a new label y0
which is de ned as the con dence in the label y being equal to 1. On the other hand, given
a prediction ^ y0 for the numeric label y0 2 R we can obtain a prediction ^y for the binary label
y 2 f􀀀1; 1g by setting ^y := 1 if ^ y0   0 and ^y := 􀀀1 otherwise. A prominent example for this
link between regression and classi cation is logistic regression which is discussed in Section
3.6. Logistic regression is a binary classi cation method that uses the same model as linear
regression but a di erent loss function.
We refer to a data point as being labeled if, besides its features x, the value of its label
y is known. The acquisition of labeled data points typically involves human labour, such
as verifying if an image shows a cat. In other applications, acquiring labels might require
sending out a team of marine biologists to the Baltic sea [131], to run a particle physics
44
experiment at the European organization for nuclear research (CERN) [19], or to conduct
animal trials in pharmacology [44].
Let us also point out online market places for human labelling workforce [99]. These
market places, allow to upload data points, such as collections of images or audio recordings,
and then o er an hourly rate to humans that label the data points. This labeling work might
amount to marking images that show a cat.
Many applications involve data points whose features can be determined easily, but whose
labels are known for few data points only. Labeled data is a scarce resource. Some of the most
successful ML methods have been devised in application domains where label information
can be acquired easily [56]. ML methods for speech recognition and machine translation can
make use of massive labeled datasets that are freely available [78].
In the extreme case, we do not know the label of any single data point. Even in the
absence of any labeled data, ML methods can be useful for extracting relevant information
from features only. We refer to ML methods which do not require any labeled data points
as \unsupervised" ML methods. We discuss some of the most important unsupervised ML
methods in Chapter 8 and Chapter 9).
ML methods learn (or search for) a \good" predictor h : X ! Y which takes the features
x 2 X of a data point as its input and outputs a predicted label (or output, or target)
^y = h(x) 2 Y. A good predictor should be such that ^y   y, i.e., the predicted label ^y is
close (with small error ^y 􀀀 y) to the true underlying label y.
2.1.3 Scatterplot
Consider data points characterized by a single numeric feature x and single numeric label
y. To gain more insight into the relation between the features and label of a data point, it
can be instructive to generate a scatterplot as shown in Figure 1.2. A scatterplot depicts
the data points z(i) = (x(i); y(i)) in a two-dimensional plane with the axes representing the
values of feature x and label y.
The visual inspection of a scatterplot might suggest potential relationships between feature
x (minimum daytime temperature) and label y (maximum daytime temperature). From
Figure 1.2, it seems that there might be a relation between feature x and label y since data
points with larger x tend to have larger y. This makes sense since having a larger minimum
daytime temperature typically implies also a larger maximum daytime temperature.
To construct a scatterplot for data points with more than two features we can use feature
learning methods (see Chapter 9). These methods transform high-dimensional data points,
45
having billions of raw features, to three or two new features. These new features can then
be used as the coordinates of the data points in a scatterplot.
2.1.4 Probabilistic Models for Data
A powerful idea in ML is to interpret each data points as the realization of a RV. For ease
of exposition let us consider data points that are characterized by a single feature x. The
following concepts can be extended easily to data points characterized by a feature vector x
and a label y.
One of the most basic examples of a probabilistic model for data points in ML is the
iid assumption. This assumption interprets data points x(1); : : : ; x(m) as realizations of statistically
independent RVs with the same probability distribution p(x). It might not be
immediately clear why it is a good idea to interpret data points as realizations of RVs with
the common probability distribution p(x). However, this interpretation allows us to use the
properties of the probability distribution to characterize overall properties of entire datasets,
i.e., large collections of data points.
The probability distribution p(x) underlying the data points within the i.i.d. assumption
is either known (based on some domain expertise) or estimated from data. It is often enough
to estimate only some parameters of the distribution p(x). Section 3.12 discusses a principled
approach to estimate the parameters of a probability distribution from a given set of data
points. This approach is sometimes referred to as maximum likelihood and aims at  nding
(parameter of) a probability distribution p(x) such that the probability (density) of observing
the given data points is maximized [85, 76, 10].
Two of the most basic and widely used parameters of a probability distribution p(x) are
the expected value or mean [12]
 x = Efxg :=
Z
x0
x0p(x0)dx0
and the variance
 2x
:= E
 􀀀
x 􀀀 Efxg
 2 
:
46
These parameters can be estimated using the sample mean (average) and sample variance,
^ x := (1=m)
Xm
i=1
x(i) , and
b  2x
:= (1=m)
Xm
i=1
􀀀
x(i) 􀀀 ^ x
 2
: (2.1)
The sample mean and sample variance (2.1) are the maximum likelihood estimators for the
mean and variance of a normal (Gaussian) distribution p(x) (see [13, Chapter 2.3.4]).
Most of the ML methods discussed in this book are motivated by an i.i.d. assumption. It is
important to note that this i.i.d. assumption is only a modelling assumption (or hypothesis).
There is no means to verify if an arbitrary set of data points are \exactly" realizations of
iid RVs. There are principled statistical methods (hypothesis tests) that allow to verify if a
given set of data point can be well approximated as realizations of iid RVs [88]. Alternatively,
we can enforce the i.i.d. assumption if we generate synthetic data using a random number
generator. Such synthetic iid data points could be obtained by sampling algorithms that
incrementally build a synthetic dataset by adding randomly chosen raw data points [33].
2.2 The Model
Consider some ML application that generates data points, each characterized by features
x 2 X and label y 2 Y. The goal of a ML method is to learn a hypothesis map h : X ! Y
such that
y   h|{(xz})
^y
for any data point: (2.2)
The informal goal (2.2) will be made precise in several aspects throughout the rest of our
book. First, we need to quantify the approximation error (2.2) incurred by a given hypothesis
map h. Second, we need to make precise what we actually mean by requiring (2.2) to hold
for \any" data point. We solve the  rst issue by the concept of a loss function in Section
2.3. The second issue is then solved in Chapter 4 by using a simple probabilistic model for
data.
Let us assume for the time being that we have found a reasonable hypothesis h in the
sense of (2.2). We can then use this hypothesis to predict the label of any data point for
which we know its features. The prediction ^y = h(x) is obtained by evaluating the hypothesis
for the features x of a data point (see Figure 2.6 and 2.7). We refer to a hypothesis map
47
also as a predictor map since it is used or compute the prediction ^y of a (true) label y.
For ML problems using a  nite label space Y (e..g, Y = f􀀀1; 1g, we refer to a hypothesis
also as a classi er. For a  nite Y, we can characterize a particular classi er map h using its
di erent decision regions
Ra :=
 
x 2 Rn : h(x) = a
 
  X: (2.3)
Each label value a 2 Y is associated with a speci c decision region Ra. For a given label
value a 2 Y, the decision region Ra is constituted by all feature vectors x 2 X which are
mapped to this label value, h(x) = a.
Figure 2.6: A hypothesis (predictor) h maps features x2X, of an on-board camera snapshot,
to the prediction ^y=h(x)2Y for the coordinate of the current location of a cleaning robot.
ML methods use data to learn predictors h such that ^y y (with true label y).
In principle, ML methods could use any possible map h : X ! Y to predict the label
y 2 Y via computing ^y = h(x). The set of all maps from the feature space X to the label
space is typically denoted as YX .1 In general, the set YX is way too large to be search over
by a practical ML methods. As a point in case, consider data points characterized by a single
numeric feature x 2 R and label y 2 R. The set of all real-valued maps h(x) of a real-valued
argument already contains uncountably in nite many di erent hypothesis maps [57].
Practical ML methods can search and evaluate only a (tiny) subset of all possible hypothesis
maps. This subset of computationally feasible (\a ordable") hypothesis maps is referred
to as the hypothesis space or model underlying a ML method. As depicted in Figure 2.10,
ML methods typically use a hypothesis space H that is a tiny subset of YX . Similar to the
features and labels used to characterize data points, also the hypothesis space underlying a
ML method is a design choice. As we will see, the choice for the hypothesis space involves
1The notation YX is to be understood as a symbolic shorthand and should not be understood literately
as a power such as 45.
48
feature x prediction h(x)
0 0
1/10 10
2/10 3
...
...
1 22.3
Table 2.1: A look-up table de nes a hypothesis map h. The value h(x) is given by the
entry in the second column of the row whose  rst column entry is x. We can construct a
hypothesis space H by using a collection of di erent look-up tables.
a trade-o  between computational complexity and statistical properties of the resulting ML
methods.
The preference for a particular hypothesis space often depends on the computational infrastructure
that is available to a ML method. Di erent computational infrastructures favour
di erent hypothesis spaces. ML methods implemented in a small embedded system, might
prefer a linear hypothesis space which results in algorithms that require a small number
of arithmetic operations. Deep learning methods implemented in a cloud computing environment
typically use much larger hypothesis spaces obtained from large ANN (see Section
3.11).
ML methods can also be implemented using a spreadsheet software. Here, we might use
a hypothesis space consisting of maps h : X ! Y that are represented by look up tables
(see Table 2.1). If we instead use the programming language Python to implement a ML
method, we can obtain a hypothesis class by collecting all possible Python subroutines with
one input (scalar feature x), one output argument (predicted label ^y) and consisting of less
than 100 lines of code.
Broadly speaking, the design choice for the hypothesis space H of a ML method has to
balance between two con
icting requirements.
• It has to be su ciently large such that it contains at least one accurate predictor map
^h
2 H. A hypothesis space H that is too small might fail to include a predictor map
required to reproduce the (potentially highly non-linear) relation between features and
label.
Consider the task of grouping or classifying images into \cat" images and \no cat
image". The classi cation of each image is based solely on the feature vector ob-
49
tained from the pixel colour intensities. The relation between features and label
(y 2 fcat; no catg) is highly non-linear. Any ML method that uses a hypothesis space
consisting only of linear maps will most likely fail to learn a good predictor (classi er).
We say that a ML method is under tting if it uses a hypothesis space that does not
contain any hypotheses maps that can accurately predict the label of any data points.
• It has to be su ciently small such that its processing  ts the available computational
resources (memory, bandwidth, processing time). We must be able to e ciently search
over the hypothesis space to  nd good predictors (see Section 2.3 and Chapter 4).
This requirement implies also that the maps h(x) contained in H can be evaluated
(computed) e ciently [5]. Another important reason for using a hypothesis space H
that is not too large is to avoid over tting (see Chapter 7). If the hypothesis space H
is too large, then we can easily  nd a hypothesis which (almost) perfectly predicts the
labels of data points in a training set which is used to learn a hypothesis. However,
such a hypothesis might deliver poor predictions for labels of data points outside the
training set. We say that the hypothesis does not generalize well.
2.2.1 Parametrized Hypothesis spaces
A wide range of current scienti c computing environments allow for e cient numerical linear
algebra. This hard- and software allows to e ciently process data that is provided in the
form of numeric arrays such as vectors, matrices or tensors [112]. To take advantage of such
computational infrastructure, many ML methods use the hypothesis space
H(n) :=fh(w) :Rn!R : h(w)(x):=xTw with some parameter vector w2Rng: (2.4)
The hypothesis space (2.4) is constituted by linear maps (functions)
h(w)􀀀
x
 
: Rn ! R : x 7! wTx: (2.5)
The function h(w) (2.5) maps, in a linear fashion, the feature vector x 2 Rn to the predicted
label h(w)(x) = xTw 2 R. For n = 1 the feature vector reduces a single feature x and the
hypothesis space (2.4) consists of all maps h(w)(x) = wx with weight w 2 R (see Figure 2.8).
The elements of the hypothesis space H in (2.4) are parametrized by the parameter
vector w 2 Rn. Each map h(w) 2 H is fully speci ed by the parameter vector w 2 Rn. This
parametrization of the hypothesis space H allows to process and manipulate hypothesis
50
Figure 2.7: Consider a hypothesis h : X ! Y that is used for locating a cleaning robot. The
hypothesis h reads in the feature vector x(t) 2 X, that might be RGB pixel intensities of
an on-board camera snapshot, at time t. It then outputs a prediction ^y(t) = h(x(t)) for the
y-coordinate y(t) of the cleaning robot at time t. A key problem studied within ML is how
to automatically learn a good (accurate) predictor h such that y(t)   h(x(t)).
1
1 h(1)(x)=x
h(0:2)(x)=0:2x
h(0:7)(x)=0:7x
feature x
h(w)(x)
Figure 2.8: Three particular members of the hypothesis space H = fh(w) : R ! R; h(w)(x) =
wxg which consists of all linear functions of the scalar feature x. We can parametrize this
hypothesis space conveniently using the weight w 2 R as h(w)(x) = wx.
51
maps by vector operations. In particular, instead of searching over the function space H
(its elements are functions!) to  nd a good hypothesis, we can equivalently search over all
possible parameter vectors w 2 Rn.
The search space Rn is still (uncountably) in nite but it has a rich geometric and algebraic
structure that allows to e ciently search over this space. Chapter 5 discusses methods that
use the concept ot a gradient to implement an e cient search for useful parameter vectors
w 2 Rn.
The hypothesis space (2.4) is also appealing because of the broad availability of computing
hardware such as graphic processing units. Another factor boosting the widespread use of
(2.4) might be the o er for optimized software libraries for numerical linear algebra.
The hypothesis space (2.4) can also be used for classi cation problems, e.g., with label
space Y = f􀀀1; 1g. Indeed, given a linear predictor map h(w) we can classify data points
according to ^y=1 for h(w)(x)   0 and ^y=􀀀1 otherwise. We refer to a classi er that computes
the predicted label by  rst applying a linear map to the features as a linear classi er.
Figure 2.9 illustrates the decision regions (2.3) of a linear classi er for binary labels.
The decision regions are half-spaces and, in turn, the decision boundary is a hyperplane
fx : wTx = bg. Note that each linear classi er corresponds to a particular linear hypothesis
map from the hypothesis space (2.4). However, we can use di erent loss functions to measure
the quality of a linear classi er. Three widely-used examples for ML methods that learn a
linear classi er are logistic regression (see Section 3.6), the SVM (see Section 3.7) and the
naive Bayes classi er (see Section 3.8).
In some application domains, the relation between features x and label y of a data point
is highly non-linear. As a case in point, consider data points that are images of animals. The
map that relates the pixel intensities of an image to a label value indicating if the image shows
a cat is highly non-linear. For such applications, the hypothesis space (2.4) is not suitable as
it only contains linear maps. The second main example for a parametrized hypothesis space
studied in this book contains also non-linear maps. This parametrized hypothesis space is
obtained from a parametrized signal 
ow diagram which is referred to as an ANN. Section
3.11 discusses the construction of non-linear parametrized hypothesis spaces using an ANN.
Upgrading a Hypothesis Space via Feature Maps. Let us discuss a simple but
powerful technique for enlarging (\upgrading") a given hypothesis space H to a larger hypothesis
space H0   H that o ers a wider selection of hypothesis maps. The idea is to
replace the original features x of a data point with new (transformed) features z =  (x).
The transformed features are obtained by applying a feature map  ( ) to the original features
52
adf
h(x) < 0; ^y = 􀀀1
decision boundary
h(x)   0; ^y = 1 w
Figure 2.9: A hypothesis h : X !Y for a binary classi cation problem, with label space
Y = f􀀀1; 1g and feature space X = R2, can be represented conveniently via the decision
boundary (dashed line) which separates all feature vectors x with h(x)   0 from the region
of feature vectors with h(x) < 0. If the decision boundary is a hyperplane fx : wTx = bg
(with normal vector w 2 Rn), we refer to the map h as a linear classi er.
x. This upgraded hypothesis space H0 consists of all concatenations of the feature map  
and some hypothesis h 2 H,
H0 :=
 
h0(x) := h
􀀀
 (x)
 
: h 2 H
 
: (2.6)
The construction (2.6) used for arbitrary combinations of a feature map  ( ) and a \base"
hypothesis space H. The only requirement is that the output of the feature map can be used
as input for a hypothesis h 2 H. More formally, the range of the feature map must belong
to the domain of the maps in H. Examples for ML methods that use a hypothesis space of
the form (2.6) include polynomial regression (see Section 3.2), Gaussian basis regression (see
Section 3.5) and the important family of kernel methods (see Section 3.9). The feature map
in (2.6) might also be obtained from clustering or feature learning methods (see Section 8.4
and Section 9.2.1).
For the special case of the linear hypothesis space (2.4), the resulting enlarged hypothesis
space (2.6) is given by all linear maps wT z of the transformed features  (x). Combining
the hypothesis space (2.4) with a non-linear feature map results in a hypothesis space that
contains non-linear maps from the original feature vector x to the predicted label ^y,
^y = wT z = wT (x): (2.7)
Non-Numeric Features. The hypothesis space (2.4) can only be used for data points
53
whose features are numeric vectors x = (x1; : : : ; xn)T 2 Rn. In some application domains,
such as natural language processing, there is no obvious natural choice for numeric features.
However, since ML methods based on the hypothesis space (2.4) are well developed (using
numerical linear algebra), it might be useful to construct numerical features even for nonnumeric
data (such as text). For text data, there has been signi cant progress recently on
methods that map a human-generated text into sequences of vectors (see [49, Chap. 12] for
more details). Moreover, Section 9.3 will discuss an approach to generate numeric features
for data points that have an intrinsic notion of similarity.
2.2.2 The Size of a Hypothesis Space
The notion of a hypothesis space being too small or being too large can be made precise in
di erent ways. The size of a  nite hypothesis space H can be de ned as its cardinality jHj
which is simply the number of its elements. For example, consider data points represented
by 100 10=1000 black-and-white pixels and characterized by a binary label y 2 f0; 1g. We
can model such data points using the feature space X = f0; 1g1000 and label space Y = f0; 1g.
The largest possible hypothesis space H = YX consists of all maps from X to Y. The size
or cardinality of this space is jHj = 221000 .
Many ML methods use a hypothesis space which contains in nitely many di erent predictor
maps (see, e.g., (2.4)). For an in nite hypothesis space, we cannot use the number of
its elements as a measure for its size. Indeed, for an in nite hypothesis space, the number of
elements is not well-de ned. Therefore, we measure the size of a hypothesis space H using
its e ective dimension de  (H).
Consider a hypothesis space H consisting of maps h : X ! Y that read in the features
x 2 X and output an predicted label ^y = h(x) 2 Y. We de ne the e ective dimension de  (H)
of H as the maximum number D 2 N such that for any set D =
 􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(D); y(D)
 
g
of D data points with di erent features, we can always  nd a hypothesis h 2 H that perfectly
 ts the labels, y(i) = h
􀀀
x(i)
 
for i = 1; : : : ;D.
The e ective dimension of a hypothesis space is closely related to the Vapnik{Chervonenkis
(VC) dimension [144]. The VC dimension is maybe the most widely used concept for measuring
the size of in nite hypothesis spaces [144, 126, 13, 58]. However, the precise de nition
of the VC dimension are beyond the scope of this book. Moreover, the e ective dimension
captures most of the relevant properties of the VC dimension for our purposes. For a precise
de nition of the VC dimension and discussion of its applications in ML we refer to [126].
Let us illustrate the concept of e ective dimension as a measure for the size of a hypothesis
54
space with two examples: linear regression and polynomial regression.
Linear regression uses the hypothesis space
H(n) = fh : Rn ! R : h(x) = wTx with some vector w 2 Rng:
Consider a dataset D =
 􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
  
consisting of m data points. We refer
to this number also as the sample size of the dataset. Each data point is characterized by
a feature vector x(i) 2 Rn and a numeric label y(i) 2 R. Let us further assume that data
points are realizations of iid RVs with a common probability distribution.
Under the i.i.d. assumption, the matrix
X =
􀀀
x(1); : : : ; x(m) 
2 Rn m;
which is obtained by stacking (column-wise) the feature vectors x(i) (for i = 1; : : : ;m), is
full (column-) rank with probability one. Basic results of linear algebra allow to show that
the data points in D can be perfectly  t by a linear map h 2 H(n) as long as m   n.
As long as the number m of data points does not exceed the number of features characterizing
each data point, i.e., as long as m   n, we can  nd (with probability one) a parameter
vector bw
such that y(i) = bw
Tx(i) for all i = 1; : : : ;m. The e ective dimension of the linear
hypothesis space H(n) is therefore D = n.
As a second example, consider the hypothesis space H(n)
poly which is constituted by the
set of polynomials with maximum degree n. The fundamental theorem of algebra tells us
that any set of m data points with di erent features can be perfectly  t by a polynomial of
degree n as long as n   m. Therefore, the e ective dimension of the hypothesis space H(n)
poly
is D = n. Section 3.2 discusses polynomial regression in more detail.
YX
H
Figure 2.10: The hypothesis space H is a (typically very small) subset of the (typically very
large) set YX of all possible maps from feature space X into the label space Y.
55
2.3 The Loss
Every ML method uses a (more of less explicit) hypothesis space H which consists of all
computationally feasible predictor maps h. Which predictor map h out of all the maps in
the hypothesis space H is the best for the ML problem at hand? To answer this questions,
ML methods use the concept of a loss function. Formally, a loss function is a map
L : X   Y   H ! R+ :
􀀀􀀀
x; y
 
; h
 
7! L
􀀀
(x; y); h
 
which assigns a pair consisting of a data point, itself characterized by features x and label
y, and a hypothesis h 2 H the non-negative real number L
􀀀
(x; y); h
 
.
hypothesis h
L
􀀀
(x; y); h
 
Figure 2.11: Some loss function L
􀀀
(x; y); h
 
for a  xed data point, with features x and
label y, and varying hypothesis h. ML methods try to  nd (learn) a hypothesis that incurs
minimum loss.
The loss value L
􀀀
(x; y); h
 
quanti es the discrepancy between the true label y and the
predicted label h(x). A small (close to zero) value L
􀀀
(x; y); h
 
indicates a low discrepancy
between predicted label and true label of a data point. Figure 2.11 depicts a loss function
for a given data point, with features x and label y, as a function of the hypothesis h 2 H.
The basic principle of ML methods can then be formulated as: Learn ( nd) a hypothesis out
of a given hypothesis space H that incurs a minimum loss L
􀀀
(x; y); h
 
for any data point
(see Chapter 4).
Much like the choice for the hypothesis space H used in a ML method, also the loss
function is a design choice. We will discuss some widely used examples for loss function in
Section 2.3.1 and Section 2.3.2. The choice for the loss function should take into account the
56
computational complexity of searching the hypothesis space for a hypothesis with minimum
loss. Consider a ML method that uses a parametrized hypothesis space and a loss function
that is a convex and di erentiable (smooth) function of the parameters of a hypothesis.
In this case, searching for a hypothesis with small loss can be done e ciently using the
gradient-based methods discussed in Chapter 5. The minimization of a loss function that
is either non-convex or non-di erentiable is typically computationally much more di cult.
Section 4.2 discusses the computational complexities of di erent types of loss functions in
more detail.
Beside computational aspects, the choice for the loss function should also take into account
statistical aspects. For example, some loss functions result in ML methods that are
more robust against outliers (see Section 3.3 and Section 3.7). The choice of loss function
might also be guided by probabilistic models for the data generated in an ML application.
Section 3.12 details how the maximum likelihood principle of statistical inference provides
an explicit construction of loss functions in terms of an (assumed) probability distribution
for data points.
The choice for the loss function used to evaluate the quality of a hypothesis might also
be in
uenced by its interpretability. Section 2.3.2 discusses loss functions for hypotheses
that are used to classify data points into two categories. It seems natural to measure the
quality of such a hypothesis by the average number of wrongly classi ed data points, which
is precisely the average 0=1 loss (2.9) (see Section 2.3.2).
In contrast to its appealing interpretation as error-rate, the computational aspects of
the average 0=1 loss are less pleasant. Minimizing the average 0=1 loss to learn an accurate
hypothesis amounts to a non-convex and non-smooth optimization problem which is
computationally challenging. Section 2.3.2 introduces the logistic loss as a computationally
attractive alternative choice for the loss function in binary classi cation problems. Learning
a hypothesis that minimizes a (average) logistic loss amounts to a smooth convex optimization
problem. Chapter 5 discusses computationally cheap gradient-based methods for solving
smooth convex optimization problems.
The above aspects (computation, statistic, interpretability) result typically in con
icting
goals for the choice of a loss function. A loss function that has favourable statistical
properties might incur a high computational complexity of the resulting ML method. Loss
functions that result in computationally e cient ML methods might not allow for an easy
interpretation (what does it mean intuitively if the logistic loss of a hypothesis in a binary
classi cation problem is 10􀀀1?). It might therefore be useful to use di erent loss functions
57
for the search of a good hypothesis (see Chapter 4) and for its  nal evaluation. Figure 2.12
depicts an example for two such loss functions, one of them used for learning a hypothesis
by minimizing the loss and the other one used for the  nal performance evaluation.
For example, in a binary classi cation problem, we might use the logistic loss to search
for (learn) an accurate hypothesis using the optimization methods in Chapter 4. The logistic
loss is appealing for this purpose as it can be minimized via e cient gradient-based methods
(see Chapter 5). After having found (learnt) an accurate hypothesis, we use the average 0=1
loss for the  nal performance evaluation. The 0=1 loss is appealing for this purpose as it
can be interpreted as an error or misclassi cation rate. The loss function used for the  nal
performance evaluation of a learnt hypothesis is sometimes referred to as metric.
hypothesis h
loss for learning a good h
loss (metric) for  nal evaluation
Figure 2.12: Two di erent loss functions for a given data point and varying hypothesis h.
One of these loss functions (solid curve) is used to learn a good hypothesis by minimizing
the loss. The other loss function (dashed curve) is used to evaluate the performance of the
learnt hypothesis. The loss function used for this  nal performance evaluation is sometimes
referred to as a metric.
2.3.1 Loss Functions for Numeric Labels
For ML problems involving data points with a numeric label y 2 R, i.e., for regression
problems (see Section 2.1.2), a widely used ( rst) choice for the loss function can be the
squared error loss
L
􀀀
(x; y); h
 
:=
􀀀
y 􀀀 |h{(xz})
=^y
 2
: (2.8)
58
The squared error loss (2.8) depends on the features x only via the predicted label value
^y = h(x). We can evaluate the squared error loss solely using the prediction h(x) and the
true label value y. Besides the prediction h(x), no other properties of the features x are
required to determine the squared error loss. We will slightly abuse notation and use the
shorthand L
􀀀
y; ^y
 
for any loss function that depends on the features x only via the predicted
label ^y = h(x). Figure 2.13 depicts the squared error loss as a function of the prediction
error y 􀀀 ^y.
􀀀2 􀀀1 1 2
1
2
prediction error y 􀀀 h(x)
squared error loss L
Figure 2.13: A widely used choice for the loss function in regression problems (with data
points having numeric labels) is the squared error loss (2.8). Note that, for a given hypothesis
h, we can evaluate the squared error loss only if we know the features x and the label y of
the data point.
The squared error loss (2.8) has appealing computational and statistical properties. For
linear predictor maps h(x) = wTx, the squared error loss is a convex and di erentiable
function of the parameter vector w. This allows, in turn, to e ciently search for the optimal
linear predictor using e cient iterative optimization methods (see Chapter 5). The squared
error loss also has a useful interpretation in terms of a probabilistic model for the features
and labels. Minimizing the squared error loss is equivalent to maximum likelihood estimation
within a linear Gaussian model [58, Sec. 2.6.3].
Another loss function used in regression problems is the absolute error loss j^y 􀀀 yj.
Using this loss function to guide the learning of a predictor results in methods that are
robust against a few outliers in the training set (see Section 3.3). However, this improved
robustness comes at the expense of increased computational complexity of minimizing the
(non-di erentiable) absolute error loss compared to the (di erentiable) squared error loss
59
(2.8).
2.3.2 Loss Functions for Categorical Labels
Classi cation problems involve data points whose labels take on values from a discrete label
space Y. In what follows, unless stated otherwise, we focus on binary classi cation problems.
Moreover, without loss of generality, we use the label space Y = f􀀀1; 1g. Classi cation
methods aim at learning a hypothesis or classi er that maps the features x of a data point
to a predicted label ^y 2 Y.
It is often convenient to implement a classi er by thresholding the value h(x) 2 R of
a hypothesis map that can deliver arbitrary real numbers. We then classify a data point
as ^y = 1 if h(x) > 0 and ^y = 􀀀1 otherwise. Thus, the predicted label is obtained from
the sign of the value h(x). While the sign of h(x) determines the predicted label ^y, we can
interpret the absolute value jh(x)j as the con dence in this classi cation. Is is customary
to abuse notation and refer to both, the  nal classi cation rule (obtained by a thresholding
step) x 7! ^y and the hypothesis h(x) (whose outout is thresholded) as a binary classi er.
In principle, we can measure the quality of a hypothesis when used to classify data points
using the squared error loss (2.8). However, the squared error is typically a poor measure
for the quality of a hypothesis h(x) that is used to classify a data point with binary label
y 2 f􀀀1; 1g. Figure 2.14 illustrates how the squared error loss of a hypothesis can be (highly)
misleading for binary classi cation.
Figure 2.14 depicts a dataset D consisting of m = 4 data points with binary labels
y(i) 2 f􀀀1; 1g, for i = 1; : : : ;m. The  gure also depicts two candidate hypotheses h(1)(x)
and h(2)(x) that can be used for classifying data points. The classi cations ^y obtained with
the hypothesis h(2)(x) would perfectly match the labels of the four training data points since
h(2)
􀀀
x(i)
 
  0 if and if only if y(i) = 1. In contrast, the classi cations ^y(i) obtained by
thresholding h(1)(x) are wrong for data points with y = 􀀀1.
Looking at D, we might prefer using h(2)(x) over h(1) to classify data points. However,
the squared error loss incurred by the (reasonable) classi er h(2) is much larger than the
squared error loss incurred by the (poor) classi er h(1). The squared error loss is typically
a bad choice for assessing the quality of a hypothesis map that is used for classifying data
points into di erent categories.
Generally speaking, we want the loss function to punish (deliver large values for) a
hypothesis that is very con dent (jh(x)j is large) in a wrong classi cation (^y 6= y). Moreover,
a useful loss function loss function should not punish (deliver small values for) a hypothesis
60
(1;􀀀1)
(2;􀀀1)
(5; 1) (x=7; y=1)
feature x
predictor h(2)(x) = 2(x􀀀3)
predictor h(1)(x) = 1
label y
Figure 2.14: A training set consisting of four data points with binary labels ^y(i) 2 f􀀀1; 1g.
Minimizing the squared error loss (2.8) would prefer the (poor) classi er h(1) over the (reasonable)
classi er h(2).
is very con dent (jh(x)j is large) in a correct classi cation (^y = y). However, by its very
de nition, the squared loss (2.8) yields large values if the con dence jh(x)j is large, no matter
if the resulting (after thresholding) classi cation is correct or wrong.
We now discuss some loss functions which have proven useful for assessing the quality
of a hypothesis that is used to classify data points. Unless noted otherwise, the formulas
for these loss functions are valid only if the label values are the real numbers 􀀀1 and 1 (the
label space is Y = f􀀀1; 1g). These formulas need to modi ed accordingly if di erent label
values are used. For example, instead of the label space Y = f􀀀1; 1g, we could equally well
use the label space Y = f0; 1g, or Y = f ;4g or Y = f \Class 1"; \Class 2"g.
A natural choice for the loss function can be based on the requirement that a reasonable
hypothesis should deliver a correct classi cations, ^y = y for any data point. This suggests
to learn a hypothesis h(x) by minimizing the 0=1 loss
L
􀀀
(x; y); h
 
:=
8<
:
1 if y 6= ^y
0 else,
with ^y = 1 for h(x)   0, and ^y = 􀀀1 for h(x) < 0: (2.9)
Figure 2.15 illustrates the 0=1 loss (2.9) for a data point with features x and label y=1 as a
function of the hypothesis value h(x). The 0=1 loss is equal to zero if the hypothesis yields
a correct classi cation ^y = y. For a wrong classi cation ^y 6= y, the 0=1 loss yields the value
one.
61
The 0=1 loss (2.9) is conceptually appealing when data points are interpreted as realizations
of iid RVs with the same probability distribution p(x; y). Given m realizations
(x(i); y(i))
 m
i=1 of such iid RVs,
(1=m)
Xm
i=1
L
􀀀
(x(i); y(i)); h
 
  p(y 6= ^y) (2.10)
with high probability for su ciently large sample size m. A precise formulation of the
approximation (2.10) can be obtained from the law of large numbers [12, Section 1]. We
can apply the law of large numbers since the loss values L
􀀀
(x(i); y(i)); h
 
are realizations of
iid RVs. It is customary to indicate the average 0=1 loss of a hypothesis as the accuracy
1 􀀀 (1=m)
Pm
i=1 L
􀀀
(x(i); y(i)); h
 
.
In view of (2.10), the 0=1 loss (2.9) seems a very natural choice for assessing the quality
of a classi er if our goal is to enforce correct classi cations ^y = y. This appealing statistical
property of the 0=1 loss comes at the cost of a high computational complexity. Indeed,
for a given data point (x; y), the 0=1 loss (2.9) is non-convex and non-di erentiable when
viewed as a function of the hypothesis h. Thus, ML methods that use the 0=1 loss to learn
a hypothesis map typically involve advanced optimization methods to solve the resulting
learning problem (see Section 3.8).
To avoid the non-convexity of the 0=1 loss (2.9) we might approximate it by a convex
loss function. One popular convex approximation of the 0=1 loss is the hinge loss
L
􀀀
(x; y); h
 
:= maxf0; 1 􀀀 yh(x)g: (2.11)
Figure 2.15 depicts the hinge loss (2.11) as a function of the hypothesis h(x). The hinge
loss (2.11) becomes minimal (equal to zero) for a correct classi cation (^y = y) with suf-
 cient con dence h(x)   1. For a wrong classi cation (^y 6= y), the hinge loss increases
monotonically with the con dence jh(x)j in the wrong classi cation. While the hinge loss
avoids the non-convexity of the 0=1 loss, it still is a non-di erentiable function of h(x). A
non-di erentiable loss function cannot be minimized by simple gradient-based methods (see
Chapter 5) but require more advanced optimization methods.
Beside the 0=1 loss and the hinge loss, another popular loss function for binary classi -
cation problems is the logistic loss
L
􀀀
(x; y); h
 
:= log(1 + exp(􀀀yh(x))): (2.12)
62
The logistic loss (2.12) is used in logistic regression (see Section 3.6) to measure the usefulness
of a linear hypothesis h(x) = wTx. Figure 2.15 depicts the logistic loss (2.12) as a function
of the hypothesis h(x). The logistic loss (2.12) is a convex and di erentiable function of
h(x). For a correct classi cation (^y = y), the logistic loss (2.12) decreases monotonically
with increasing con dence h(x). For a wrong classi cation (^y 6=y), the logistic loss increases
monotonically with increasing con dence jh(x)j in the wrong classi cation.
Both the hinge loss (2.11) and the logistic loss (2.12) are convex functions of the weights
w 2 Rn in a linear hypothesis map h(x) = wTx. However, in contrast to the hinge loss, the
logistic loss (2.12) is also a di erentiable function of the w. The convex and di erentiable
logistic loss function can be minimized using simple gradient-based methods such as gradient
descent (GD) (see Chapter 5.5). In contrast, we cannot use basic gradient-based methods to
minimize the hinge loss since it is not di erentiable (it does not have a gradient everywhere).
However, we can apply a generalization of GD which is known as subgradient descent [17].
Subgradient descent is obtained from GD by generalizing the concept of a gradient to that
of a subgradient.
􀀀2 􀀀1 1 2
1
2
hypothesis h(x)
loss L very con dent in ^y=1 )
logistic loss (for y=1)
squared error (for y=1)
hinge loss (for y=1)
0=1 loss (for y=1)
( very con dent in ^y=􀀀1
Figure 2.15: The solid curves depict three widely-used loss functions for binary classi cation.
A data point with features x and label y = 1 is classi ed as ^y = 1 if h(x)   0 and classi ed
as ^y = 􀀀1 if h(x) < 0. We can interpret the absolute value jh(x)j as the con dence in the
classi cation ^y. The more con dent we are in a correct classi cation (^y = y = 1), i.e, the
more positive h(x), the smaller the loss. Note that each of the three loss functions for binary
classi cation tends monotonically towards 0 for increasing h(x). The dashed curve depicts
the squared error loss (2.8), which increases for increasing h(x).
63
2.3.3 Loss Functions for Ordinal Label Values
Some loss functions are particularly suited for predicting ordinal label values (see Section
2.1). Consider data points representing areal images of rectangular areas of size 1km by
1km. We characterize each data point (rectangular area) by the feature vector x obtained
by stacking the RGB values of each image pixel (see Figure 2.5). Beside the feature vector,
each rectangular area is characterized by a label y 2 f1; 2; 3g where
• y = 1 means that the area contains no trees.
• y = 2 means that the area is partially covered by trees.
• y = 3 means that the are is entirely covered by trees.
We might consider the label value y = 2 to be \larger" than label value y = 1 and label
value y = 3 to be \larger" than label value y = 2. Let us construct a loss function that takes
such an ordering of label values into account when evaluating the quality of the predictions
h(x).
Consider a data point with feature vector x and label y = 1 as well as two di erent
hypotheses h(a); h(b) 2 H. The hypothesis h(a) delivers the predicted label ^y(a) = h(a)(x) =
2, while the other hypothesis h(b) delivers the predicted label ^y(a) = h(a)(x) = 3. Both
predictions are wrong, since they are di erent form the true label value y = 1. It seems
reasonable to consider the prediction ^y(a) to be less wrong than the prediction ^y(b) and
therefore we would prefer the hypothesis h(a) over h(b). However, the 0=1 loss is the same for
h(a) and h(b) and therefore does not re
ect our preference for h(a). We need to modify (or
tailor) the 0=1 loss to take into account the application-speci c ordering of label values. For
the above application, we might de ne a loss function via
L
􀀀
(x; y); h
 
:=
8>>><
>>>:
0 , when y = h(x)
10 , when jy 􀀀 h(x)j = 1
100 otherwise.
(2.13)
2.3.4 Empirical Risk
The basic idea of ML methods (including those discussed in Chapter 3) is to  nd (or learn)
a hypothesis (out of a given hypothesis space H) that incurs minimum loss when applied to
arbitrary data points. To make this informal goal precise we need to specify what we mean
64
by \arbitrary data point". One of the most successful approaches to de ne the notion of
\arbitrary data point" is by probabilistic models for the observed data points.
The most basic and widely-used probabilistic model interprets data points
􀀀
x(i); y(i)
 
as realizations of iid RVs with a common probability distribution p(x; y). Given such a
probabilistic model, it seems natural to measure the quality of a hypothesis h by the expected
loss or Bayes risk [85]
E
 
L
􀀀
(x; y); h
 
g :=
Z
x;y
L
􀀀
(x; y); h
 
dp(x; y): (2.14)
The Bayes risk of h is the expected value of the loss L
􀀀
(x; y); h
 
incurred when applying
the hypothesis h to (the realization of) a random data point with features x and label y.
Note that the computation of the Bayes risk (2.15) requires the joint probability distribution
p(x; y) of the (random) features and label of data points.
The Bayes risk (2.15) seems to be reasonable performance measure for a hypothesis h.
Indeed, the Bayes risk of a hypothesis is small only if the hypothesis incurs a small loss on
average for data points drawn from the probability distribution p(x; y). However, it might be
challenging to verify if the data points generated in a particular application domain can be
accurately modelled as realizations (draws) from a probability distribution p(x; y). Moreover,
it is also often the case that we do not know the correct probability distribution p(x; y).
Let us assume for the moment, that data points are generated as iid realizations of a
common probability distribution p(x; y) which is known. It seems reasonable to learn a
hypothesis h  that incurs minimum Bayes risk,
E
 
L
􀀀
(x; y); h  
g := min
h2H
E
 
L
􀀀
(x; y); h
 
g: (2.15)
A hypothesis that solves (2.15), i.e., that achieves the minimum possible Bayes risk, is referred
to as a Bayes estimator [85, Chapter 4]. The main computational challenge for learning
the optimal hypothesis is the e cient (numerical) solution of the optimization problem
(2.15). E cient methods to solve the optimization problem (2.15) are studied within estimation
theory [85, 152].
The focus of this book is on ML methods which do not require knowledge of the underlying
probability distribution p(x; y). One of the most widely used principle for these ML methods
is to approximate the Bayes risk by an empirical (sample) average over a  nite set of labeled
data points D =
􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
 
. In particular, we de ne the empirical risk of a
65
hypothesis h 2 H for a dataset D as
bL
(hjD) = (1=m)
Xm
i=1
L
􀀀
(x(i); y(i)); h
 
: (2.16)
The empirical risk of the hypothesis h 2 H is the average loss on the data points in D. To
ease notational burden, we use bL
(h) as a shorthand for bL
(hjD) if the underlying dataset
D is clear from the context. Note that in general the empirical risk depends on both, the
hypothesis h and the (features and labels of the) data points in the dataset D.
If the data points used to compute the empirical risk (2.16) are (can be modelled as)
realizations of iid RVs whose common distribution is p(x; y), basic results of probability
theory tell us that
E
 
L
􀀀
(x; y); h
 
g   (1=m)
Xm
i=1
L
􀀀
(x(i); y(i)); h
 
for su ciently large sample size m: (2.17)
The approximation error in (2.17) can be quanti ed precisely by some of the most basic
results of probability theory. These results are often summarized under the umbrella term
law of large numbers [12, 51, 10].
Many (if not most) ML methods are motivated by (2.17) which suggests that a hypothesis
with small empirical risk (2.16) will also result in a small expected loss. The minimum
possible expected loss is achieved by the Bayes estimator of the label y, given the features
x. However, to actually compute the optimal estimator we would need to know the (joint)
probability distribution p(x; y) of features x and label y.
Confusion Matrix
Consider a dataset D with data points characterized by feature vectors x(i) and labels y(i) 2
f1; : : : ; kg. We might interpret the label value of a data point as the index of a category or
class to which the data point belongs to. Multi-class classi cation problems aim at learning
a hypothesis h such that h(x)   y for any data point.
In principle, we could measure the quality of a given hypothesis h by the average 0=1
loss incurred on the labeled data points in (the training set) D. However, if the dataset D
contains mostly data points with one speci c label value, the average 0=1 loss might obscure
the performance of h for data points having one of the rare label values. Indeed, even if
the average 0=1 loss is very small, the hypothesis might perform poorly for data points of a
66
minority category.
The confusion matrix generalizes the concept of the 0=1 loss to application domains where
the relative frequency (fraction) of data points with a speci c label value varies signi cantly
(imbalanced data). Instead of considering only the average 0=1 loss incurred by a hypothesis
on a dataset D, we use a whole family of loss functions. In particular, for each pair of label
values c; c0 2 f1; : : : ; kg, we de ne the loss
L(c!c0)􀀀􀀀
x; y
 
; h
 
:=
8<
:
1 if y = c and h(x) = c0
0 otherwise
: (2.18)
We then compute the average loss (2.18) incurred on the dataset D,
bL
(c!c0)(hjD) := (1=m)
Xm
i=1
L(c!c0)􀀀
(x(i); y(i)); h
 
for c; c0 2 f1; : : : ; kg: (2.19)
It is convenient to arrange the values (2.19) as a matrix which is referred to as a confusion
matrix. The rows of a confusion matrix correspond to di erent values c of the true label of
a data point. The columns of a confusion matrix correspond to di erent values c0 delivered
by the hypothesis h(x). The (c; c0)-th entry of the confusion matrix is bL
(c!c0)(hjD).
Precision, Recall and F-Measure
Consider an object detection application where data points are images. The label of data
points might indicate the presence (y = 1) or absence (y = 􀀀1) of an object, it is then
customary to de ne the [6]
recall := bL
(1!1)(hjD), and the precision :=
bL
(1!1)(hjD)
bL
(1!1)(hjD) +bL
(􀀀1!1)(hjD)
: (2.20)
Clearly, we would like to  nd a hypothesis with both, large recall and large precision. However,
these two goals are typically con
icting, a hypothesis with a high recall will have small
precision. Depending on the application, we might prefer having a high recall and tolerate
a lower precision.
It might be convenient to combine the recall and precision of a hypothesis into a single
quantity,
F1 := 2
precision   recall
precision + recall
(2.21)
67
The F measure (2.21) is the harmonic mean [2] of the precision and recall of a hypothesis
h. It is a special case of the F -score
F  :=
􀀀
1 +  2  precision   recall
 2precision + recall
: (2.22)
The F measure (2.21) is obtained from (2.22) for the choice   = 1. It is therefore customary
to refer to (2.21) as the F1-score of a hypothesis h.
2.3.5 Regret
In some ML applications, we might have access to the predictions obtained from some
reference methods which are referred to as experts [21, 60]. The quality of a hypothesis
h is measured via the di erence between the loss incurred by its predictions h(x) and the
loss incurred by the predictions of the experts. This di erence, which is referred to as
the regret, measures by how much we regret to have used the prediction h(x) instead of
using(or following) the prediction of the expert. The goal of regret minimization is to learn
a hypothesis with a small regret compared to given set of experts.
The concept of regret minimization is useful when we do not make any probabilistic
assumptions (see Section 2.1.4) about the data. Without a probabilistic model we cannot
use the Bayes risk of the (optimal) Bayes estimator as a baseline (or benchmark). The
concept of regret minimization avoids the need for a probabilistic model of the data to
obtain a baseline [21]. This approach replaces the Bayes risk with the regret relative to
given reference methods (the experts).
2.3.6 Rewards as Partial Feedback
Some applications involve data points whose labels are so di cult or costly to determine that
we cannot assume to have any labeled data available. Without any labeled data, we cannot
evaluate the loss function for di erent choices for the hypothesis. Indeed, the evaluation of
the loss function typically amounts to measuring the distance between predicted label and
true label of a data point. Instead of evaluating a loss function, we must rely on some indirect
feedback or \reward" that indicates the usefulness of a particular prediction [21, 136].
Consider the ML problem of predicting the optimal steering directions for an autonomous
car. The prediction has to be recalculated for each new state of the car. ML methods can
sense the state via a feature vector x whose entries are pixel intensities of a snapshot. The
68
goal is to learn a hypothesis map from the feature vector x to a guess ^y = h(x) for the
optimal steering direction y (true label). Unless the car circles around in small area with
 xed obstacles, we have no access to labeled data points or reference driving scenes for which
we already know the optimum steering direction. Instead, the car (control unit) needs to
learn the hypothesis h(x) based solely on the feedback signals obtained from various sensing
devices (cameras, distance sensors).
2.4 Putting Together the Pieces
A guiding theme of this book is that ML methods are obtained by di erent combinations
of data, model and loss. We will discuss some key principles behind these methods in
depth in the following chapters. Let us develop some intuition for how ML methods operate
by considering a very simple ML problem. This problem involves data points that are
characterized by a single numeric feature x 2 R and a numeric label y 2 R. We assume to
have access to m labeled data points
􀀀
x(1); y(1) 
; : : : ;
􀀀
x(m); y(m) 
(2.23)
for which we know the true label values y(i).
The assumption of knowing the exact true label values y(i) for any data point is an
idealization. We might often face labelling or measurement errors such that the observed
labels are noisy versions of the true label. Later on, we will discuss techniques that allow
ML methods to cope with noisy labels in Chapter 7.
Our goal is to learn a (hypothesis) map h : R ! R such that h(x)   y for any data
point. Given a data point with feature x, the function value h(x) should be an accurate
approximation of its label value y. We require the map to belong to the hypothesis space H
of linear maps,
h(w0;w1)(x) = w1x + w0: (2.24)
The predictor (2.24) is parameterized by the slope w1 and the intercept (bias or o set)
w0. We indicate this by the notation h(w0;w1). A particular choice for the weights w1;w0
de nes a linear hypothesis h(w0;w1)(x) = w1x + w0.
Let us use the linear hypothesis map h(w0;w1)(x) to predict the labels of training data
points. In general, the predictions ^y(i) = h(w0;w1)
􀀀
x(i)
 
will not be perfect and incur a nonzero
prediction error ^y(i) 􀀀 y(i) (see Figure 2.16).
69
We measure the goodness of the predictor map h(w0;w1) using the average squared error
loss (see (2.8))
f(w0;w1) := (1=m)
Xm
i=1
􀀀
y(i) 􀀀 h(w0;w1)(x(i))
 2
(2.24)
= (1=m)
Xm
i=1
􀀀
y(i) 􀀀 (w1x(i) + w0)
 2
: (2.25)
The training error f(w0;w1) is the average of the squared prediction errors incurred by the
predictor h(w0;w1)(x) to the labeled data points (2.23).
It seems natural to learn a good predictor (2.24) by choosing the parameters w0;w1 to
minimize the training error
min
w0;w12R
f(w0;w1)
(2.25)
= min
w1;w02R
(1=m)
Xm
i=1
􀀀
y(i) 􀀀 (w1x(i) + w0)
 2
: (2.26)
The optimal parameters ^ w0; ^ w1 are characterized by the zero-gradient condition,2
@f(w0;w1)
@w0
    
w0= ^ w0;w1= ^ w1
= 0, and
@f(w0;w1)
@w1
    
w0= ^ w0;w1= ^ w1
= 0: (2.27)
Inserting (2.25) into (2.27) and by using basic rules for calculating derivatives (see, e.g.,
[119]), we obtain the following optimality conditions
(1=m)
Xm
i=1
􀀀
y(i) 􀀀 ( ^ w1x(i) + ^ w0)
 
= 0, and (2.28)
(1=m)
Xm
i=1
x(i)􀀀
y(i) 􀀀 ( ^ w1x(i) + ^ w0)
 
= 0:
Any parameter values ^ w0; ^ w1 2 R that satisfy (2.28) de ne a hypothesis map h( ^ w0; ^ w1)(x) =
^ w1x + ^ w0 that is optimal in the sense of incurring a minimum training error,
f( ^ w0; ^ w1) = min
w0;w12R
f(w0;w1):
Let us rewrite the optimality condition (2.28) using matrices and vectors. To this end,
2A necessary and su cient condition for bw
to minimize a convex di erentiable function f(w) is rf(bw
) = 0
[15, Sec. 4.2.3].
70
we  rst rewrite the hypothesis (2.24) as
h(x) = wTx with w =
􀀀
w0;w1
 T
; x =
􀀀
1; x
 T
:
Let us stack the feature vectors x(i) =
􀀀
1; x(i)
 T
and labels y(i) of the data points (2.23) into
a feature matrix and a label vector,
X =
􀀀
x(1); : : : ; x(m) T
2 Rm 2; y =
􀀀
y(1); : : : ; y(m) T
2 Rm: (2.29)
We can then reformulate (2.28) as
XT 􀀀
y 􀀀 Xbw
 
= 0: (2.30)
The optimality conditions (2.30) and (2.28) are equivalent in the following sense. The entries
of any parameter vector bw
=
􀀀
^ w0; ^ w1
 
that satis es (2.30) are solutions to (2.28) and viceversa.
feature x
label y
x(1) x(2) x(3)
h(x(1))
h(x(2))
h(x(3))
h(x)
y(1)
y(2)
y(3)
y(1) 􀀀 h(x(1))
Figure 2.16: We can evaluate the quality of a particular predictor h 2 H by measuring the
prediction error y 􀀀 h(x) obtained for a labeled data point (x; y).
71
2.5 Exercises
Exercise 2.1. Perfect Prediction. Consider data points that are characterized by
a single numeric feature x2R and a numeric label y 2 R. We use a ML method to learn a
hypothesis map h : R ! R based on a training set consisting of three data points
(x(1) = 1; y(1) = 3); (x(2) = 4; y(2) = 􀀀1); (x(3) = 1; y(3) = 5):
Is there any chance for the ML method to learn a hypothesis map that perfectly  ts the data
points such that h
􀀀
x(i)
 
= y(i) for i = 1; : : : ; 3. Hint: Try to visualize the data points in a
scatterplot and various hypothesis maps (see Figure 1.3).
Exercise 2.2. Temperature Data. Consider a dataset of daily air temperatures
x(1); : : : ; x(m) measured at the FMI observation station \Utsjoki Nuorgam" during 01.12.2019
and 29.02.2020. Thus, x(1) is the daily temperature measured on 01.12.2019, x(2) is the
daily temperature measure don 02.12.2019, and x(m) is the daily temperature measured on
29.02.2020. You can download this dataset from the link https://en.ilmatieteenlaitos.
fi/download-observations. ML methods often determine few parameters to characterize
large collections of data points. Compute, for the above temperature measurement dataset,
the following quantities:
• the minimum A := mini=1;:::;m x(i)
• the maximum B := maxi=1;:::;m x(i)
• the average C := (1=m)
P
i=1;:::;m x(i)
• the standard deviation D :=
q
(1=m)
P
i=1;:::;m
􀀀
x(i) 􀀀 C
 2
Exercise 2.3. Deep Learning on Raspberry PI. Consider the tiny desktop computer
\RaspberryPI" equipped with a total of 8 Gigabytes memory [32]. We want implement a
ML algorithm that learns a hypothesis map that is represented by a deep ANN involving
n = 106 numeric parameters. Each parameter is quantized using 8 bits (= 1 Byte). How
many di erent hypotheses can we store at most on a RaspberryPI computer? (You can
assume that 1Gigabyte = 109Bytes.)
Exercise 2.4. Ensembles. For some applications it can be a good idea to not learn
a single hypothesis but to learn a whole ensemble of hypothesis maps h(1); : : : ; h(B). These
72
hypotheses might even belong to di erent hypothesis spaces, h(1) 2 H(1); : : : ; h(B) 2 H(B).
These hypothesis spaces can be arbitrary except that they are de ned for the same feature
space and label space. Given such an ensemble we can construct a new (\meta") hypothesis
~h
by combining (or aggregating) the individual predictions obtained from each hypothesis,
~h
(x) := a
􀀀
h(1)(x); : : : ; h(B)(x)
 
: (2.31)
Here, a( ) denotes some given ( xed) combination or aggregation function. One example for
such an aggregation function is the average a
􀀀
h(1)(x); : : : ; h(B)(x)
 
:= (1=B)
PB
b=1 h(b)(x).
We obtain a new \meta" hypothesis space eH
, that consists of all hypotheses of the form
(2.31) with h(1) 2 H(1); : : : ; h(B) 2 H(B). Which conditions on the aggregation function a( )
and the individual hypothesis spaces H(1); : : : ;H(B) ensure that eH
contains each individual
hypothesis space, i.e., H(1); : : : ;H(B)   eH
.
Exercise 2.5. How Many Features? Consider the ML problem underlying a
music information retrieval smartphone app [153]. Such an app aims at identifying a song
title based on a short audio recording of a song interpretation. Here, the feature vector x
represents the sampled audio signal and the label y is a particular song title out of a huge
music database. What is the length n of the feature vector x 2 Rn if its entries are the signal
amplitudes of a 20-second long recording which is sampled at a rate of 44 kHz?
Exercise 2.6. Multilabel Prediction. Consider data points that are characterized
by a feature vector x 2 R10 and a vector-valued label y 2 R30. Such vector-valued labels
arise in multi-label classi cation problems. We want to predict the label vector using a linear
predictor map
h(x) = Wx with some matrix W 2 R30 10: (2.32)
How many di erent linear predictors (2.32) are there ? 10, 30, 40, or in nite?
Exercise 2.7. Average Squared Error Loss as Quadratic Form Consider the
hypothesis space constituted by all linear maps h(x) = wTx with some weight vector w 2 Rn.
We try to  nd the best linear map by minimizing the average squared error loss (the empirical
risk) incurred on labeled data points (training set) (x(1); y(1)); (x(2); y(2)); : : : ; (x(m); y(m)). Is
it possible to represent the resulting empirical risk as a convex quadratic functionf(w) =
wTCw + bw + c? If this is possible, how are the matrix C, vector b and constant c related
to the features and labels of data points in the training set?
73
Exercise 2.8. Find Labeled Data for Given Empirical Risk. Consider linear
hypothesis space consisting of linear maps h(w)(x) = wTx that are parametrized by a weight
vector w. We learn an optimal weight vector by minimizing the average squared error loss
f(w) = bL
􀀀
h(w)jD
 
incurred by h(w)(x) on the training set D =
􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
 
.
Is it possible to reconstruct the dataset D just from knowing the function f(w)?. Is the
resulting labeled training data unique or are there di erent training sets that could have
resulted in the same empirical risk function? Hint: Write down the training error f(w) in
the form f(w) = wTQw + c + bTw with some matrix Q, vector b and scalar c that might
depend on the features and labels of the training data points.
Exercise 2.9. Dummy Feature Instead of Intercept. Show that any hypothesis
map of the form h(x) = w1x + w0 can be obtained from the concatenation of a feature map
  : x 7! z with the linear map ~h(z) := ew
T z using parameter vector ew
=
􀀀
w1;w0
 T
2 R2.
Exercise 2.10. Approximate Non-Linear Maps Using Indicator Functions for
Feature Maps. Consider an ML application generating data points characterized by a
scalar feature x 2 R and numeric label y 2 R. We construct a non-linear map by  rst transforming
the feature x to a new feature vector z = ( 1(x);  2(x);  3(x);  4(x))T 2 R4. The
components  1(x); : : : ;  4(x) are indicator functions of intervals [􀀀10;􀀀5); [􀀀5; 0); [0; 5); [5; 10].
In particular,  1(x) = 1 for x 2 [􀀀10;􀀀5) and  1(x) = 0 otherwise. We obtain a hypothesis
space H(1) by collecting all maps from feature x to predicted label ^y that can written as
a a weighted linear combination wT z (with some parameter vector w) of the transformed
features. Which of the following hypothesis maps belong to H(1)?
x
h(x)
-10 -5 0 5 10
x
h(x)
-10 -5 0 5 10
(a) (b)
Exercise 2.11. Python Hypothesis space. Consider the source codes below for  ve
di erent Python functions that read in the numeric feature x, perform some computations
74
that result in a prediction ^y. How large is the hypothesis space that is constituted by all
maps that can be represented by one of those Python functions.
1 def func1 (x):
2 hat_y = 5*x+3
3 return hat_y
4
1 def func2 (x):
2 tmp = 3*x+3
3 hat_y = tmp +2*x
4 return hat_y
5
1 def func3 (x):
2 tmp = 3*x+3
3 hat_y = tmp -2*x
4 return hat_y
5
1 def func4 (x):
2 tmp = 3*x+3
3 hat_y = tmp -2*x+4
4 return hat_y
5
1 def func5 (x):
2 tmp = 3*x+3
3 hat_y = 4* tmp -2*x
4 return hat_y
5
Exercise 2.12. A Lot of Features. One important application domain for ML
methods is healthcare. Here, data points represent human patients that are characterized by
health-care records. These records might contain physiological parameters, CT scans along
with various diagnoses provided by healthcare professionals. Is it a good idea to use every
data  eld of a healthcare record as features of the data point ?
Exercise 2.13. Over-Parametrization. Consider data points characterized by
feature vectors x 2 R2 and a numeric label y 2 R. We want to learn the best predictor out
of the hypothesis space
H =
 
h(x) = xTAw : w 2 Sg:
Here, we used the matrix A =
 
1 􀀀1
􀀀1 1
!
and the set
S =
 
(1; 1)T ; (2; 2)T ; (􀀀1; 3)T ; (0; 4)T 
  R2:
What is the cardinality of the hypothesis space H, i.e., how many di erent predictor maps
does H contain?
Exercise 2.14. Squared Error Loss. Consider a hypothesis space H constituted by
three predictors h(1)( ); h(2)( ); h(3)( ). Each predictor h(j)(x) is a real-valued function of a
75
real-valued argument x. Moreover, for each j 2 f1; 2; 3g,
h(j)(x) =
8<
:
0 if x2   j
j otherwise.
(2.33)
Can you tell which of these hypothesis is optimal in the sense of having smallest average
squared error loss on the three data points (x = 1=10; y = 3), (0; 0) and (1;􀀀1).
Exercise 2.15. Classi cation Loss. The Figure 2.15 depicts di erent loss functions
for a  xed data point with label y = 1 and varying hypothesis h 2 H. How would Figure
2.15 change if we evaluate the same lloss functions for another data point z = (x; y) with
label y = 􀀀1?
Exercise 2.16. Intercept Term. Linear regression methods model the relation
between the label y and feature x of a data point as y = h(x) + e with some small additive
term e. The predictor map h(x) is assumed to be linear h(x) = w1x+w0. The parameter w0
is sometimes referred to as the intercept (or bias) term. Assume we know for a given linear
predictor map its values h(x) for x = 1 and x = 3. Can you determine the weights w1 and
w0 based on h(1) and h(3)?
Exercise 2.17. Picture Classi cation. Consider a huge collection of outdoor pictures
you have taken during your last adventure trip. You want to organize these pictures as three
categories (or classes) dog, bird and  sh. How could you formalize this task as a ML problem?
Exercise 2.18. Maximal Hypothesis space. Consider data points characterized
by a single real-valued feature x and a single real-valued label y. How large is the largest
possible hypothesis space of predictor maps h(x) that read in the feature value of a data
point and deliver a real-valued prediction ^y = h(x) ?
Exercise 2.19. A Large but Finite Hypothesis space. Consider data points whose
features are 10   10 pixel black-and-white (bw) images. Besides the pixels, each data point
is also characterized by a binary label y 2 f0; 1g. Consider the hypothesis space which is
constituted by all maps that take a 10 10 pixel bw image as input and deliver a prediction
for the label. How large is this hypothesis space?
Exercise 2.20. Size of Linear Hypothesis space. Consider a training set of m
data points with feature vectors x(i) 2 Rn and numeric labels y(1); : : : ; y(m). The feature
76
vectors and labels of the data points in the training set are arbitrary except that we assume
the feature matrix X =
􀀀
x(1); : : : ; x(m)
 
is full rank. What condition on m and n guarantees
that we can  nd a linear predictor h(x) = wTx that perfectly  ts the training set, i.e.,
y(1) = h
􀀀
x(1)
 
; : : : ; y(m) = h
􀀀
x(m)
 
.
Exercise 2.21. Classi cation with Imbalanced Data. Consider a dataset D of
m data points with feature vectors x(i) 2 Rn and discrete-valued labels y(i) 2 f1; 2; : : : ; 10g.
The data is highly imbalanced, more than 90 percent of data points have a label y = 1.
We learn a hypothesis out of the hypothesis space H0 that is constituted by the ten maps
h(c)(x) = c for c = 1; 2; : : : ; 10. Is there a hypothesis h 2 H0 whose average 0=1 loss on D
does not exceed 0:3 ?
Exercise 2.22. Accuracy and Average Logistic loss. Consider a dataset D that
consists of m data points , indexed by i = 1; : : : ;m. Each data point is characterized by a
feature vector x(i) 2 R2 and by a binary label y(i) 2 f􀀀1; 1g. We use a linear hypothesis
map h(w)(x) = wTx to classify data points according to ^y = 1 if h(w)(x)   0 and ^y = 􀀀1
otherwise. Two popular quality measures of a hypothesis are the accuracy and the average
logistic loss (1=m)
P
i L
􀀀􀀀
x(i); y(i)
 
; h(w)
 
with the logistic loss (2.12). The accuracy of a
hypothesis h(w)(x) is de ned as 1 􀀀 (1=m)
P
i L
􀀀􀀀
x(i); y(i)
 
; h(w)
 
with the 0=1 loss (2.9).
Loosely speaking, the accuracy of a hypothesis is \one minus the average 0=1 loss ". Can
you construct a speci c dataset with arbitrary but  nite size m such that there are two
di erent linear hypotheses h(w) and h(w0) with accuracy and average logistic loss of h(w)
being strictly larger than accuracy and average logistic loss of h(w0).
77
Chapter 3
The Landscape of ML
model
absolute
loss
logistic
loss
squared
error
regularized
squared
error
hinge
loss
regret
0=1 loss
loss function
upgraded linear
maps (2.7)
linear maps ANN
piecewise
constant
Sec. 3.3
Sec. 3.1
Sec. 3.2
Sec. 3.5
Sec. 3.9
Sec. 3.9
Sec. 3.9
Sec. 3.6
Sec. 3.7
Sec. 3.4 Sec. 3.9
Sec. 3.15
Sec. 3.8
Sec. 3.10
Sec. 3.13
Sec. 3.10
Sec. 3.13
Sec. 3.11
Sec. 3.11
Figure 3.1: ML methods  t a model to data by minimizing a loss function. Di erent ML
methods use di erent design choices for data, model and loss.
As discussed in Chapter 2, ML methods combine three main components:
78
• a set of data points that are characterized by features and labels
• a model or hypothesis space H that consists of di erent hypotheses h 2 H.
• a loss function to measure the quality of a particular hypothesis h.
Each of these three components involves design choices for the representation of data, their
features and labels, the model and loss function. This chapter details the high-level design
choices used by some of the most popular ML methods. Figure 3.1 depicts these ML methods
in a two-dimensional plane whose horizontal axes represents di erent hypothesis spaces and
the vertical axis represents di erent loss functions.
To obtain a practical ML method we also need to combine the above components. The
basic principle of any ML method is to search the model for a hypothesis that incurs minimum
loss on any data point. Chapter 4 will then discuss a principled way to turn this informal
statement into actual ML algorithms that could be implemented on a computer.
3.1 Linear Regression
Consider data points characterized by feature vectors x 2 Rn and numeric label y 2 R.
Linear regression methods learn a hypothesis out of the linear hypothesis space
H(n) := fh(w) : Rn!R : h(w)(x)=wTx with some parameter vector w 2 Rng: (3.1)
Figure 1.3 depicts the graphs of some maps from H(2) for data points with feature vectors of
the form x = (1; x)T . The quality of a particular predictor h(w) is measured by the squared
error loss (2.8). Using labeled data D = f(x(i); y(i))gmi
=1, linear regression learns a linear
predictor ^h which minimizes the average squared error loss (2.8), or \mean squared error",
^h
= argmin
h2H(n)
bL
(hjD)
(2.16)
= argmin
h2H(n)
(1=m)
Xm
i=1
(y(i) 􀀀 h(x(i)))2: (3.2)
Since the hypothesis space H(n) is parametrized by the parameter vector w (see (3.1)),
79
we can rewrite (3.2) as an optimization of the parameter vector w,
bw
= argmin
w2Rn
(1=m)
Xm
i=1
(y(i) 􀀀 h(w)(x(i)))2
h(w)(x)=wT x
= argmin
w2Rn
(1=m)
Xm
i=1
(y(i) 􀀀 wTx(i))2: (3.3)
The optimization problems (3.2) and (3.3) are equivalent in the following sense: Any optimal
parameter vector bw
which solves (3.3), can be used to construct an optimal predictor^h, which
solves (3.2), via ^h(x) = h(bw
)(x) =
􀀀
bw
 T
x.
3.2 Polynomial Regression
Consider an ML problem involving data points which are characterized by a single numeric
feature x 2 R (the feature space is X = R) and a numeric label y 2 R (the label space is
Y = R). We observe a bunch of labeled data points which are depicted in Figure 3.2.
0:2 0:4 0:6 0:8 1
0:2
0:4
0:6
0:8
1
feature x
label y
Figure 3.2: A scatterplot that depicts a set of data points (x(1); y(1)); : : : ; (x(m); y(m)). The
ith data point is depicted by a dot whose coordinates are the feature x(i) and label y(i) of
that data point.
Figure 3.2 suggests that the relation x 7! y between feature x and label y is highly nonlinear.
For such non-linear relations between features and labels it is useful to consider a
80
hypothesis space which is constituted by polynomial maps
H(n)
poly = fh(w) : R ! R : h(w)(x) =
Xn
j=1
wjxj􀀀1;
with some w=(w1; : : : ;wn)T 2Rng: (3.4)
We can approximate any non-linear relation y=h(x) with any desired level of accuracy using
a polynomial
Pn
j=1 wjxj􀀀1 of su ciently large degree n.1
For linear regression (see Section 3.1), we measure the quality of a predictor by the
squared error loss (2.8). Based on labeled data points D = f(x(i); y(i))gmi
=1, each having a
scalar feature x(i) and label y(i), polynomial regression minimizes the average squared error
loss (see (2.8)):
min
h2H(n)
poly
(1=m)
Xm
i=1
(y(i) 􀀀 h(w)(x(i)))2: (3.5)
It is customary to refer to the average squared error loss also as the mean squared error.
We can interpret polynomial regression as a combination of a feature map (transformation)
(see Section 2.1.1) and linear regression (see Section 3.1). Indeed, any polynomial
predictor h(w) 2 H(n)
poly is obtained as a concatenation of the feature map
 (x) 7! (1; x; : : : ; xn)T 2 Rn+1 (3.6)
with some linear map ~h(w) : Rn+1 ! R : x 7! wTx, i.e.,
h(w)(x) = ~h(w)( (x)): (3.7)
Thus, we can implement polynomial regression by  rst applying the feature map   (see
(3.6)) to the scalar features x(i), resulting in the transformed feature vectors
x(i) =  
􀀀
x(i) 
=
􀀀
1; x(i); : : : ;
􀀀
x(i) n􀀀1 T
2 Rn; (3.8)
and then applying linear regression (see Section 3.1) to these new feature vectors.
By inserting (3.7) into (3.5), we obtain a linear regression problem (3.3) with feature
vectors (3.8). Thus, while a predictor h(w) 2 H(n)
poly is a non-linear function h(w)(x) of the
original feature x, it is a linear function ~h(w)(x) = wTx (see (3.7)), of the transformed
1The precise formulation of this statement is known as the \Stone-Weierstrass Theorem" [119, Thm.
7.26].
81
squared error loss
prediction error ^y 􀀀 y
absolute di erence loss
Figure 3.3: The Huber loss (3.9) resembles the squared error loss (2.8) for small prediction
error and the absolute di erence loss for larger prediction errors.
features x (3.8).
3.3 Least Absolute Deviation Regression
Learning a linear predictor by minimizing the average squared error loss incurred on training
data is not robust against the presence of outliers. This sensitivity to outliers is rooted in
the properties of the squared error loss (y 􀀀 h(x))2. Minimizing the average squared error
forces the resulting predictor ^y to not be too far away from any data point. However, it
might be useful to tolerate a large prediction error y 􀀀 h(x) for an unusual or exceptional
data point that can be considered an outlier.
Replacing the squared loss with a di erent loss function can make the learning robust
against outliers. One important example for such a \robustifying" loss function is the Huber
loss [63]
L
􀀀􀀀
x; y
 
; h
 
=
8< :
(1=2)(y 􀀀 h(x))2 for jy 􀀀 h(x)j   "
"(jy 􀀀 h(x)j 􀀀 "=2) else.
(3.9)
Figure 3.3 depicts the Huber loss as a function of the prediction error y 􀀀 h(x).
The Huber loss de nition (3.9) contains a tuning parameter  . The value of this tuning
parameter de nes when a data point is considered as an outlier. Figure 3.4 illustrates the
role of this parameter as the width of a band around a hypothesis map. The prediction error
of this hypothesis map for data points within this band are measured used squared error loss
(2.8). For data points outside this band (outliers) we use instead the absolute value of the
prediction error as the resulting loss.
82
feature x
label y
h(x)
"
\outlier"
Figure 3.4: The Huber loss measures prediction errors via squared error loss for regular data
points inside the band of width " around the hypothesis map h(x) and via the absolute
di erence loss for an outlier outside the band.
The Huber loss is robust to outliers since the corresponding (large) prediction errors
y 􀀀 ^y are not squared. Outliers have a smaller e ect on the average Huber loss (over the
entire dataset) compared to the average squared error loss. The improved robustness against
outliers of the Huber loss comes at the expense of increased computational complexity. The
squared error loss can be minimized using e cient gradient based methods (see Chapter
5). In contrast, for " = 0, the Huber loss is non-di erentiable and requires more advanced
optimization methods.
The Huber loss (3.9) contains two important special cases. The  rst special case occurs
when " is chosen to be very large, such that the condition jy􀀀^yj   " is satis ed for most data
points. In this case, the Huber loss resembles the squared error loss (2.8) (up to a scaling
factor 1=2). The second special case is obtained for " = 0. Here, the Huber loss reduces to
the scaled absolute di erence loss jy 􀀀 ^yj.
3.4 The Lasso
We will see in Chapter 6 that linear regression (see Section 3.1) typically requires a training
set larger than the number of features used to characterized a data point. However, many
important application domains generate data points with a number n of features much higher
than the number m of available labeled data points in the training set.
In the high-dimensional regime, where m   n, basic linear regression methods will not
be able to learn useful weights w for a linear hypothesis. Section 6.4 shows that for m   n,
linear regression will typically learn a hypothesis that perfectly predicts labels of data points
in the training set but delivers poor predictions for data points outside the training set. This
83
phenomenon is referred to as over tting and poses a main challenge for ML applications in
the high-dimensional regime.
Chapter 7 discusses basic regularization techniques that allow to prevent ML methods
from over tting. We can regularize linear regression by augmenting the squared error loss
(2.8) of a hypothesis h(w)(x) = wTx with an additional penalty term. This penalty term
depends solely on the weights w and serves as an estimate for the increase of the average loss
on data points outside the training set. Di erent ML methods are obtained from di erent
choices for this penalty term. The least absolute shrinkage and selection operator (Lasso) is
obtained from linear regression by replacing the squared error loss with the regularized loss
L
􀀀
(x; y); h(w) 
= (y 􀀀 wTx)2 +  kwk1: (3.10)
Here, the penalty term is given by the scaled norm  kwk1. The value of   can be chosen
based on some probabilistic model that interprets a data point as the realization of a RV.
The label of this data point (which is a realization of a RV) is related to its features via
y = wTx + ":
Here, w denotes some true underlying parameter vector and " is a realization of an a RV that
is independent of the features x. We need the \noise" term " since the labels of data points
collected in some ML application are typically not exactly obtained by a linear combination
wTx of its features.
The tuning of   in (3.10) can be guided by the statistical properties (such as the variance)
of the noise ", the number of non-zero entries in w and a lower bound on the non-zero values
[150, 18]. Another option for choosing the value   is to try out di erent candidate values
and pick the one resulting in smallest validation error (see Section 6.2).
3.5 Gaussian Basis Regression
Section 3.2 showed how to extend linear regression by  rst transforming the feature x using a
vector-valued feature map   : R ! Rn. The output of this feature map are the transformed
features  (x) which are fed, in turn, to a linear map h
􀀀
 (x)
 
= wT (x). Polynomial regression
in Section 3.2 has been obtained for the speci c feature map (3.6) whose entries are
the powers xl of the scalar original feature x. However, it is possible to use other functions,
di erent from polynomials, to construct the feature map  . We can extend linear regression
84
using an arbitrary feature map
 (x) = ( 1(x); : : : ;  n(x))T (3.11)
with the scalar maps  j : R ! R which are referred to as \basis functions". The choice
of basis functions depends heavily on the particular application and the underlying relation
between features and labels of the observed data points. The basis functions underlying
polynomial regression are  j(x) = xj .
Another popular choice for the basis functions are \Gaussians"
  ; (x) = exp(􀀀(1=(2 2))(x􀀀 )2): (3.12)
The family (3.12) of maps is parameterized by the variance  2 and the mean (shift)  .
Gaussian basis linear regression combines the feature map
 (x) =
􀀀
  1; 1(x); : : : ;   n; n(x)
 T
(3.13)
with linear regression (see Figure 3.5). The resulting hypothesis space is
H(n)
Gauss = fh(w) : R ! R : h(w)(x)=
Xn
j=1
wj  j; j (x)
with weights w = (w1; : : : ;wn)T 2 Rng: (3.14)
Di erent choices for the variance  2
j and shifts  j of the Gaussian function in (3.12) results
in di erent hypothesis spaces HGauss. Chapter 6.3 will discuss model selection techniques
that allow to  nd useful values for these parameters.
The hypotheses of (3.14) are parametrized by a parameter vector w 2 Rn. Each hypothesis
in HGauss corresponds to a particular choice for the parameter vector w. Thus, instead
of searching over HGauss to  nd a good hypothesis, we can search over the Euclidean space
Rn. Highly developed methods for searching over the space Rn, for a wide range of values
for n, are provided by numerical linear algebra [47].
85
^y = h(w)(x) with h(w)2H(2)
Gauss
y = h(x)
x
y
0
1
􀀀3 􀀀2 􀀀1 0 1 2 3
Figure 3.5: The true relation x 7! y (blue) between feature x and label y of data points
is highly non-linear. Therefore it seems reasonable to predict the label using a non-linear
hypothesis map h(w)(x)2H(2)
Gauss with some parameter vector w 2 R2.
3.6 Logistic Regression
Logistic regression is a ML method that allows to classify data points according to two
categories. Thus, logistic regression is a binary classi cation method that can be applied
to data points characterized by feature vectors x 2 Rn (feature space X = Rn) and binary
labels y. These binary labels take on values from a label space that contains two di erent
label values. Each of these two label values represents one of the two categories to which
the data points can belong.
It is convenient to use the label space Y = R and encode the two label values as y = 1
and y = 􀀀1. However, it is important to note that logistic regression can be used with
an arbitrary label space which contains two di erent elements. Another popular choice for
the label space is Y = f0; 1g. Logistic regression learns a hypothesis out of the hypothesis
space H(n) (see (3.1)). Note that logistic regression uses the same hypothesis space as linear
regression (see Section 3.1).
At  rst sight, it seems wasteful to use a linear hypothesis h(x) = wTx, with some
parameter vector w 2 Rn, to predict a binary label y. Indeed, while the prediction h(x) can
take any real number, the label y 2 f􀀀1; 1g takes on only one of the two real numbers 1 and
􀀀1.
It turns out that even for binary labels it is quite useful to use a hypothesis map h which
can take on arbitrary real numbers. We can always obtain a predicted label ^y 2 f􀀀1; 1g by
comparing hypothesis value h(x) with a threshold. A data point with features x, is classi ed
as ^y = 1 if h(x)   0 and ^y = 􀀀1 for h(x) < 0. Thus, we use the sign of the predictor h
to determine the  nal prediction for the label. The absolute value jh(x)j is then used to
quantify the reliability of (or con dence in) the classi cation ^y.
Consider two data points with feature vectors x(1); x(2) and a linear classi er map h
yielding the function values h(x(1)) = 1=10 and h(x(2)) = 100. Whereas the predictions for
86
both data points result in the same label predictions, i.e., ^y(1)=^y(2)=1, the classi cation of
the data point with feature vector x(2) seems to be much more reliable.
Logistic regression uses the logistic loss (2.12) to assess the quality of a particular hypothesis
h(w) 2 H(n). In particular, given some labeled training set D = fx(i); y(i)gmi
=1, logistic
regression tries to minimize the empirical risk (average logistic loss)
bL
(wjD) = (1=m)
Xm
i=1
log(1 + exp(􀀀y(i)h(w)(x(i))))
h(w)(x)=wT x
= (1=m)
Xm
i=1
log(1 + exp(􀀀y(i)wTx(i))): (3.15)
Once we have found the optimal parameter vector bw
, which minimizes (3.15), we can
classify any data point solely based on its features x. Indeed, we just need to evaluate the
hypothesis h(bw
) for the features x to obtain the predicted label
^y =
8<
:
1 if h(bw
)(x)   0
􀀀1 otherwise.
(3.16)
Since h(bw
)(x) =
􀀀
bw
 T
x (see (3.1)), the classi er (3.16) amounts to testing whether
􀀀
bw
 T
x  
0 or not.
The classi er (3.16) partitions the feature space X =Rn into two half-spaces R1 =
 
x : 􀀀
bw
 T
x 0
 
and R􀀀1=
 
x :
􀀀
bw
 T
x<0
 
which are separated by the hyperplane
􀀀
bw
 T
x = 0
(see Figure 2.9). Any data point with features x 2 R1 (x 2 R􀀀1) is classi ed as ^y = 1
(^y=􀀀1).
Logistic regression can be interpreted as a statistical estimation method for a particular
probabilistic model for the data points. This probabilistic model interprets the label y 2
f􀀀1; 1g of a data point as a RV with the probability distribution
p(y = 1;w) = 1=(1 + exp(􀀀wTx))
h(w)(x)=wT x
= 1=(1 + exp(􀀀h(w)(x)))): (3.17)
As the notation indicates, the probability (3.17) is parametrized by the parameter vector
w of the linear hypothesis h(w)(x) = wTx. Given the probabilistic model (3.17), we can
interpret the classi cation (3.16) as choosing ^y to maximize the probability p(y = ^y;w).
87
Since p(y = 1) + p(y = 􀀀1) = 1,
p(y = 􀀀1) = 1 􀀀 p(y = 1)
(3.17)
= 1 􀀀 1=(1 + exp(􀀀wTx))
= 1=(1 + exp(wTx)): (3.18)
In practice we do not know the parameter vector in (3.17). Rather, we have to estimate
the parameter vector w in (3.17) from observed data points. A principled approach to
estimate the parameter vector is to maximize the probability (or likelihood) of actually
obtaining the dataset D = f(x(i); y(i))gmi
=1 as realizations of iid data points whose labels are
distributed according to (3.17). This yields the maximum likelihood estimator
bw
= argmax
w2Rn
p(fy(i)gmi
=1)
y(i)iid
= argmax
w2Rn
Ym
i=1
p(y(i))
(3.17);(3.18)
= argmax
w2Rn
Ym
i=1
1=(1 + exp(􀀀y(i)wTx(i))): (3.19)
Note that the last expression (3.19) is only valid if we encode the binary labels using the
values 1 and 􀀀1. Using di erent label values results in a di erent expression.
Maximizing a positive function f(w) > 0 is equivalent to maximizing log f(w),
argmax
w2Rn
f(w)=argmax
w2Rn
log f(w):
Therefore, (3.19) can be further developed as
bw
(3.19)
= argmax
w2Rn
Xm
i=1
􀀀log
􀀀
1+exp(􀀀y(i)wTx(i))
 
= argmin
w2Rn
(1=m)
Xm
i=1
log
􀀀
1+exp(􀀀y(i)wTx(i))
 
: (3.20)
Comparing (3.20) with (3.15) reveals that logistic regression is nothing but maximum likelihood
estimation of the parameter vector w in the probabilistic model (3.17).
88
3.7 Support Vector Machines
SVMs are a family of ML methods for learning a hypothesis to predict a binary label y of
a data point based on its features x. Without loss of generality we consider binary labels
taking values in the label space Y = f􀀀1; 1g. A SVM uses the linear hypothesis space (3.1)
which consists of linear maps h(x) = wTx with some parameter vector w 2 Rn. Thus, the
SVM uses the same hypothesis space as linear regression and logistic regression which we
have discussed in Section 3.1 and Section 3.6, respectively. What sets the SVM apart from
these other methods is the choice of loss function.
Di erent instances of a SVM are obtained by using di erent constructions for the features
of a data point. Kernel SVMs use the concept of a kernel map to construct (typically highdimensional)
features (see Section 3.9 and [82]). In what follows, we assume the feature
construction has been solved and we have access to a feature vector x 2 Rn for each data
point.
Figure 3.6 depicts a dataset D of labeled data points, each characterized by a feature
vector x(i) 2 R2 (used as coordinates of a marker) and a binary label y(i) 2 f􀀀1; 1g (indicated
by di erent marker shapes). We can partition dataset D into two classes
C(y=1)=fx(i) : y(i)=1g, and C(y=􀀀1)=fx(i) : y(i)=􀀀1g: (3.21)
The SVM tries to learn a linear map h(w)(x) = wTx that perfectly separates the two classes
in the sense of
h
􀀀
x(i) 
| {z }
wT x(i)
> 0 for x(i) 2 C(y=1) and h
􀀀
x(i) 
| {z }
wT x(i)
< 0 for x(i) 2 C(y=􀀀1): (3.22)
We refer to a dataset, whose data points have binary labels. as linear separable if we can
 nd at least one linear map that separates in the sense of (3.22). The dataset in Figure 3.6
is linearly separable.
As can be veri ed easily, any linear map h(w)(x) = wTx achieving zero average hinge
loss (2.11) on the dataset D perfectly satis es this dataset (3.22). It seems reasonable to
learn a linear map by minimizing the average hinge loss (2.11). However, one drawback of
this approach is that there might be (in nitely) many di erent linear maps that achieve zero
average hinge loss and, in turn, perfectly separate the data points in Figure 3.6. Indeed,
consider a linear map h(w) that achieves zero average hinge loss for the D in Figure 3.6 (and
89
therefore perfectly separates it). Then, any other linear map h(w0) with weights w0 =  w,
using an arbitrary number   > 1 also achieves zero average hinge loss (and perfectly separates
the dataset).
Neither the separability requirement (3.22) nor the hinge loss (2.11) are su cient as a
sole training criterion. Indeed, there are many (if not most) datasets that are not linearly
separable. Even for a linearly separable dataset (such as the one Figure 3.6), there are
in nitely many linear maps with zero average hinge loss. Which one of these in nitely many
di erent maps should we use? To settle these issues, the SVM uses a \regularized" hinge
loss,
L
􀀀
(x; y); h(w) 
:= maxf0; 1 􀀀 y   h(w)(x)g +  



w

 
2
2
h(w)(x)=wT x
= maxf0; 1 􀀀 y   wTxg +  



w

 
2
2: (3.23)
The loss (3.7) augments the hinge loss (2.11) by the term  



w

 
2
2. This term is the scaled (by
  > 0) squared Euclidean norm of the weights w of the linear hypothesis h used to classify
data points. it can be shown that adding the term  



w

 
2
2 to the hinge loss (2.11) has an
regularization e ect.
The loss favours linear maps h(w) that are robust against (small) perturbations of the
data points. The tuning parameter   in (3.7) controls the strength of this regularization
e ect and might therefore also be referred to as a regularization parameter. We will discuss
regularization on a more general level in Chapter 7.
Let us now develop a useful geometric interpretation of the linear hypothesis obtained
by minimizing the loss function (3.7). According to [82, Chapter 2], a classi er h(wSVM) that
minimizes the average loss (3.7), maximizes the distance (margin)   between its decision
boundary and each of the two classes C(y=1) and C(y=􀀀1) (see (3.21)). The decision boundary
is given by the set of feature vectors x satisfying wT
SVMx = 0,
Making the margin as large as possible is reasonable as it ensures that the resulting
classi cations are robust against small perturbations of the features (see Section 7.2). As
depicted in Figure 3.6, the margin between the decision boundary and the classes C1 and C2
is typically determined by few data points (such as x(6) in Figure 3.6) which are closest to
the decision boundary. These data points have minimum distance to the decision boundary
and are referred to as support vectors.
We highlight that both, the SVM and logistic regression use the same hypothesis space
of linear maps. Both methods learn a linear classi er h(w) 2 H(n) (see (3.1)) whose decision
90
x(5)
x(4)
x(3)
x(6)
\support vector"
 
h(w)
x(2)
x(1)
Figure 3.6: The SVM learns a hypothesis (or classi er) h(w) with minimum average softmargin
hinge loss (3.7). Minimizing this loss is equivalent to maximizing the margin  
between the decision boundary of h(w) and each class of the training set.
boundary is a hyperplane in the feature space X = Rn (see Figure 2.9). The di erence
between SVM and logistic regression is in their choice for the loss function used to evaluate
the quality of a hypothesis h(w) 2 H(n).
The hinge loss (2.11) is a (in some sense optimal) convex approximation to the 0=1 loss
(2.9). Thus, we expect the classi er obtained by the SVM to yield a smaller classi cation
error probability p(^y 6= y) (with ^y = 1 if h(x)   0 and ^y = 􀀀1 otherwise) compared to logistic
regression which uses the logistic loss (2.12). The SVM is also statistically appealing as it
learns a robust hypothesis. Indeed, the hypothesis map with maximum margin is maximally
robust against perturbations of the feature vectors of data points. Section 7.2 discusses the
importance of robustness in ML methods in more detail.
The statistical superiority (in terms of robustness) of the SVM comes at the cost of increased
computational complexity. The hinge loss (2.11) is non-di erentiable which prevents
the use of simple gradient-based methods (see Chapter 5) and requires more advanced optimization
methods. In contrast, the logistic loss (2.12) is convex and di erentiable. Logistic
regression allows for the use of gradient-based methods to minimize the average logistic loss
incurred on a training set (see Chapter 5).
3.8 Bayes Classi er
Consider data points characterized by features x 2 X and some binary label y 2 Y. We
can use any two di erent label values but let us assume that the two possible label values
are y = 􀀀1 or y = 1. We would like to  nd (or learn) a classi er h : X ! Y such that
the predicted (or estimated) label ^y = h(x) agrees with the true label y 2 Y as much as
91
possible. Thus, it is reasonable to assess the quality of a classi er h using the 0=1 loss (2.9).
We could then learn a classi er using the ERM with the loss function (2.9). However, the
resulting optimization problem is typically intractable since the loss (2.9) is non-convex and
non-di erentiable.
Instead of solving the (intractable) ERM for 0=1 loss (2.9), we can take a di erent route
to construct a classi er. This construction is based on a simple probabilistic model for
data. Using this model, we can interpret the average 0=1 loss incurred by a hypothesis on a
training set as an approximation to the probability perr = p(y 6= h(x)). Any classi er ^h that
minimizes the error probability perr, which is the expected 0=1 loss, is referred to as a Bayes
estimator. Section 4.5 will discuss ML methods using Bayes estimator in more detail.
Let us derive the Bayes estimator for a the special case of a binary classi cation problem.
Here, data points are characterized by features x and label y 2 f􀀀1; 1g. Elementary
probability theory allows to derive the Bayes estimator, which is the hypothesis minimizing
the expected 0=1 loss, as
^h
(x) =
8<
:
1 if p(y = 1jx) > p(y = 􀀀1jx)
􀀀1 otherwise.
: (3.24)
Note that the Bayes estimator (3.24) depends on the probability distribution p(x; y)
underlying the data points.2 We obtain di erent Bayes estimators for di erent probabilistic
models. One widely used probabilistic model results in a Bayes estimator that belongs to
the linear hypothesis space (3.1). Note that this hypothesis space underlies also logistic
regression (see Section 3.6) and the SVM (see Section 3.7). Thus, logistic regression, SVM
and Bayes estimator are all examples of a linear classi er (see Figure 2.9).
A linear classi er partitions the feature space X into two half-spaces. One half-space
consists of all feature vectors x which result in the predicted label ^y = 1 and the other
half-space constituted by all feature vectors x which result in the predicted label ^y = 􀀀1.
The family of ML methods that learn a linear classi er di er in their choices for the loss
functions used to assess the quality of these half-spaces.
2Remember that we interpret data points as realizations of iid RVs with common probability distribution
p(x; y).
92
3.9 Kernel Methods
Consider a ML (classi cation or regression) problem with an underlying feature space X.
In order to predict the label y 2 Y of a data point based on its features x 2 X, we apply
a predictor h selected out of some hypothesis space H. Let us assume that the available
computational infrastructure only allows us to use a linear hypothesis space H(n) (see (3.1)).
For some applications, using a linear hypothesis h(x) = wTx is not suitable since the
relation between features x and label y might be highly non-linear. One approach to extend
the capabilities of linear hypotheses is to transform the raw features of a data point before
applying a linear hypothesis h.
The family of kernel methods is based on transforming the features x to new features
^x 2 X0 which belong to a (typically very) high-dimensional space X0 [82]. It is not uncommon
that, while the original feature space is a low-dimensional Euclidean space (e.g., X = R2),
the transformed feature space X0 is an in nite-dimensional function space.
The rationale behind transforming the original features into a new (higher-dimensional)
feature space X0 is to reshape the intrinsic geometry of the feature vectors x(i) 2 X such
that the transformed feature vectors ^x(i) have a \simpler" geometry (see Figure 3.7).
Kernel methods are obtained by formulating ML problems (such as linear regression or
logistic regression) using the transformed features ^x =  (x). A key challenge within kernel
methods is the choice of the feature map   : X ! X0 which maps the original feature vector
x to a new feature vector ^x =  (x).
93
X
x(5)
x(4)
x(3)
x(2)
x(1)
X0
^x(5)^x(4)^x(3)^x(2)
^x(1)
Figure 3.7: The data set D = f(x(i); y(i))g5i
=1 consists of 5 data points with features x(i) and
binary labels y(i). Left: In the original feature space X, the data points cannot be separated
perfectly by any linear classi er Right: The feature map   : X ! X0 transforms the features
x(i) to the new features ^x(i) =  
􀀀
x(i)
 
in the new feature space X0. In the new feature space
X0 the data points can be separated perfectly by a linear classi er.
3.10 Decision Trees
A decision tree is a 
owchart-like description of a map h : X ! Y which maps the features
x 2 X of a data point to a predicted label h(x) 2 Y [58]. While we can use decision trees
for an arbitrary feature space X and label space Y, we will discuss them for the particular
feature space X = R2 and label space Y = R.
Figure 3.8 depicts an example for a decision tree. A decision tree, consists of nodes
which are connected by directed edges. We can think of a decision tree,as a step-by-step
instruction, or a \recipe", for how to compute the function value h(x) given the features
x 2 X of a data point. This computation starts at the \root" node and ends at one of the
\leaf" nodes of the decision tree.
A leaf node ^y, which does not have any outgoing edges, represents a decision region
R^y   X in the feature space. The hypothesis h associated with a decision tree, is constant
over the regions R^y, such that h(x) = ^y for all x 2 R^y and some label value ^y 2 R.
The nodes in a decision tree are of two di erent types,
• decision (or test) nodes, which represent particular \tests" about the feature vector x,
e.g., \is the norm of x larger than 10?").
• leaf nodes, which correspond to subsets of the feature space.
The particular decision tree,depicted in Figure 3.8 consists of two decision nodes (including
94
the root node) and three leaf nodes.
Given limited computational resources, we can only use decision trees with a limited
depth. The depth of a decision tree, is the maximum number of hops it takes to reach a leaf
node starting from the root and following the arrows. The decision tree,depicted in Figure
3.8 has depth 2. We obtain an entire hypothesis space by collecting all hypothesis maps that
are obtained from the decision tree in Figure 3.8 with some vectors u and v, some positive
radius w > 0. The resulting hypothesis space is parametrized by the vectors u; v and the
number w.
To assess the quality of a particular decision tree,we can use various loss functions.
Examples of loss functions used to measure the quality of a decision tree, are the squared
error loss for numeric labels (regression) or the impurity of individual decision region for
categorical labels (classi cation).
Decision tree methods use as a hypothesis space the set of all hypotheses which represented
by a family of decision trees. Figure 3.9 depicts a collection of decision trees which
are characterized by having depth at most two. More generally, we can construct a collection
of decision trees using a  xed set of \elementary tests" on the input feature vector such as
kxk > 3, x3 < 1. These tests might also involve a continuous (real-valued) parameter such
as fx2 > wgw2[0;10]. We then build a hypothesis space by considering all decision trees not
exceeding a maximum depth and whose decision nodes carry out one of the elementary tests.
kx 􀀀 uk   w?
h(x) = h1
no
kx􀀀vk w?
h(x)=h2
no
h(x)=h3
yes
yes
R^y3 R^y2
R^y1
u v
Figure 3.8: A decision tree represents a hypothesis h which is constant on the decision region
R^y, i.e., h(x)= ^y for all x2R^y. Each decision region R^y  X corresponds to a speci c leaf
node in the decision tree.
A decision tree represents a map h : X ! Y, which is piecewise-constant over regions
of the feature space X. These non-overlapping decision regions partition the feature space
into subsets of features that are all mapped to the same predicted label. Each leaf node
of a decision tree corresponds to one particular decision region. Using large decision trees
that contain many di erent test nodes, we can learn a hypothesis with highly complicated
95
kx 􀀀 uk   r?
h(x) = 1
no
h(x) = 2
yes
kx 􀀀 uk   w?
h(x) = 1
no
kx 􀀀 vk   w?
h(x) = 10
no
h(x) = 20
yes
yes
Figure 3.9: A hypothesis space H which consists of two decision trees with depth 2 and using
the tests kx􀀀uk w and kx􀀀vk w with a  xed radius w and vectors u; v 2 Rn.
decision regions. These decision regions can be chosen such they perfectly align with almost
any given labeled dataset (see Figure 3.10).
Using a su ciently large (deep) decision tree, we can obtain a hypothesis map that closely
approximates any given non-linear map (under mild technical conditions such as Lipschitz
continuity). This is quite di erent from ML methods using the linear hypothesis space
(3.1), such as linear regression, logistic regression or the SVM. These methods learn linear
hypothesis maps with a rather simple geometry. Indeed, a linear map is constant along
hyperplanes. Moreover, the decision regions obtained from linear classi ers are always entire
half-spaces (see Figure 2.9).
x(3)
x(4)
x(2)
x(1)
x1
x2
0
1
2
3
4
5
6
0 1 2 3 4 5 6
x1 3?
x2 3?
h(x)=y(3)
no
h(x)=y(2)
yes
no
x2 3?
h(x)=y(1)
no
h(x)=y(4)
yes
yes
Figure 3.10: Using a su ciently large (deep) decision tree, we can construct a map h that
perfectly  ts any given labeled dataset f(x(i); y(i))gmi
=1 such that h(x(i))=y(i) for i = 1; : : : ;m.
96
3.11 Deep Learning
Another example of a hypothesis space uses a signal-
ow representation of a hypothesis
map h : Rn ! R. This signal-
ow representation is referred to as a ANN. As the name
indicates, an ANN is a network of interconnected elementary computational units. These
computational units might be referred to as arti cial neurons or just neurons.
Figure 3.11 depicts the simplest possible ANN that consists of a single neuron. The
neuron computes a weighted sum of the inputs and then applies an activation function  (z)
to produce the output  (z). The j-th input of a neuron is assigned a parameter or weight
wj . For a given choice of weights the ANN in Figure 3.11 represents a hypothesis map
h(w)(x) =  (z) =  
􀀀P
j wjxj
 
.
The ANN in Figure 3.11 de nes a hypothesis space that is constituted by all maps h(w)
obtained for di erent choices for the weights w in Figure 3.11. Note that the single-neuron
ANN in Figure 3.11 reduces to a linear map when we use the activation function  (z) = z.
However, even when we use a non-linear activation function in Figure 3.11, the resulting
hypothesis space is essentially the same as the space of linear maps (3.1). In particular, if
we threshold the output of the ANN in Figure 3.11 to obtain a binary label, we will always
obtain a linear classi er like logistic regression and SVM (see Section 3.6 and Section 3.7).
x1
w1
x2
w2
x3
w3
 (z)
Figure 3.11: ANN consisting of a single neuron that implements a weighted summation
z =
P
j wjxj of its inputs xj followed by applying a non-linear activation function  (z).
Deep learning methods use ANN consisting of many (thousands to millions) interconnected
neurons [49]. In principle the interconnections between neurons can be arbitrary.
One widely used approach however is to organize neurons as layers and place connections
mainly between neurons in consecutive layers [49]. Figure 3.12 depicts an example for a ANN
97
consisting of one hidden layer to represent a (parametrized) hypothesis h(w) : Rn ! R. The
input
layer
hidden
layer
output
layer
x1
x2
w1
w2
w3
w4
w5
w6
w7
w8
w9
h(w)(x)
Figure 3.12: ANN representation of a predictor h(w)(x) which maps the input (feature)
vector x = (x1; x2)T to a predicted label (output) h(w)(x). This ANN de nes a hypothesis
space consisting of all maps h(w)(x) obtained from all possible choices for the weights w =
(w1; : : : ;w9)T
 rst layer of the ANN in Figure 3.12 is referred to as the input layer. The input layer reads
in the feature vector x 2 Rn of a data point. The features xj are then multiplied with the
weights wj;j0 associated with the link between the jth input node (\neuron") with the j0th
node in the middle (hidden) layer. The output of the j0-th node in the hidden layer is given
by sj0 = (
Pn
j=1 wj;j0xj) with some (typically non-linear) activation function   : R ! R. The
argument to the activation function is the weighted combination
Pn
j=1 wj;j0sj0 of the outputs
sj of the nodes in the previous layer. For the ANN depicted in Figure 3.12, the output of
neuron s1 is  (z) with z = w1;1x1 + w1;2x2.
The hypothesis map represented by an ANN is parametrized by the weights of the connections
between neurons. Moreover, the resulting hypothesis map depends also on the
choice for the activation functions of the individual neurons. These activation function are
a design choice that can be adapted to the statistical properties of the data. However, a
few particular choices for the activation function have proven useful in many important application
domains. Two popular choices for the activation function used within ANNs are
the sigmoid function  (z) = 1
1+exp(􀀀z) or the recti ed linear unit (ReLU)  (z) = maxf0; zg.
ANNs using many, say 10, hidden layers, is often referred to as a \deep net". ML methods
98
using hypothesis spaces obtained from deep ANN (deep net)s are known as deep learning
methods [49].
It can be shown that an ANN with only one single (but arbitrarily large) hidden layer
can approximate any given map h : X ! Y = R to any desired accuracy [29]. Deep learning
methods often use a ANN with a relatively large number (more than hundreds) of hidden
layers. We refer to a ANN with a relatively large number of hidden layers as a deep net.
There is empirical and theoretical evidence that using many hidden layers, instead of few
but wide layers, is computationally and statistically favourable [34, 114] and [49, Ch. 6.4.1.].
The hypothesis map h(w) represented by an ANN can be evaluated (to obtain the predicted
label at the output) e ciently using message passing over the ANN. This message passing
can be implemented using parallel and distributed computers. Moreover, the graphical
representation of a parametrized hypothesis in the form of a ANN allows us to e ciently
compute the gradient of the loss function via a (highly scalable) message passing procedure
known as back-propagation [49]. Being able to quickly compute gradients is instrumental
for the e ciency of gradient based methods for learning a good choice for the ANN weights
(see Chapter 5).
3.12 Maximum Likelihood
For many applications it is useful to model the observed data points z(i), with i = 1; : : : ;m, as
iid realizations of a RV z with probability distribution p(z;w). This probability distribution
is parametrized by a parameter vector w 2 Rn. A principled approach to estimating the
parameter vector w based on a set of iid realizations z(1); : : : ; z(m)   p(z;w) is maximum
likelihood estimation [85].
Maximum likelihood estimation can be interpreted as an ML problem with a hypothesis
space parametrized by the parameter vector w. Each element h(w) of the hypothesis space
H corresponds to one particular choice for the parameter vector w. Maximum likelihood
methods use the loss function
L
􀀀
z; h(w) 
:= 􀀀log p(z;w): (3.25)
A widely used choice for the probability distribution p
􀀀
z;w
 
is a multivariate normal
(Gaussian) distribution with mean   and covariance matrix  , both of which constitute the
parameter vector w = ( ; ) (we have to reshape the matrix   suitably into a vector form).
99
Given the iid realizations z(1); : : : ; z(m)   p
􀀀
z;w
 
, the maximum likelihood estimates ^ , b 
of the mean vector and the covariance matrix are obtained via
^ ;b 
= argmin
 2Rn; 2Sn
+
(1=m)
Xm
i=1
􀀀log p
􀀀
z(i); ( ; )
 
: (3.26)
The optimization in (3.26) is over all possible choices for the mean vector   2 Rn and
the covariance matrix   2 Sn
+. Here, Sn
+ denotes the set of all psd Hermitian n n matrices.
The maximum likelihood problem (3.26) can be interpreted as an instance of ERM (4.3)
using the particular loss function (3.25). The resulting estimates are given explicitly as
^  = (1=m)
Xm
i=1
z(i), and b 
= (1=m)
Xm
i=1
(z(i) 􀀀 ^ )(z(i) 􀀀 ^ )T : (3.27)
Note that the expressions (3.27) are only valid when the probability distribution of the data
points is modelled as a multivariate normal distribution.
3.13 Nearest Neighbour Methods
Nearest neighbour (NN) methods are a family of ML methods that are characterized by
a speci c construction of the hypothesis space. NN methods can be applied to regression
problems involving numeric labels (e.g., using label space Y = R ) as well as for classi cation
problems involving categorical labels (e.g., with label space Y = f􀀀1; 1g).
While NN methods can be combined with arbitrary label spaces, they require the feature
space to have a speci c structure. NN methods require the feature space be a metric space
[119] that provides a measure for the distance between di erent feature vectors. We need a
metric or distance measure to determine the nearest neighbour of a data point. A prominent
example for a metric feature space is the Euclidean space Rn. The metric of Rn given by
the Euclidean distance kx􀀀x0k between two vectors x; x0 2 Rn.
Consider a training set D = f(x(i); y(i))gmi
=1 that consists of labeled data points. Thus,
for each data point we know the features and the label value. Given such a training set, NN
methods construct a hypothesis space that consist of piece-wise constant maps h : X ! Y.
For any hypothesis h in that space, the function value h(x) obtained for a data point with
features x depends only on the (labels of the) k nearest data points (smallest distance to
x) in the training set D. The number k of NNs used to determine the function value h(x)
is a design (hyper-) parameter of a NN method. NN methods are also referred to as k-NN
100
methods to make their dependence on the parameter k explicit.
Let us illustrate NN methods by considering a binary classi cation problem using an
uneven number for k (e.g., k = 3 or k = 5). The goal is to learn a hypothesis that predicts
the binary label y 2 f􀀀1; 1g of a data point based on its feature vector x 2 Rn. This
learning task can make use of a training set D containing m > k data points with known
labels. Given a data point with features x, denote by N(k) a set of k data points in D whose
feature vectors have smallest distance to x. The number of data points in N(k) whose label is
1 is denoted m(k)
1 and those with label value 􀀀1 is denoted m(k)
􀀀1. The k-NN method \learns"
a hypothesis ^h given by
^h
(x) =
8<
:
1 if m(k)
1 > m(k)
􀀀1
􀀀1 otherwise.
(3.28)
It is important to note that, in contrast to the ML methods in Section 3.1 - Section
3.11, the hypothesis space of k-NN depends on a labeled dataset (training set) D. As
a consequence, k-NN methods need to access (and store) the training set whenever the
compute a prediction (evaluate h(x)). To compute the prediction h(x) for a data point with
features x, k-NN needs to determine its NNs in the training set. When using a large training
set this implies a large storage requirement for k-NN methods. Moreover, k-NN methods
might be prone to revealing sensitive information with its predictions (see Exercise 3.8).
For a  xed k, NN methods do not require any parameter tuning. Such parameter tuning
(or learning) is required linear regression, logistic regression and deep learning methods. In
contrast, the hypothesis \learnt" by NN methods is characterized point-wise, for each possible
value of features x, by the NN in the training set. Compared to the ML methods in Section
3.1 - Section 3.11, NN methods do not require to solve (challenging) optimization problems
for model parameters. Beside their low computational requirements (put aside the memory
requirements), NN methods are also conceptually appealing as natural approximations of
Bayes estimators (see [27] and Exercise 3.9).
3.14 Deep Reinforcement Learning
Deep reinforcement learning (DRL) refers to a subset of ML problems and methods that
revolve around the control of dynamic systems such as autonomous driving cars or cleaning
robots [129, 136, 104]. A DRL problem involves data points that represent the states of a
dynamic system at di erent time instants t = 0; 1; : : :. The data points representing the
state at some time instant t is characterized by the feature vector x(t). The entries of this
101
x(i)
Figure 3.13: A hypothesis map h for k-NN with k = 1 and feature space X = R2. The
hypothesis map is constant over regions (indicated by the coloured areas) located around
feature vectors x(i) (indicated by a dot) of some data D = f(x(i); y(i))g.
feature vector are the individual features of the state at time t. These features might be
obtained via sensors, onboard-cameras or other ML methods (that predict the location of
the dynamic system). The label y(t) of a data point might represent the optimal steering
angle at time t.
DRL methods learn a hypothesis h that delivers optimal predictions ^y(t) := h
􀀀
x(t)
 
for
the optimal steering angle y(t). As their name indicates, DRL methods use hypothesis spaces
obtained from a deep net (see Section 3.11). The quality of the prediction ^y(t) obtained from
a hypothesis is measured by the loss L
􀀀􀀀
x(t); y(t)
 
; h
 
:= 􀀀r(t) with a reward signal r(t). This
reward signal might be obtained from a distance (collision avoidance) sensor or low-level
characteristics of an on-board camera snapshot.
The (negative) reward signal 􀀀r(t) typically depends on the feature vector x(t) and the
discrepancy between optimal steering direction y(t) (which is unknown) and its prediction
^y(t) := h
􀀀
x(t)
 
. However, what sets DRL methods apart from other ML methods such
as linear regression (see Section 3.1) or logistic regression (see Section 3.6) is that they
can evaluate the loss function only point-wise L
􀀀􀀀
x(t); y(t)
 
; h
 
for the speci c hypothesis
h that has been used to compute the prediction ^y(t) := h
􀀀
x(t)
 
at time instant t. This is
fundamentally di erent from linear regression that uses the squared error loss (2.8) which
can be evaluated for every possible hypothesis h 2 H.
102
3.15 LinUCB
ML methods are instrumental for various recommender systems [86]. A basic form of a
recommender system amount to chose at some time instant t the most suitable item (product,
song, movie) among a  nite set of alternatives a = 1; : : : ;A. Each alternative is characterized
by a feature vector x(t;a) that varies between di erent time instants.
The data points arising in recommender systems might represent time instants t at which
recommendations are computed. The data point at time t is characterized by a feature vector
x(t) =
􀀀􀀀
x(t;1) T
; : : : ;
􀀀
x(t;A) T  T
: (3.29)
The feature vector x(t) is obtained by stacking the feature vectors of alternatives at time
t into a single long feature vector. The label of the data point t is a vector of rewards
y(t) :=
􀀀
r(t)
1 ; : : : ; r(t)
A
 T
2 RA. The entry r(t)
a represents the reward obtained by choosing
(recommending) alternative a (with features x(t;a)) at time t. We might interpret the reward
r(t;a) as an indicator if the costumer actually buys the product corresponding to the
recommended alternative a.
The ML method LinUCB (the name seems to be inspired by the terms \linear" and
\upper con dence bound" (UCB)) aims at learning a hypothesis h that allows to predict the
rewards y(i) based on the feature vector x(t) (3.29). As its hypothesis space H, LinUCB uses
the space of linear maps from the stacked feature vectors RnA to the space of reward vectors
RA. This hypothesis space can be parametrized by matrices W 2 RA nA. Thus, LinUCB
learns a hypothesis that computes predicted rewards via
by
(t) := Wx(t): (3.30)
The entries of by
(t) =
􀀀
^r(t)
1 ; : : : ; ^r(t)
A
 
are predictions of the individual rewards r(t;a). It seems
natural to recommend at time t the alternative a whose predicted reward is maximum.
However, it turns out that this approach is sub-optimal as it prevents the recommender
system from learning the optimal predictor map W.
Loosely speaking, LinUCB tries out (explores) each alternative a 2 f1; : : : ;Ag su ciently
often to obtain a su cient amount of training data for learning a good weight matrix W.
At time t, LinUCB chooses the alternative a(t) that maximizes the quantity
^r(t)
a + R(t; a) , a = 1; : : : ; A: (3.31)
103
We can think of the component R(t; a) as a form of con dence interval. It is constructed
such that (3.31) upper bounds the actual reward r(t)
a with a prescribed level of con dence
(or probability). The con dence term R(t; a) depends on the feature vectors x(t0;a) of the
alternative a at previous time instants t0 < t. Thus, at each time instant t, LinUCB chooses
the alternative a that results in the largest upper con dence bound (UCB) (3.31) on the
reward (hence the \UCB" in LinUCB). We refer to the relevant literature on sequential
learning (and decision making) for more details on the LinUCB [86].
3.16 Exercises
Exercise 3.1. Logistic loss and Accuracy. Section 3.6 discussed logistic regression as a
ML method that learns a linear hypothesis map by minimizing the logistic loss (3.15). The
logistic loss has computationally pleasant properties as it is smooth and convex. However,
in some applications we might be ultimately interested in the accuracy or (equivalently) the
average 0=1 loss (2.9). Can we upper bound the average 0=1 loss using the average logistic
loss incurred by a given hypothesis on a given training set?
Exercise 3.2. How Many Neurons? Consider a predictor map h(x) which is piecewise
linear and consisting of 1000 pieces. Assume we want to represent this map by an ANN
using neurons with one hidden layer of neurons having a ReLU activation function. The
output layer consists of a single neuron with linear activation function. How many neurons
must the ANN contain at least ?
Exercise 3.3. E ective Dimension of ANN. Consider a ANN with n = 10 input
neurons following by three hidden layers consisting of 4, 9 and 3 nodes. The three hidden
layers are followed by the output layer consisting of a single neuron. Assume that all neurons
use a linear activation function and no bias term. What is the e ective dimension de  (H) of
the hypothesis space H that consists of all hypothesis maps that can be obtained from this
ANN.
Exercise 3.4. Linear Classi ers. Consider data points characterized by feature
vectors x 2 Rn and binary labels y 2 f􀀀1; 1g. We are interested in  nding a good linear
classi er which is such that the feature vectors resulting in h(x) = 1 is a half-space. Which
of the methods discussed in this chapter aim at learning a linear classi er?
104
Exercise 3.5. Data Dependent Hypothesis space. Consider a ML application
involving data points with features x 2 R6 and a numeric label y 2 R. We learn a hypothesis
by minimizing the average loss incurred on a training set D =
 􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
  
.
Which of the following ML methods uses a hypothesis space that depends on the dataset D?
• logistic regression
• linear regression
• k-NN
Exercise 3.6. Triangle. Consider the ANN in Figure 3.12 using the ReLU activation
function (see Figure 3.11). Show that there is a particular choice for the weights w =
(w1; : : : ;w9)T such that the resulting hypothesis map h(w)(x) is a triangle as depicted in
Figure 3.14. Can you also  nd a choice for the weights w = (w1; : : : ;w9)T that produce
the same triangle shape if we replace the ReLU activation function with the linear function
 (z) = 10   z?
x
h(x)
0
1
􀀀3 􀀀2 􀀀1 0 1 2 3
Figure 3.14: A hypothesis map h : R ! R with the shape of a triangle.
Exercise 3.7. Approximate Triangle with Gaussians Try to approximate the
hypothesis map depicted in Figure 3.14 by an element of HGauss (see (3.14)) using   = 1=10,
n = 10 and  j = 􀀀1 + (2j=10).
Exercise 3.8. Privacy Leakage in k-NN Consider a k-NN method for a binary
classi cation problem. We use k = 1 and a given training set whose data points characterize
humans. Each human is characterized by a feature vector and label that indicates sensitive
information (e.g., some sickness). Assume that you have access to the feature vectors of the
data points in the training set but not to their labels. Can you infer the label value of a data
point in the training set based on the prediction that you obtained based on your feature
vector?
105
Exercise 3.9. k-NN Approximates Bayes estimator Consider a binary classi -
cation problem involving data points that are characterized by feature vectors x 2 Rn and
binary labels y 2 f􀀀1; 1g. We have access to a labeled training set D of size m. Show that
the k-NN hypothesis (3.28) is obtained from the Bayes estimator (3.24) by approximating
or estimating the conditional probability distribution p(xjy) via the density estimator [13,
Sec. 2.5.2.]
^p(x jy) := (k=m)
1
vol(Rk)
: (3.32)
Here, vol(R) denotes the volume of a ball with radius R and Rk is the distance between x
and the kth nearest feature vector of a data point in D.
106
Chapter 4
Empirical Risk Minimization
predictor h 2 H
expected loss (or risk)
empirical risk (or training error)
Figure 4.1: ML methods learn a hypothesis h 2 H that incur small loss when predicting
the label y of a data point based on its features x. ERM approximates the expected loss
or risk by the empirical risk (solid curve) incurred on a  nite set of labeled data points
(the training set). Note that we can compute the empirical risk based on the observed data
points. However, to compute the risk we would need to know the underlying probability
distribution which is rarely the case.
Chapter 2 discussed three components (see Figure 2.1):
• data points characterized by features x 2 X and labels y 2 Y,
• a hypothesis space H of computationally feasible maps h : X ! Y,
107
• and a loss function L
􀀀
(x; y); h
 
that measures the discrepancy between the predicted
label h(x) and the true label y.
Ideally we would like to learn a hypothesis h 2 H such that L
􀀀
(x; y); h
 
is small for any
data point (x; y). However, to implement this informal goal we need to de ne what is meant
precisely by \any data point". Maybe the most widely used approach to the de ne the
concept of \any data point" is the i.i.d. assumption.
The i.i.d. assumption interprets data points as realizations of iid RVs with a common
probability distribution p(x; y). The probability distribution p(x; y) allows us to de ne the
risk of a hypothesis h as the expectation of the loss incurred by h on (the realizations of) a
random data point. We can interpret the risk of a hypothesis as a measure for its quality in
predicting the label of\any data point".
If we know the probability distribution p(x; y) from which data points are drawn (iid),
we can precisely determine the hypothesis with minimum risk. This optimal hypothesis,
which is referred to as a Bayes estimator, can be read o  almost directly from the posterior
probability distribution p(yjx) of the label y given the features x of a data point. The precise
form of the Bayes estimator depends on the choice for the loss function. When using the
squared error loss, the optimal hypothesis (or Bayes estimator) is given by the posterior
mean h(x) = E
 
yjxg [85].
In most ML application, we do not know the true underlying probability distribution
p(x; y) from which data points are generated. Therefore, we cannot compute the Bayes
estimator exactly. However, we can approximately compute this estimator by replacing the
exact probability distribution with an estimate or approximation. Section 4.5 discusses a
speci c ML method that implements this approach.
The risk of the Bayes estimator (which is the Bayes risk) provides a useful baseline against
which we can compare the average loss incurred by a ML method on a set of data points.
Section 6.6 shows how to diagnose ML methods by comparing its average loss of a hypothesis
on a training set and its average loss on a validation set with a baseline.
Section (4.1) motivates ERM by approximating the risk using the empirical risk (or
average loss) computed for a set of labeled (training) data points (see Figure 4.1). This
approximation is justi ed by the law of large numbers which characterizes the deviation
between averages of RVs and their expectation. Section 4.2 discusses the statistical and
computational aspects of ERM. We then specialize the ERM for three particular ML methods
arising from di erent combinations of hypothesis space and loss function. Section 4.3
discusses ERM for linear regression (see Section 3.1). Here, ERM amounts to minimizing a
108
di erentiable convex function, which can be done e ciently using gradient-based methods
(see Chapter 5).
We then discuss in Section 4.4 the ERM obtained for decision tree models. The resulting
ERM problems becomes a discrete optimization problem which are typically much harder
than convex optimization problems. We cannot apply gradient-based methods to solve the
ERM for decision trees. To solve ERM for a decision tree, we essentially must try out all
possible choices for the tree structure [64].
Section 4.5 considers the ERM obtained when learning a linear hypothesis using the
0=1 loss for classi cation problems. The resulting ERM amounts to minimizing a nondi
 erentiable and non-convex function. Instead of applying optimization methods to solve
this ERM instance, we will instead directly construct approximations of the Bayes estimator.
Section 4.6 decomposes the operation of ML methods into training periods and inference
periods. The training period amounts to learning a hypothesis by solving the ERM on a
given training set. The resulting hypothesis is then applied to new data points, which are
not contained in the training set. This application of a learnt hypothesis to data points
outside the training set is referred to as the inference period. Section 4.7 demonstrates how
an online learning method can be obtained by solving the ERM sequentially as new data
points come in. Online learning methods alternate between training and inference periods
whenever new data is collected.
4.1 Approximating Risk by Empirical Risk
The data points arising in many important application domains can be modelled (or approximated)
as realizations of iid RVs with a common (joint) probability distribution p(x; y) for
the features x and label y. The probability distribution p(x; y) used in this i.i.d. assumption
allows us to de ne the expected loss or risk of a hypothesis h 2 H as
E
 
L
􀀀
(x; y); h
 
g: (4.1)
It seems reasonable to learn a hypothesis h such that its risk (4.1) is minimal,
h  := argmin
h2H
E
 
L
􀀀
(x; y); h
 
g: (4.2)
We refer to any hypothesis h  that achieves the minimum risk (4.2) as a Bayes estimator
[85]. Note that the Bayes estimator h  depends on both, the probability distribution p(x; y)
109
and the loss function. When using the squared error loss (2.8) in (4.2), the Bayes estimator
h  is given by the posterior mean of y given the features x (see [109, Ch. 7]).
Risk minimization (4.2) cannot be used for the design of ML methods whenever we do
not know the probability distribution p(x; y). If we do not know the probability distribution
p(x; y), which is the rule for many ML applications, we cannot evaluate the expectation in
(4.1). One exception to this rule is if the data points are synthetically generated by drawing
realizations from a given probability distribution p(x; y).
The idea of ERM is to approximate the expectation in (4.2) with an average loss (the
empirical risk) incurred on a given set of data points (the \training set"),
D =
 􀀀
x(1); y(1) 
; : : : ;
􀀀
x(m); y(m) 
g:
As discussed in Section 2.3.4, this approximation is justi ed by the law of large numbers.
We obtain ERM by replacing the risk in the minimization problem (4.2) with the empirical
risk (2.16),
^h
2 argmin
h2H
bL
(hjD)
(2.16)
= argmin
h2H
(1=m)
Xm
i=1
L
􀀀
(x(i); y(i)); h
 
: (4.3)
As the notation in (4.3) indicates there might be several di erent hypotheses that minimize
bL
(hjD). We denote by ^h any of them. Mathematically, ERM (4.3) is just an instance of an
optimization problem [15]. The optimization domain in (4.3) is the hypothesis space H of a
ML method, the objective (or cost) function is the empirical risk (2.16). ML methods that
learn a hypothesis via ERM (4.3) are instances of optimization algorithms [133].
ERM (4.3) is a form of \learning by trial and error". A (hypothetical) instructor (or
supervisor) provides us the labels y(i) for the data points in D which are characterized by
features x(i). This dataset serves as a training set in the following sense. We use a current
guess for a good hypothesis h to predict the labels y(i) of the data points in D only from their
features x(i) . We then determine average loss bL
(hjD) that is incurred by the predictions
^y(i) = h
􀀀
x(i)
 
. If the training error bL
(hjD) is too large, we try out another hypothesis map
h0 di erent from h with the hope of achieving a smaller training error bL
(h0jD).
We highlight that ERM (4.3) is motivated by the law of large numbers. The law of large
numbers, in turn, is only useful if the data points generated within an ML application can
be well modelled as realizations of iid RVs. This i.i.d. assumption is one of the most widely
110
used working assumptions for the design and analysis of ML methods. However, there are
many important application domains involving data points that clearly violate this i.i.d.
assumption.
One example for non-i.i.d. data is time series data that consists of temporally ordered
(consecutive) data points [16, 88]. Each data point in a time series might represent a speci c
time interval or single time instants. Another example for non-i.i.d. data arises in active
learning where ML methods actively choose (or query) new data points [26]. For a third
example of non-i.i.d. data, we refer to FL applications that involve collections (networks)
of data generators with di erent statistical properties [93, 70, 74, 141, 124]. A detailed
discussion of ML methods for non-i.i.d. data is beyond the scope of this book.
4.2 Computational and Statistical Aspects of ERM
Solving the optimization problem (4.3) provides two things. First, the minimizer ^h is a
predictor which performs optimal on the training set D. Second, the corresponding objective
value bL
(^ hjD) (the \training error") can be used to estimate for the risk or expected loss of
^h
. However, as we will discuss in Chapter 7, for some datasets D, the training error bL
(^ hjD)
obtained for D can be very di erent from the expected loss (risk) of ^h when applied to new
data points which are not contained in D.
The i.i.d. assumption implies that the training error bL
(hjD) is only a noisy approximation
of the risk E
 
L
􀀀
(x; y); h
  
. The ERM solution ^h is a minimizer of this noisy approximation
and therefore in general di erent from the Bayes estimator which minimizes the risk itself.
Even if the hypothesis ^h delivered by ERM (4.3) has small training error bL
(^ hjD), it might
have unacceptably large risk E
 
L
􀀀
(x; y);^h
  
. We refer to such a situation as over tting and
will discuss techniques for detecting and avoiding it in Chapter 6.
Many important ML methods use hypotheses that are parametrized by a parameter
vector w. For each possible parameter vector, we obtain a hypothesis h(w)(x). Such a
parametrization is used in linear regression methods which learn a linear hypothesis h(w)(x) =
wTx with some parameter vector w. Another example for such a parametrization is obtained
from ANNs with the weights assigned to inputs of individual (arti cial) neurons (see Figure
3.12).
For ML methods that use a parametrized hypothesis h(w)(x), we can reformulate the
111
optimization problem (4.3) as an optimization of the parameter vector,
bw
= argmin
w2Rn
f(w) with f(w) := (1=m)
Xm
i=1
L
􀀀
(x(i); y(i)); h(w) 
| {z }
bL
􀀀
h(w)jD
 
: (4.4)
The objective function f(w) in (4.4) is the empirical risk bL
􀀀
h(w)jD
 
incurred by the hypothesis
h(w) when applied to the data points in the dataset D. The optimization problems
(4.4) and (4.3) are fully equivalent. Given the optimal parameter vector bw
solving (4.4), the
hypothesis h(bw
) solves (4.3).
We highlight that the precise shape of the objective function f(w) in (4.4) depends on
the parametrization of the hypothesis space H. The parametrization is the precise rule that
assigns a hypothesis map h(w) to a given parameter vector w. The shape of f(w) depends
also on the choice for the loss function L
􀀀
(x(i); y(i)); h(w)
 
. As depicted in Figure 4.2, the
di erent combinations of parametrized hypothesis space and loss functions can result in
objective functions with fundamentally di erent properties such that their optimization is
more or less di cult.
The objective function f(w) for the ERM obtained for linear regression (see Section
3.1) is di erentiable and convex and can therefore be minimized using simple gradient-based
methods (see Chapter 5). In contrast, the objective function f(w) of ERM obtained for
least absolute deviation regression or the SVM (see Section 3.3 and 3.7) is non-di erentiable
but still convex. The minimization of such functions is more challenging but still tractable
as there exist e cient convex optimization methods which do not require di erentiability of
the objective function [110].
The objective function f(w) obtained for ANN are typically highly non-convex with
many local minima (see Figure 4.2). The optimization of non-convex objective function is in
general more di cult than optimizing convex objective functions. However, it turns out that
despite the non-convexity, iterative gradient-based methods can still be successfully applied
to solve the resulting ERM [49]. The implementation of ERM might be even more challenging
for ML methods that use decision trees or the 0=1 loss (2.9). Indeed, the ERM obtained for
ML methods using decision trees or the 0=1 loss (2.9) involve non-di erentiable objective
functions which are harder to minimize compared with smooth functions (see Section 4.4).
112
smooth and convex
f(w)
smooth and non-convex
non-smooth and convex non-smooth and non-convex
Figure 4.2: Di erent types of objective functions that arise in ERM for di erent combinations
of hypothesis space and loss function.
4.3 ERM for Linear Regression
As discussed in Section 3.1, linear regression methods learn a linear hypothesis h(w)(x) =
wTx with minimum squared error loss (2.8). For linear regression, the ERM problem (4.4)
becomes
bw
= argmin
w2Rn
(1=m)
Xm
m=1
􀀀
y(i)􀀀wTx(i) 2
: (4.5)
Here, m = jDj denotes the sample size of the training set D. The objective function f(w)
in (4.5) is convex and smooth. Such a function can be minimized using the gradient-based
methods discussed in Chapter 5.
We can rewrite the ERM problem (4.5) more concisely by stacking the labels y(i) and
feature vectors x(i), for i = 1; : : : ;m, into a \label vector" y and \feature matrix" X,
y = (y(1); : : : ; y(m))T 2 Rm, and
X =
 
x(1); : : : ; x(m)
 T
2 Rm n: (4.6)
This allows us to rewrite the objective function in (4.5) as
(1=m)
Xm
m=1
􀀀
y(i)􀀀wTx(i) 2
= (1=m)ky 􀀀 Xwk22
: (4.7)
113
ky 􀀀 Xbw
k
fXw : w 2 Rng
y
Xbw
Figure 4.3: The ERM (4.8) for linear regression amounts to an orthogonal projection of the
label vector y =
􀀀
y(1); : : : ; y(m)
 T
on the subspace spanned by the columns of the feature
matrix X =
􀀀
x(1); : : : ; x(m)
 T
.
Inserting (4.7) into (4.5), allows to rewrite the ERM problem for linear regression as
bw
= argmin
w2Rn
(1=m)ky 􀀀 Xwk22
: (4.8)
The formulation (4.8) allows for an interesting geometric interpretation of linear regression.
Solving (4.8) amounts to  nding a vector Xw (with feature matrix X (4.6)), that is closest
(in the Euclidean norm) to the label vector y 2 Rm (4.6). The solution to this approximation
problem is precisely the orthogonal projection of the vector y onto the subspace of Rm that
is spanned by the columns of the feature matrix X (see Figure 4.3).
To solve the optimization problem (4.8), it is convenient to rewrite it as the quadratic
problem
min
w2Rn
(1=2)wTQw 􀀀 qT | {z w}
=f(w)
with Q = (1=m)XTX; q = (1=m)XTy: (4.9)
Since f(w) is a di erentiable and convex function, a necessary and su cient condition for
bw
to be a minimizer f(bw
)=minw2Rn f(w) is the zero-gradient condition [15, Sec. 4.2.3]
rf(bw
) = 0: (4.10)
Combining (4.9) with (4.10), yields the following necessary and su cient condition for a
114
parameter vector bw
to solve the ERM (4.5),
(1=m)XTXbw
= (1=m)XTy: (4.11)
This condition can be rewritten as
(1=m)XT 􀀀
y 􀀀 Xbw
 
= 0: (4.12)
As indicated in Figure 4.3, the optimality condition (4.12) requires the vector
􀀀
y 􀀀 Xbw
 
=
􀀀􀀀
y(1) 􀀀 ^y(1) 
; : : : ;
􀀀
y(m) 􀀀 ^y(m)  T
;
whose entries are the prediction errors for the data points in the training set, to be orthogonal
(or normal) to the subspace spanned by the columns of the feature matrix X. In view of
this geometric interpretation, we refer to (4.12) as a \normal equation".
It can be shown that, for any given feature matrix X and label vector y, there always
exists at least one optimal parameter vector bw
which solves (4.11). The optimal parameter
vector might not be unique, i.e., there might be several di erent parameter vectors achieving
the minimum in (4.5). However, every vector bw
which solves (4.11) achieves the same
minimum empirical risk
bL
(h(bw
) j D) = min
w2Rn
bL
(h(w) j D) = k(I 􀀀 P)yk2: (4.13)
Here, we used the orthogonal projection matrix P 2 Rm m on the linear span of the
feature matrix X = (x(1); : : : ; x(m))T 2 Rm n (see (4.6)). The linear span of a matrix
A = (a(1); : : : ; a(m)) 2 Rn m, denoted as span
 
Ag, is the subspace of Rn consisting of all
linear combinations of the columns a(r) 2 Rn of A.
If the columns of the feature matrix X (see (4.6)) are linearly independent, which implies
that the matrix XTX is invertible, the projection matrix P is given explicitly as
P = X
􀀀
XTX
 􀀀1
XT :
Moreover, the solution of (4.11) is then unique and given by
bw
=
􀀀
XTX
 􀀀1
XTy: (4.14)
115
The closed-form solution (4.14) requires the inversion of the n   n matrix XTX.
Note that formula (4.14) is only valid if the matrix XTX is invertible. The feature matrix
X is determined by the data points obtained in a ML application. Its properties are therefore
not under the control of a ML method and it might well happen that the matrix XTX is
not invertible. As a point in case, the matrix XTX cannot be invertible for any dataset
containing fewer data points than the number of features used to characterize data points
(this is referred to as high-dimensional data). Moreover, the matrix XTX is not invertible if
there two co-linear features xj ; xj0 such that xj =  xj0 holds for any data point with some
constant   2 R.
Let us now consider a dataset such that the feature matrix X is not full column-rank and,
in turn, the matrix XTX is not invertible. In this case we cannot use (4.14) to compute the
optimal parameter vector since the inverse of XTX does not exist. Moreover, in this case,
there are in nitely many di erent parameter vectors that solve (4.11), i.e., the corresponding
linear hypothesis map incurs minimum average squared error loss on the training set. Section
7.3 explains the bene ts of using weights with small Euclidean norm. The parameter vector
bw
solving the linear regression optimality condition (4.11) and having minimum Euclidean
norm among all such vectors is given by
bw
=
􀀀
XTX
 y
XTy: (4.15)
Here,
􀀀
XTX
 y
denotes the pseudoinverse (or the Moore{Penrose inverse) of XTX (see [47,
46]).
Computing the (pseudo-)inverse of XTX can be computationally challenging for large
number n of features. Figure 2.5 depicts a simple ML problem where the number of features
is already in the millions. The computational complexity of inverting the matrix XTX
depends crucially on its condition number. We refer to a matrix as ill-conditioned if its
condition number is much larger than 1. In general, ML methods do not have any control
on the condition number of the matrix XTX. Indeed, this matrix is determined solely by
the (features of the) data points fed into the ML method.
Section 5.4 will discuss a method for computing the optimal parameter vector bw
that
does not require any matrix inversion. This method, referred to as GD constructs a sequence
w(0);w(1); : : : of increasingly accurate approximations of bw
. This iterative method has two
major bene ts compared to evaluating the formula (4.14) using direct matrix inversion, such
as Gauss-Jordan elimination [47].
First, GD typically requires signi cantly fewer arithmetic operations compared to direct
116
matrix inversion. This is crucial in modern ML applications involving large feature matrices.
Second, GD does not break when the matrix X is not full rank and the formula (4.14) cannot
be used any more.
4.4 ERM for Decision Trees
Consider ERM (4.3) for a regression problem with label space Y = R and feature space
X = Rn and the hypothesis space de ned by decision trees(see Section 3.10). In stark
contrast to ERM for linear regression or logistic regression, ERM for decision trees amounts
to a discrete optimization problem. Consider the particular hypothesis space H depicted
in Figure 3.9. This hypothesis space contains a  nite number of di erent hypothesis maps.
Each individual hypothesis map corresponds to a particular decision tree.
For the small hypothesis space H in Figure 3.9, ERM is easy. Indeed, we just have to
evaluate the empirical risk (\training error") bL
(h) for each hypothesis in H and pick the one
yielding the smallest empirical risk. However, when allowing for a very large (deep) decision
tree, the computational complexity of exactly solving the ERM becomes intractable [65]. A
popular approach to learn a decision tree is to use greedy algorithms which try to expand
(grow) a given decision tree by adding new branches to leaf nodes in order to reduce the
average loss on the training set (see [66, Chapter 8] for more details).
The idea behind many decision tree learning methods is quite simple: try out expanding
a decision tree by replacing a leaf node with a decision node (implementing
another \test" on the feature vector) in order to reduce the overall empirical risk
much as possible.
Consider the labeled dataset D depicted in Figure 4.4 and a given decision tree for
predicting the label y based on the features x. We might  rst try a hypothesis obtained
from the simple tree shown in the top of Figure 4.4. This hypothesis does not allow to
achieve a small average loss on the training set D. Therefore, we might grow the tree by
replacing a leaf node with a decision node. According to Figure 4.4, to so obtained larger
decision tree provides a hypothesis that is able to perfectly predict the labels of the training
set (it achieves zero empirical risk).
One important aspect of methods that learn a decision tree by sequentially growing the
tree is the question of when to stop growing. A natural stopping criterion might be obtained
from the limitations in computational resources, i.e., we can only a ord to use decision trees
117
x(3)
x(4)
x(2)
x(1)
x1
x2
0
1
2
3
4
5
6
0 1 2 3 4 5 6
x1 3?
h(x)= 
no
h(x)= 
yes
x(3)
x(4)
x(2)
x(1)
x1
x2
x1 3?
x2 3?
h(x)= 
no
h(x)= 
yes
no
h(x)= 
yes x(3)
x(4)
x(2)
x(1)
x1
x2
x1 3?
h(x)= 
no
x2 3?
h(x)= 
no
h(x)= 
yes
yes
Figure 4.4: Consider a given labeled dataset and the decision tree in the top row. We
then grow the decision tree by expanding one of its two leaf nodes. The bottom row shows
the resulting decision trees, along with their decision boundaries. Each decision tree in the
bottom row is obtained by expanding a di erent leaf node of the decision tree in the top
row.
118
up to certain maximum depth. Besides the computational limitations, we also face statistical
limitations for the maximum size of decision trees. ML methods that allow for very deep
decision trees, which represent highly complicated maps, tend to over t the training set (see
Figure 3.10 and Chapter 7). In particular, Even if a deep decision tree incurs small average
loss on the training set, it might incur large loss when predicting the labels of data points
outside the training set.
4.5 ERM for Bayes Classi ers
The family of ML methods referred to as Bayes estimator uses the 0=1 loss (2.9) to measuring
the quality of a classi er h. The resulting ERM is
^h
= argmin
h2H
(1=m)
Xm
i=1
L
􀀀
(x(i); y(i)); h
 
(2.9)
= argmin
h2H
(1=m)
Xm
i=1
I(h(x(i)) 6= y(i)): (4.16)
The objective function in this optimization problem is non-di erentiable and non-convex
(see Figure 4.2). This prevents us from using gradient-based methods (see Chapter 5) to
solve (4.16).
We will now approach the ERM (4.16) via a di erent route by interpreting the data
points (x(i); y(i)) as realizations of iid RVs with the common probability distribution p(x; y).
As discussed in Section 2.3, the empirical risk obtained using 0=1 loss approximates the
error probability p(^y 6= y) with the predicted label ^y = 1 for h(x) > 0 and ^y = 􀀀1 otherwise
(see (2.10)). Thus, we can approximate the ERM (4.16) as
^h
(2.10)
  argmin
h2H
p(^y 6= y): (4.17)
Note that the hypothesis h, which is the optimization variable in (4.17), enters into the
objective function of (4.17) via the de nition of the predicted label ^y, which is ^y = 1 if
h(x) > 0 and ^y = 􀀀1 otherwise.
It turns out that if we would know the probability distribution p(x; y), which is required
to compute p(^y 6= y), the solution of (4.17) can be found via elementary Bayesian decision
theory [115]. In particular, the optimal classi er h(x) is such that ^y achieves the maximum
\a-posteriori" probability p(^yjx) of the label being ^y, given (or conditioned on) the features
119
x.
Since we typically do not know the probability distribution p(x; y), we have to estimate
(or approximate) it from the observed data points (x(i); y(i)). This estimation is feasible if
the data points can be considered (approximated) as realizations of iid RVs with a common
probability distribution p(x; y). We can then estimate (the parameters) of the probability
distribution p(x; y) using maximum likelihood methods (see Section 3.12). For numeric features
and labels, a widely-used parametric probability distribution p(x; y) is the multivariate
normal (Gaussian) distribution. In particular, conditioned on the label y, the feature vector
x is a Gaussian random vector with mean  y and covariance  ,
p(xjy) = N(x; y; ):1 (4.18)
The conditional expectation of the features x, given (conditioned on) the label y of a
data point, is  1 if y = 1, while for y = 􀀀1 the conditional mean of x is  􀀀1. In contrast,
the conditional covariance matrix   = Ef(x 􀀀  y)(x 􀀀  y)T jyg of x is the same for both
values of the label y 2 f􀀀1; 1g. The conditional probability distribution p(xjy) of the feature
vector, given the label y, is multivariate normal. In contrast, the marginal distribution of
the features x is a Gaussian mixture model (GMM). We will revisit GMMs later in Section
8.2 where we will see that they are a great tool for soft clustering.
For this probabilistic model of features and labels, the optimal classi er minimizing the
error probability p(^y 6= y) is ^y =1 for h(x)>0 and ^y =􀀀1 for h(x) 0 using the classi er
map
h(x) = wTx with w =  􀀀1( 1 􀀀  􀀀1): (4.19)
Carefully note that this expression is only valid if the matrix   is invertible.
We cannot implement the classi er (4.19) directly, since we do not know the true values
of the class-speci c mean vectors  1,  􀀀1 and covariance matrix  . Therefore, we have
to replace those unknown parameters with some estimates ^ 1, ^ 􀀀1 and b 
. A principled
1We use the shorthand N(x; ; ) to denote the probability density function (pdf)
p(x) =
1 p
det(2  )
exp
􀀀
􀀀 (1=2)(x􀀀 )T 􀀀1(x􀀀 )
 
of a Gaussian random vector x with mean   = Efxg and covariance matrix   = E
 
(x􀀀 )(x􀀀 )T
 
.
120
approach is to use the maximum likelihood estimates (see (3.27))
^ 1 = (1=m1)
Xm
i=1
I(y(i) = 1)x(i);
^ 􀀀1 = (1=m􀀀1)
Xm
i=1
I(y(i) = 􀀀1)x(i);
^  = (1=m)
Xm
i=1
x(i);
and b 
= (1=m)
Xm
i=1
(z(i) 􀀀 ^ )(z(i) 􀀀 ^ )T ; (4.20)
with m1 =
Pm
i=1 I(y(i) = 1) denoting the number of datapoints with label y = 1 (m􀀀1
is de ned similarly). Inserting the estimates (4.20) into (4.19) yields the implementable
classi er
h(x) = wTx with w = b 
􀀀1( ^ 1 􀀀 ^ 􀀀1): (4.21)
We highlight that the classi er (4.21) is only well-de ned if the estimated covariance matrix
b 
(4.20) is invertible. This requires to use a su ciently large number of training datapoints
such that m   n.
We derived the classi er (4.21) as an approximate solution to the ERM (4.16). The
classi er (4.21) partitions the feature space Rn into two half-spaces. One half-space consists
of feature vectors x for which the hypothesis (4.21) is non-negative and, in turn, ^y = 1.
The other half-space is constituted by feature vectors x for which the hypothesis (4.21) is
negative and, in turn, ^y = 􀀀1. Figure 2.9 illustrates these two half-spaces and the decision
boundary between them.
The Bayes estimator (4.21) is another instance of a linear classi er like logistic regression
and the SVM. Each of these methods learns a linear hypothesis h(x) = wTx, whose decision
boundary (vectors x with h(x) = 0) is a hyperplane (see Figure 2.9). However, these
methods use di erent loss functions for assessing the quality of a particular linear hypothesis
h(x) = wx (which de ned the decision boundary via h(x) = 0). Therefore, these three
methods typically learn classi ers with di erent decision boundaries.
For the estimator b 
(3.27) to be accurate (close to the unknown covariance matrix) we
need a number of datapoints (sample size) which is at least of the order n2. This sample size
requirement might be infeasible for applications with only few datapoints available.
The maximum likelihood estimate b 
(4.20) is not invertible whenever m < n. In this case,
121
the expression (4.21) becomes useless. To cope with small sample size m < n we can simplify
the model (4.18) by requiring the covariance to be diagonal   = diag( 2
1; : : : ;  2n
). This is
equivalent to modelling the individual features x1; : : : ; xn of a data point as conditionally
independent, given its label y. The resulting special case of a Bayes estimator is often referred
to as a \naive Bayes" classi er.
We  nally highlight that the classi er (4.21) is obtained using the generative model (4.18)
for the data. Therefore, Bayes estimator belong to the family of generative ML methods
which involve modelling the data generation. In contrast, logistic regression and the SVM
do not require a generative model for the data points but aim directly at  nding the relation
between features x and label y of a data point. These methods belong therefore to the family
of discriminative ML methods.
Generative methods such as those learning a Bayes estimator are preferable for applications
with only very limited amounts of labeled data. Indeed, having a generative model
such as (4.18) allows us to synthetically generate more labeled data by generating random
features and labels according to the probability distribution (4.18). We refer to [105] for a
more detailed comparison between generative and discriminative methods.
4.6 Training and Inference Periods
Some ML methods repeat the cycle in Figure 1 in a highly irregular fashion. Consider a
large image collection which we use to learn a hypothesis about how cat images look like.
It might be reasonable to adjust the hypothesis by  tting a model to the image collection.
This  tting or training amounts to repeating the cycle in Figure 1 during some speci c time
period (the \training time") for a large number.
After the training period, we only apply the hypothesis to predict the labels of new
images. This second phase is also known as inference period and might be much longer
compared to the training period. Ideally, we would like to only have a very short training
period to learn a good hypothesis and then only use the hypothesis for inference.
4.7 Online Learning
In it most basic form, ERM requires a given set of labeled data points, which we refer to as
the training set. However, some ML methods can access data only in a sequential fashion. As
a point in case, consider time series data such as daily minimum and maximum temperatures
122
recorded by a FMI weather station. Such a time series consists of a sequence of data points
that are generated at successive time instants.
Online learning studies ML methods that learn (or optimize) a hypothesis incrementally
as new data arrives. This mode of operation is quite di erent from ML methods that learn a
hypothesis at once by solving an ERM problem. These di erent operation modes corresponds
to di erent frequencies of iterating the basic ML cycle depicted in Figure 1. Online learning
methods start a new cycle in Figure 1 whenever a new data point arrives (e.g., we have
recorded the minimum and maximum temperate of a day that just ended).
We now present an online learning variant of linear regression (see Section 3.1) which is
suitable for time series data with data points
􀀀
x(t); y(t)
 
gathered sequentially (over time).
In particular, the data points
􀀀
x(t); y(t)
 
become available (are gathered) at a discrete time
instants t = 1; 2; 3 : : :.
Let us stack the feature vectors and labels of all data points available at time t into
feature matrix X(t) and label vector y(t), respectively. The feature matrix and label vector
for the  rst three time instants are
t = 1 : X(1) :=
􀀀
x(1) T
, y(1) =
􀀀
y(1) T
, (4.22)
t = 2 : X(2) :=
􀀀
x(1); x(2) T
, y(2) =
􀀀
y(1); y(2) T
, (4.23)
t = 3 : X(3) :=
􀀀
x(1); x(2); x(3) T
, y(3 =
􀀀
y(1); y(2); y(3) T
: (4.24)
As detailed in Section 3.1, linear regression aims at learning the weights w of a linear
map h(x) := wTx such that the squared error loss
􀀀
y 􀀀 h(x)
 
is as small as possible. This
informal goal of linear regression is made precise by the ERM problem (4.5) which de nes
the optimal weights via incurring minimum average squared error loss (empirical risk) on
a given training set D. These optimal weights are given by the solutions of (4.12). When
the feature vectors of datapoints in D are linearly independent, we obtain the closed-form
expression (4.14) for the optimal weights.
Inserting the feature matrix X(t) and label vector y(t) (4.22) into (4.14), yields
bw
(t) =
􀀀􀀀
X(t) T
X(t) 􀀀1􀀀
X(t) T
y(t): (4.25)
For each time instant we can evaluate the RHS of (4.25) to obtain the parameter vector
bw
(t) that minimizes the average squared error loss over all data points gathered up to time t.
However, computing bw
(t) via direct evaluation of the RHS in (4.25) for each new time instant
t misses an opportunity for recycling computations done already at earlier time instants.
123
Let us now show how to (partially) reuse the computations used to evaluate (4.25) for
time t in the evaluation of (4.25) for the next time instant t+1. To this end, we  rst rewrite
the matrix Q(t) :=
􀀀
X(t)
 T
X(t) as
Q(t) =
Xt
r=1
x(r)􀀀
x(r) T
: (4.26)
Since Q(t+1) = Q(t)+x(t+1)
􀀀
x(t+1)
 T
, we can use a well-known identity for matrix inverses (see
[8, 94]) to obtain
􀀀
Q(t+1) 􀀀1
=
􀀀
Q(t) 􀀀1
+
􀀀
Q(t)
 􀀀1
x(t+1)
􀀀
x(t+1)
 T 􀀀
Q(t)
 􀀀1
1 􀀀
􀀀
x(t+1)
 T 􀀀
Q(t)
 􀀀1
x(t+1)
: (4.27)
Inserting (4.27) into (4.25) yields the following relation between optimal parameter vectors
at consecutive time instants t and t + 1,
bw
(t+1) = bw
(t) 􀀀
􀀀
Q(t+1) 􀀀1
x(t+1)􀀀􀀀
x(t+1) T bw
(t) 􀀀 y(t+1) 
: (4.28)
Note that neither evaluating the RHS of (4.28) nor evaluating the RHS of (4.27) requires to
actually invert a matrix of with more than one entry (we can think of a scalar number as 1 1
matrix). In contrast, evaluating the RHS (4.25) requires to invert the matrix Q(t) 2 Rn n.
We obtain an online algorithm for linear regression via computing the updates (4.28) and
(4.27) for each new time instant t. Another online method for linear regression will be
discussed at the end of Section 5.7.
4.8 Weighted ERM
Consider a ML method that uses some hypothesis space H and loss function L to measure
the quality predictions obtained from a speci c hypothesis when applied to a data point. A
principled approach to learn a useful hypothesis is via ERM (4.3) using a training set
D =
 􀀀
x(1); y(1) 
; : : : ;
􀀀
x(m); y(m)  
:
.
For some applications it might be useful to modify the ERM principle (4.3) by putting
di erent weights on the data points. In particular, for each data point
􀀀
x(i); y(i)
 
we specify
124
a non-negative weight q(i) 2 R+. Weighted ERM is obtained from ERM (4.3) by replacing
the average loss over the training set with a weighted average loss,
^h
2 argmin
h2H
Xm
i=1
q(i)L
􀀀
(x(i); y(i)); h
 
: (4.29)
Note that we obtain ERM (4.3) as the special case of weighted ERM (4.29) for the weights
q(i) = 1=m.
We might interpret the weight q(i) as a measure for the importance or relevance of the
data point
􀀀
x(i); y(i)
 
for the hypothesis ^h learnt via (4.29). The extreme case q(i) = 0
means that the data point
􀀀
x(i); y(i)
 
becomes irrelevant for learning a hypothesis via (4.29).
This could be useful if the data point
􀀀
x(i); y(i)
 
represents an outlier that violates the i.i.d.
assumption which is satis ed by most of the other data points. Thus, using suitable weights
in (4.29) could make the resulting ML method robust against outliers in the training set.
Note that we have discussed another strategy (via the choice for the loss function) to achieve
robustness against outliers in Section 3.3.
Another use-case of weighted ERM (4.29) is for applications where the risk of a hypothesis
is de ned using a probability distribution that is di erent form the probability distribution
of the data points in the training set. Thus, the data points conform to an i.i.d. assumption
with underlying probability distribution p(x; y). However, we would like to measure the
quality of a hypothesis via the expected loss or risk using a di erent probability distribution
p0(x; y),
Ep0
 
L
􀀀
(x; y); h
 
g =
Z
L
􀀀
(x; y); h
 
dp0(x; y) (4.30)
Having a di erent probability distribution p0(x; y)(6= p(x; y))) to de ne the overall quality
(risk) of a hypothesis might be bene cial for binary classi cation problems with imbalanced
data. Indeed, using the average loss (which approximates the risk under p(x; y)) might not
be a useful quality measure if one class is over-represented in the training set (see Section
2.3.4). It can be shown that, under mild conditions, the weighted average loss in (4.29)
approximates (4.30) when using the weights q(i) = p0
􀀀
x(i); y(i)
 
=p
􀀀
x(i); y(i)
 
[13, Sec. 11.1.4].
125
4.9 Exercise
Exercise 4.1. Uniqueness in Linear Regression What conditions on a training set
ensure that there is a unique optimal linear hypothesis map for linear regression?
Exercise 4.2. Uniqueness in Linear Regression II Linear regression uses the
squared error loss (2.8) to measure the quality of a linear hypothesis map. We learn the
weights w of a linear map via ERM using a training set D that consists of m = 100 data
points. Each data point is characterized by n = 5 features and a numeric label. Is there
a unique choice for the weights w that results in a linear predictor with minimum average
squared error loss on the training set D)?
Exercise 4.3. A Simple Linear Regression Problem. Consider a training set of
m datapoints, each characterized by a single numeric feature x and numeric label y. We
learn hypothesis map of the form h(x) = x + b with some bias b 2 R. Can you write
down a formula for the optimal b, that minimizes the average squared error on training data 􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
 
.
Exercise 4.4. Simple Least Absolute Deviation Problem. Consider data points
characterized by single numeric feature x and label y. We learn a hypothesis map of the
form h(x) = x + b with some bias b 2 R. Can you write down a formula for the optimal b,
that minimizes the average absolute error on training data
􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
 
.
Exercise 4.5. Polynomial Regression. Consider polynomial regression for data
points with a single numeric feature x 2 R and numeric label y. Here, polynomial regression
is equivalent to linear regression using the transformed feature vectors x =
􀀀
x0; x1; : : : ; xn􀀀1
 T
.
Given a dataset D =
􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
 
, we construct the feature matrix X = 􀀀
x(1); : : : ; x(m)
 
2 Rm m with its ith column given by the feature vector x(i). Verify that this
feature matrix is a Vandermonde matrix [45]? How is the determinant of the feature matrix
related to the features and labels of data points in the dataset D?
Exercise 4.6. Training Error is not Expected Loss. Consider a training set that
consists of data points
􀀀
x(i); y(i)
 
, for i = 1; : : : ;m = 100, that are obtained as realizations
of iid RVs. The common probability distribution of these RVs is de ned by a random data
point (x; y). The feature x of this random data point is a standard Gaussian RV with zero
mean and unit variance. The label of a data point is modelled as y = x + e with Gaussian
noise e   N(0; 1). The feature x and noise e are statistically independent. We evaluate the
126
speci c hypothesis h(x) = 0 (which outputs 0 no matter what the feature value x is) by the
training error Et = (1=m)
Pm
i=1
􀀀
y(i) 􀀀 h
􀀀
x(i)
  2
. Note that Et is the average squared error
loss (2.8) incurred by hypothesis h on the datapoints
􀀀
x(i); y(i)
 
, for i = 1; : : : ;m = 100.
What is the probability that the training error Et is at least 20 % larger than the expected
(squared error) loss E
 􀀀
y 􀀀 h(x)
 2 
? What is the mean (expected value) and variance of
the training error ?
Exercise 4.7. Optimization Methods as Filters. Let us consider a  ctional (idel)
optimization method that can be represented as a  lter F. This  lter F reads in a realvalued
objective function f( ), de ned for all parameter vectors w 2 Rn. The output of the
 lter F is another real-valued function ^ f(w) that is de ned point-wise as
^ f(w) =
8<
:
1 , if w is a local minimum of f( )
0 , otherwise.
(4.31)
Verify that the  lter F is shift or translation invariant, i.e., F commutes with a translation
f0(w) := f(w + w(o)) with an arbitrary but  xed (reference) vector w(o) 2 Rn.
Exercise 4.8. Linear Regression with Sample Weighting. Consider a linear
regression method that uses ERM to learn weights bw
of a linear hypothesis map h(x) = wTx.
The weights are learnt by minimizing the average squared error loss incurred by h on a
training set that is constituted by the data points
􀀀
x(i); y(i)
 
for i = 1; : : : ; 100. Someimtes it
is useful to assign sample-weights q(i) to the data points and learn bw
. These sample-weights
re
ect varying levels of importance or relevance of di erent data points. For simplicity we use
the sample weights q(i) = 2  2 [0; 1] for i = 1; : : : ; 50 and q(i) = 2(1􀀀 ) for i = 51; : : : ; 100.
Can you  nd a closed-form expression (similar to (4.14)) for the weights bw
( ) that minimize
the weighted average squared error f(w) := (1=50)
P50
i=1  
􀀀
y(i)􀀀wTx(i)
 2
+(1=50)
P100
i=51(1􀀀
 )
􀀀
y(i) 􀀀 wTx(i)
 2
for di erent  ?
127
Chapter 5
Gradient-Based Learning
This chapter discusses an important family of optimization methods for solving ERM (4.4)
with a parametrized hypothesis space (see Chapter 4.2). The common theme of these methods
is to construct local approximations of the objective function in (4.4). These local
approximations are obtained from the gradients of the objective function. Gradient-based
methods have gained popularity recently as an e cient technique for tuning the parameters
of deep nets within deep learning methods [49].
Section 5.1 discusses GD as the most basic form of gradient-based methods. The idea of
GD is to update the weights by locally optimizing a linear approximation of the objective
function. This update is referred to as a GD step and provides the main algorithmic primitive
of gradient-based methods. One key challenge for a good use of gradient-based methods is
the appropriate extend of the local approximations. This extent is controlled by a step
size parameter that is used in the basic GD step. Section 5.2 discusses some approaches for
choosing this step size. Section 5.3 discusses a second main challenge in using gradient-based
methods which is to decide when to stop repeating the GD steps.
Section 5.4 and Section 5.5 spell out GD for two instances of ERM arising from linear
regression and logistic regression, respectively. The bene cial e ect of data normalization
on the convergence speed of gradient-based methods is brie
y discussed in Section 5.6. As
explained in Section 5.7, the use of stochastic approximations enables gradient-based methods
for applications involving massive amounts of data (\big data"). Section 5.8 develops
some intuition for advanced gradient-based methods that exploit the information gathered
during previous iterations.
128
5.1 The Basic Gradient Step
Let us rewrite ERM (4.4) as the optimization problem
min
w2Rn
f(w) := (1=m)
Xm
i=1
L
􀀀
(x(i); y(i)); h(w) 
: (5.1)
From now on we tacitly assume that each individual loss
fi(w) := L
􀀀
(x(i); y(i)); h(w) 
(5.2)
arising in (5.1) represents a di erentiable function of the parameter vector w. Trivially, differentiability
of the components (5.2) implies di erentiability of the overall objective function
f(w) (5.1).
Two important examples of ERM involving such di erentiable loss functions are linear
regression and logistic regression. In contrast, the hinge loss (2.11) used by the SVM results
in a non-di erentiable objective function f(w) (5.1). However, it is possible to (signi cantly)
extend the scope of gradient-based methods to non-di erentiable functions by replacing the
concept of a gradient with that of a subgradient.
Gradient based methods are iterative. They construct a sequence of parameter vectors
w(0) ! w(1) : : : that hopefully converge to a minimizer w of f(w),
f(w) =   f := min
w2Rn
f(w): (5.3)
Note that there might be several di erent optimal parameter vectors w that satisfy the
optimality condition (5.3). We want the sequence generated by a gradient based method to
converge towards any of them. The vectors w(r) are (hopefully) increasingly, with increasing
iteration r, more accurate approximation for a minimizer w of (5.3).
Since the objective function f(w) is di erentiable, we can approximate it locally around
the vector w(r) using a tangent hyperplane that passes through the point
􀀀
w(r); f
􀀀
w(r)
  
2
Rn+1. The normal vector of this hyperplane is given by n = (rf
􀀀
w(r)
 
;􀀀1) (see Figure 5.1).
The  rst component of the normal vector is the gradient rf(w) of the objective function
f(w) evaluated at the point w(r). Our main use of the gradient rf
􀀀
w(r)
 
will be to construct
a linear approximation [119]
f(w)   f
􀀀
w(r) 
+
􀀀
w 􀀀 w(r) T
rf
􀀀
w(r) 
for w su ciently close to w(r): (5.4)
129
Requiring the objective function f(w) in (5.4) to be di erentiable is the same as requiring
the validity of the local linear approximation (5.4) at every possible vector w(r). It turns out
that di erentiability alone is not very helpful for the design and analysis of gradient based
methods.
Gradient based methods are most useful for  nding the minimum of di erentiable functions
f(w) that are also smooth. Informally, a di erentiable function f(w) is smooth if the
gradient rf(w) does not change too rapidly as a function of the argument w. A quantitative
version of the smoothness concept refers to a function as  -smooth if its gradient is Lipschitz
continuous with Lipschitz constant   > 0 [17, Sec. 3.2],
krf(w) 􀀀 rf(w0)k    kw 􀀀 w0k: (5.5)
Note that if a function f(w) is   smooth, it is also  0 smooth for any  0 >  . The smallest
  such that (5.5) is satis ed depends on the features and labels of data points used in (5.1)
as well as on the choice for the loss function.
f(w)
f
􀀀
w(r)
 
+
􀀀
w􀀀w(r)
 T
rf
􀀀
w(r)
 
f
􀀀
w(r)
 n
Figure 5.1: A di erentiable function f(w) can be approximated locally around a point w(r)
using a hyperplane whose normal vector n = (rf
􀀀
w(r)
 
;􀀀1) is determined by the gradient
rf
􀀀
w(r)
 
[119].
Consider a current guess or approximation w(r) for the optimal parameter vector w (5.3).
We would like to  nd a new (better) parameter vector w(r+1) that has smaller objective value
f(w(r+1)) < f
􀀀
w(r)
 
than the current guess w(r). The approximation (5.4) suggests to choose
the next guess w = w(r+1) such that
􀀀
w(r+1) 􀀀 w(r)
 T
rf
􀀀
w(r)
 
is negative. We can achieve
this by the GD step
w(r+1) = w(r) 􀀀  rf(w(r)) (5.6)
with a su ciently small step size   > 0. Figure 5.2 illustrates the GD step (5.6) which is
the elementary computation of gradient based methods.
130
rf(w(r))
􀀀 rf(w(r))
1
w
f(w)
w(r+1) w(r)
1
2
3
4
Figure 5.2: A GD step (5.6) updates a current guess or approximation w(r) for the optimum
parameter vector w (5.3) by adding the correction term 􀀀 rf(w(r)). The updated
parameter vector w(r+1) is (typically) an improved approximation of the minimizer w.
The step size   in (5.6) must be su ciently small to ensure the validity of the linear
approximation (5.4). In the context of ML, the GD step size parameter   is also referred
to as learning rate. Indeed, the step size   determines the amount of progress during a GD
step towards learning the optimal parameter vector w.
We need to emphasize that the interpretation of the step size   as a learning rate is only
useful when the step size is su ciently small. Indeed, when increasing the step size   in
(5.6) beyond a critical value (that depends on the properties of the objective function f(w)),
the iterates (5.6) move away from the optimal parameter vector w. Nevertheless, from now
on we will consequently use the term learning rate for  .
The idea of gradient-based methods is to repeat the GD step (5.6) for a su cient number
of iterations (repetitions) to obtain a su ciently accurate approximation of the optimal
parameter vector w (5.3). It turns out that this is feasible for a su ciently small learning rate
and if the objective function is smooth and convex. Section 5.2 discusses precise conditions
on the learning rate such that the iterates produced by the GD step converge to the optimum
parameter vector, i.e., limr!1 f(w(r)) = f
􀀀
w
 
.
To implement the GD step (5.6) we need to choose a useful learning rate  . Moreover,
executing the GD step (5.6) requires to compute the gradient rf(w(r)). Both tasks can
be computationally challenging as discussed in Section 5.2 and 5.7. For the objective function
(5.1) obtained in linear regression and logistic regression, we can obtain closed-form
expressions for the gradient rf(w) (see Section 5.4 and 5.5).
In general, we do not have closed-form expressions for the gradient of the objective func-
131
tion (5.1) arising from a non-linear hypothesis space. One example for such a hypothesis
space is obtained from a ANN, which is used by deep learning methods (see Section 3.11).
The empirical success of deep learning methods might be partially attributed to the availability
of an e cient algorithm for computing the gradient rf(w(r)). This algorithm is
known as back-propagation [49].
5.2 Choosing the Learning Rate
f(w(r))
f(w(r+1)) f(w(r+2))
(a)
f(w(r))
f(w(r+1))
(5.6) f(w(r+2))
(5.6)
(b)
Figure 5.3: E ect of choosing bad values for the learning rate   in the GD step(5.6). (a) If
the learning rate   in the GD step (5.6) is chosen too small, the iterations make very little
progress towards the optimum or even fail to reach the optimum at all. (b) If the learning
rate   is chosen too large, the iterates w(r) might not converge at all (it might happen that
f(w(r+1)) > f(w(r))!).
The choice of the learning rate   in the GD step (5.6) has a signi cant impact on the
performance of Algorithm 1. If we choose the learning rate   too large, the GD steps
(5.6) diverge (see Figure 5.3-(b)) and, in turn, Algorithm 1 fails to deliver a satisfactory
approximation of the optimal weights w.
If we choose the learning rate   too small (see Figure 5.3-(a)), the updates (5.6) make only
very little progress towards approximating the optimal parameter vector w. In applications
that require real-time processing of data streams, it might be possible to repeat the GD steps
only for a moderate number. Thus If the learning rate is chosen too small, Algorithm 1 will
fail to deliver a good approximation within an acceptable number of iterations (runtime of
Algorithm 1).
Finding a (nearly) optimal choice for the learning rate   of GD can be a challenging task.
Many sophisticated approaches for tuning the learning rate of gradient-based methods have
been proposed [49, Chapter 8]. A detailed discussion of these approaches is beyond the scope
of this book. We will instead discuss two su cient conditions on the learning rate which
132
guarantee the convergence of the GD iterations to the optimum of a smooth and convex
objective function (5.1).
The  rst condition applies to an objective function that is  -smooth (see (5.5)) with
known constant   (not necessarily the smallest constant such that (5.5) holds). Then, the
iterates w(r) generated by the GD step (5.6) with a learning rate
  < 2= ; (5.7)
satisfy [102, Thm. 2.1.13]
f
􀀀
w(r) 
􀀀   f  
2(f
􀀀
w(0)
 
􀀀   f)



w(0) 􀀀 w

 
2
2
2



w(0) 􀀀 w

 
2
2 + r(f
􀀀
w(0)
 
􀀀   f) (2 􀀀   )
: (5.8)
The bound (5.8) not only tells us that GD iterates converge to an optimal parameter vector
but also characterize the convergence speed or rate. The sub-optimality f
􀀀
w(r)
 
􀀀minw f(w)
in terms of objective function value decreases inversely (like \1=r") with the number r of
GD steps (5.6). Convergence bounds like (5.8) can be used to specify a stopping criterion,
i.e., to determine the number of GD steps to be computed (see Section 5.3).
The condition (5.7) and the bound (5.8) is only useful if we can verify   smoothness (5.5)
assumption for a reasonable constant  . Verifying (5.8) only for a very large   results in the
bound (5.8) being too loose (pessimistic). When we use a loose bound (5.8) to determine
the number of GD steps, we might compute an unnecessary large number of GD steps (5.6).
One elegant approach to verify if a di erentiable function f(w) is   smooth (5.5) is via
the Hessian matrix r2f(w) 2 Rn n if it exists. The entries of this Hessian matrix are the
second-order partial derivatives @f(w)
@wj@wj0
of the function f(w).
Consider an objective function f(w) (5.1) that is convex and twice-di erentiable with
psd Hessian r2f(w). If the maximum eigenvalue  max
􀀀
r2f(w)
 
of the Hessian is upper
bounded uniformly (for all w) by the constant   > 0, then f(w) is   smooth (5.5) [17]. This
implies, in turn via (5.7), the su cient condition
   
2
 max
􀀀
r2f(w)
  for all w 2 Rn (5.9)
for the GD learning rate such that the GD steps converge to the minimum of the objective
function f(w).
It is important to note that the condition (5.9) guarantees convergence of the GD steps
133
for any possible initialization w(0). Note that the usefulness of the condition (5.9) depends
on the di culty of computing the Hessian matrix r2f(w). Section 5.4 and Section 5.5 will
present closed-form expressions for the Hessian of the objective function (5.1) obtained for
linear regression and logistic regression. These closed-form expressions involve the feature
vectors and labels of the data points in the training set D =
 􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
  
used in (5.1).
While it might be computationally challenging to determine the maximum (in absolute
value) eigenvalue  max
􀀀
r2f(w)
 
for arbitrary w, it might still be feasible to  nd an upper
bound U for it. If we know such an upper bound U    max
􀀀
r2f(w)
 
(valid for all w 2 Rn),
the learning rate   = 1=U still ensures convergence of the GD steps (5.6).
Up to know we have assumed a  xed (constant) learning rate   that is used for each
repetition of the GD steps (5.6). However, it might be useful to vary or adjust the learning
rate as the GD steps (5.6) proceed. Thus, we might use a di erent learning rate  r for each
iteration r of (5.6). Such a varying learning rate is useful for a variant of GD that uses
stochastic approximation (see Section 5.7). However, we might use a varying learning rate
also to avoid the burden of verifying   smoothness (5.5) with a tight (small)  . The GD steps
(5.6) with the learning rate  r := 1=r converge to the optimal parameter vector w as long
as we can ensure a bounded gradient krf(w)k   U for a su ciently large neighbourhood
of w [102].
5.3 When To Stop?
One main challenge in the successful application of GD is to decide when to stop iterating
(or repeating) the GD step (5.6). Maybe the most simple approach is to monitor the
decrease in the objective function f(w(r)) and to stop if the decrease f(w(r􀀀1)) 􀀀 f(w(r))
falls below a threshold. However, the ultimate goal of a ML method is not to minimize the
objective function f(w) in (5.1). Indeed, the objective function represents the average loss
of a hypothesis h(w) incurred on a training set. However, the ultimate goal of a ML method
is to learn a parameter vector w such that the resulting hypothesis accurately predicts any
data point, including those outside the training set.
We will see in Chapter 6 how to use validation techniques to probe a hypothesis outside
the training set. These validation techniques provide a validation error ~ f(w) that estimates
the average loss of a hypothesis with parameter vector w. Early stopping techniques monitor
the validation error ~ f(w(r)) as the GD iterations r proceed to decide when to stop iterating.
134
Another possible stopping criterion is to use a  xed number of iterations or GD steps.
This  xed number of iterations can be chosen based on convergence bounds such as (5.8)
in order to guarantee a prescribed sub-optimality of the  nal iterate w(r). A slightly more
convenient convergence bound can be obtained from (5.8) when using the the learning rate
  = 1=  in the GD step (5.6) [17],
f
􀀀
w(r) 
􀀀   f  
2 



w(0) 􀀀 w

 
2
2
r
for r = 1; 2; : : : : (5.10)
5.4 GD for Linear Regression
We now present a gradient based method for learning the parameter vector for a linear
hypothesis (see (3.1))
h(w)(x) = wTx: (5.11)
The ERM principle tells us to choose the parameter vector w in (5.11) by minimizing the
average squared error loss (2.8)
bL
(h(w)jD)
(4.4)
= (1=m)
Xm
i=1
(y(i) 􀀀 wTx(i))2: (5.12)
The average squared error loss (5.12) is computed by applying the predictor h(w)(x) to
labeled data points in a training set D = f(x(i); y(i))gmi
=1. An optimal parameter vector w
for (5.11) is obtained as
w = argmin
w2Rn
f(w) with f(w) = (1=m)
Xm
i=1
􀀀
y(i) 􀀀 wTx(i) 2
: (5.13)
The objective function f(w) in (5.13) is convex and smooth. We can therefore use GD
(5.6) to solve (5.13) iteratively, i.e., by constructing a sequence of parameter vectors that
converge to an optimal parameter vector w. To implement GD, we need to compute the
gradient rf(w).
The gradient of the objective function in (5.13) is given by
rf(w) = 􀀀(2=m)
Xm
i=1
􀀀
y(i) 􀀀 wTx(i) 
x(i): (5.14)
135
By inserting (5.14) into the basic GD iteration (5.6), we obtain Algorithm 1.
Algorithm 1 Linear regression via GD
Input: dataset D = f(x(i); y(i))gmi
=1 ; learning rate   > 0.
Initialize: set w(0) :=0; set iteration counter r :=0
1: repeat
2: r := r + 1 (increase iteration counter)
3: w(r) := w(r􀀀1) +  (2=m)
Pm
i=1
􀀀
y(i) 􀀀
􀀀
w(r􀀀1)
 T
x(i)
 
x(i) (do a GD step (5.6))
4: until stopping criterion met
Output: w(r) (which approximates w in (5.13))
Let us have a closer look on the update in step 3 of Algorithm 1, which is
w(r) := w(r􀀀1) +  (2=m)
Xm
i=1
􀀀
y(i) 􀀀
􀀀
w(r􀀀1))Tx(i) 
x(i): (5.15)
The update (5.15) has an appealing form as it amounts to correcting the previous guess (or
approximation) w(r􀀀1) for the optimal parameter vector w by the correction term
(2 =m)
Xm
i=1
(y(i) 􀀀
􀀀
w(r􀀀1))Tx(i)) | {z }
e(i)
x(i): (5.16)
The correction term (5.16) is a weighted average of the feature vectors x(i) using weights
(2 =m)   e(i). These weights consist of the global factor (2 =m) (that applies equally to all
feature vectors x(i)) and a sample-speci c factor e(i) = (y(i)􀀀
􀀀
w(r􀀀1))Tx(i)), which is the prediction
(approximation) error obtained by the linear predictor h(w(r􀀀1))(x(i)) =
􀀀
w(r􀀀1))Tx(i)
when predicting the label y(i) from the features x(i).
We can interpret the GD step (5.15) as an instance of \learning by trial and error".
Indeed, the GD step amounts to  rst \trying out" (trial) the predictor h(x(i)) =
􀀀
w(r􀀀1))Tx(i).
The predicted values are then used to correct the weight vector w(r􀀀1) according to the error
e(i) = y(i) 􀀀
􀀀
w(r􀀀1))Tx(i).
The choice of the learning rate   used for Algorithm 1 can be based on the condition
(5.9) with the Hessian r2f(w) of the objective function f(w) underlying linear regression
(see (5.13)). This Hessian is given explicitly as
r2f(w) = (1=m)XTX; (5.17)
136
with the feature matrix X =
􀀀
x(1); : : : ; x(m)
 T
2 Rm n (see (4.6)). Note that the Hessian
(5.17) does not depend on the parameter vector w.
Comparing (5.17) with (5.9), one particular strategy for choosing the learning rate in
Algorithm 1 is to (i) compute the matrix product XTX, (ii) compute the maximum eigenvalue
 max
􀀀
(1=m)XTX
 
of this product and (iii) set the learning rate to   = 1= max
􀀀
(1=m)XTX
 
.
While it might be challenging to compute the maximum eigenvalue  max
􀀀
(1=m)XTX
 
,
it might be easier to  nd an upper bound U for it.1 Given such an upper bound U  
 max
􀀀
(1=m)XTX
 
, the learning rate   = 1=U still ensures convergence of the GD steps.
Consider a dataset f(x(i); y(i))gmi
=1 with normalized features, i.e., kx(i)k = 1 for all i =
1; : : : ;m. This implies, in turn, the upper bound U = 1, i.e., 1    max
􀀀
(1=m)XTX
 
. We
can then ensure convergence of the iterates w(r) (see (5.15)) by choosing the learning rate
  = 1.
Time-Data Tradeo s. The number of GD steps required by Algorithm 1 to ensure a
prescribed sub-optimality depends crucially on the condition number of XTX. What can
we say about the condition number? In general, we have not control over this quantity as
the matrix X consists of the feature vectors of arbitrary data points. However, it is often
useful to model the feature vectors as realizations of iid random vectors. It is then possible
to bound the probability of the feature matrix having a su ciently small condition number.
These bounds can then be used to choose the step-size such that convergence is guaranteed
with su ciently large probability. The usefulness of these bounds typically depends on the
ratio n=m. For increasing sample-size, these bounds allow to use larger step-sizes and, in
turn, result in faster convergence of GD algorithm. Thus, we obtain a trade-o  between the
runtime of Algorithm 1 and the number of data points that we feed into it [107].
5.5 GD for Logistic Regression
Logistic regression learns a linear hypothesis h(w) that is used to classify data points by
predicting their binary label. The quality of such a linear classi er is measured by the logistic
loss (2.12). The ERM principle suggest to learn the parameter vector w by minimizing the
average logistic loss (3.15) obtained for a training set D = f(x(i); y(i))gmi
=1. The training set
consists of data points with features x(i) 2 Rn and binary labels y(i) 2 f􀀀1; 1g.
1The problem of computing a full eigenvalue decomposition of XTX has essentially the same complexity
as ERM via directly solving (4.11), which we want to avoid by using the \cheaper" GD Algorithm 1.
137
We can rewrite ERM for logistic regression as the optimization problem
w = argmin
w2Rn
f(w)
with f(w) = (1=m)
Xm
i=1
log
􀀀
1+exp
􀀀
􀀀 y(i)wTx(i)  
: (5.18)
The objective function f(w) is di erentiable and therefore we can use GD (5.6) to solve
(5.18). We can write down the gradient of the objective function in (5.18) in closed-form as
rf(w) = (1=m)
Xm
i=1
􀀀y(i)
1 + exp(y(i)wTx(i))
x(i): (5.19)
Inserting (5.19) into the GD step (5.6) yields Algorithm 2.
Algorithm 2 Logistic regression via GD
Input: labeled dataset D = f(x(i); y(i))gmi
=1 containing feature vectors x(i) 2 Rn and labels
y(i) 2 R; GD learning rate   > 0.
Initialize:set w(0) :=0; set iteration counter r :=0
1: repeat
2: r :=r+1 (increase iteration counter)
3: w(r) := w(r􀀀1)+ (1=m)
Pm
i=1
y(i)
1+exp
􀀀
y(i)
􀀀
w(r􀀀1)
 T
x(i)
 x(i) (do a GD step (5.6))
4: until stopping criterion met
Output: w(r), which approximates a solution w of (5.18))
Let us have a closer look on the update in step (3) of Algorithm 2. This step amounts
to computing
w(r) := w(r􀀀1) +  (1=m)
Xm
i=1
y(i)
1 + exp
􀀀
y(i)
􀀀
w(r􀀀1)
 T
x(i)
 x(i): (5.20)
Similar to the GD step (5.15) for linear regression, also the GD step (5.20) for logistic
regression can be interpreted as an implementation of the trial-and-error principle. Indeed,
(5.20) corrects the previous guess (or approximation) w(r􀀀1) for the optimal parameter vector
w by the correction term
( =m)
Xm
i=1
y(i)
|1 + exp(y{z(i)wTx(i))}
e(i)
x(i): (5.21)
138
The correction term (5.21) is a weighted average of the feature vectors x(i). The feature
vector x(i) is weighted by the factor ( =m)   e(i). These weighting factors are a product of
the global factor ( =m) that applies equally to all feature vectors x(i). The global factor is
multiplied by a data point-speci c factor e(i) = y(i)
1+exp(y(i)wT x(i)) , which quanti es the error of
the classi er h(w(r􀀀1))(x(i)) =
􀀀
w(r􀀀1))Tx(i) for a single data point with true label y(i) 2 f􀀀1; 1g
and features x(i) 2 Rn.
We can use the su cient condition (5.9) for the convergence of GD steps to guide the
choice of the learning rate   in Algorithm 2. To apply condition (5.9), we need to determine
the Hessian r2f(w) matrix of the objective function f(w) underlying logistic regression (see
(5.18)). Some basic calculus reveals (see [58, Ch. 4.4.])
r2f(w) = (1=m)XTDX: (5.22)
Here, we used the feature matrix X =
􀀀
x(1); : : : ; x(m)
 T
2 Rm n (see (4.6)) and the diagonal
matrix D = diagfd1; : : : ; dmg 2 Rm m with diagonal elements
di =
1
1 + exp(􀀀wTx(i))
 
1 􀀀
1
1 + exp(􀀀wTx(i))
 
: (5.23)
We highlight that, in contrast to the Hessian (5.17) of the objective function arising in linear
regression, the Hessian (5.22) of logistic regression varies with the parameter vector w. This
makes the analysis of Algorithm 2 and the optimal choice for the learning rate   more
di cult compared to Algorithm 1. At least, we can ensure convergence of (5.20) (towards
a solution of (5.18)) for the learning rate   = 1 if we normalize feature vectors such that
kx(i)k = 1. This follows from the fact the diagonal entries (5.23) take values in the interval
[0; 1].
5.6 Data Normalization
The number of GD steps (5.6) required to reach the minimum (within a prescribed accuracy)
of the objective function (4.5) depends crucially on the condition number [17, 68]
 (XTX) :=  max= min: (5.24)
Here, we use the largest and smallest eigenvalue of the matrix XTX, denoted as  max and
 min, respectively. The condition number (5.24) is only well-de ned if the columns of the
139
feature matrix X (4.6) (which are the feature vectors x(i)), are linearly independent. In this
case the condition number is lower bounded as 1    (XTX).
It can be shown that the GD steps (5.6) converge faster for smaller condition number
 (XTX) [68]. Thus, GD will be faster for datasets with a feature matrix X such that
 (XTX)   1. It is therefore often bene cial to pre-process the feature vectors using a
normalization (or standardization) procedure as detailed in Algorithm 3.
Algorithm 3 \Data Normalization"
Input: labeled dataset D = f(x(i); y(i))gmi
=1
1: remove sample means bx
= (1=m)
Pm
i=1 x(i) from features, i.e.,
x(i) := x(i) 􀀀bx
for i = 1; : : : ;m
2: normalise features to have unit variance,
^x(i)
j := x(i)
j =^  for j = 1; : : : ; n and i = 1; : : : ;m
with the empirical (sample) variance ^ 2
j = (1=m)
Pm
i=1
􀀀
x(i)
j
 2
Output: normalized feature vectors f^x(i)gmi
=1
Algorithm 3 transforms the original feature vectors x(i) into new feature vectorsbx
(i) such
that the new feature matrix bX
= (bx
(1); : : : ;bx
(m))T is better conditioned than the original
feature matrix, i.e.,  (bX
TbX
) <  (XTX).
5.7 Stochastic GD
Consider the GD steps (5.6) for minimizing the empirical risk (5.1). The gradient rf(w) of
the objective function (5.1) has a particular structure. Indeed, this gradient is a sum
rf(w) = (1=m)
Xm
i=1
rfi(w) with fi(w) := L
􀀀
(x(i); y(i)); h(w) 
: (5.25)
Each component of the sum (5.25) corresponds to one particular data points (x(i); y(i)), for
i = 1; : : : ;m. We need to compute a sum of the form (5.25) for each new GD step (5.6).
Computing the sum in (5.25) can be computationally challenging for at least two reasons.
First, computing the sum is challenging for very large datasets with m in the order of billions.
Second, for datasets which are stored in di erent data centres located all over the world,
the summation would require a huge amount of network resources. Moreover, the  nite
140
transmission rate of communication networks limits the rate by which the GD steps (5.6)
can be executed.
The idea of SGD is to replace the exact gradient rf(w) (5.25) by an approximation that
is easier to compute than a direct evaluation of (5.25). The word \stochastic" in the name
SGD hints already at the use of a stochastic approximation g(w)   rf(w). It turns out
that using a gradient approximation g(w) can result in signi cant savings in computational
complexity while incurring a graceful degradation in the overall optimization accuracy. The
optimization accuracy (distance to minimum of f(w)) depends crucially on the \gradient
noise"
" := rf(w) 􀀀 g(w): (5.26)
The elementary step of most SGD methods is obtained from the GD step (5.6) by replacing
the exact gradient rf(w) with some stochastic approximation g(w),
w(r+1) = w(r) 􀀀  rg
􀀀
w(r) 
; (5.27)
As the notation in (5.27) indicates, SGD methods use a learning rate  r that varies between
di erent iterations.
To avoid accumulation of the gradient noise (5.26) during the SGD updates (5.27), SGD
methods typically decrease the learning rate  r as the iterations proceed. The precise dependence
of the learning rate  r on the iteration index r is referred to as a learning rate
schedule [49, Chapter 8]. One possible choice for the learning rate schedule is  r :=1=r [101].
Exercise 5.2 discusses conditions on the learning rate schedule that guarantee convergence
of the updates SGD to the minimum of f(w).
The approximate (\noisy") gradient g(w) can be obtained by di erent randomization
strategies. The most basic form of SGD constructs the gradient approximation g(w) by
replacing the sum (5.25) with a randomly selected component,
g(w) := rf^i
(w): (5.28)
The index ^i is chosen randomly from the set f1; : : : ;mg. The resulting SGD method then
repeats the update
w(r+1) = w(r) 􀀀  rf^i
r (w(r)); (5.29)
su ciently often. Every update (5.29) uses a \fresh" randomly chosen (drawn) index ^ir.
Formally, the indices^ir are realizations of iid RVs whose common probability distribution is
141
the uniform distribution over the index set f1; : : : ;mg.
Note that (5.29) replaces the summation over the training set during the GD step (5.6) by
randomly selecting a single component of this sum. The resulting savings in computational
complexity can be signi cant when the training set consists of a large number of data points
that might be stored in a distributed fashion (in the \cloud"). The saving in computational
complexity of SGD comes at the cost of introducing a non-zero gradient noise
"r
(5.26)
= rf(w(r)) 􀀀 g
􀀀
w(r) 
= rf(w(r)) 􀀀 rf^i
r (w): (5.30)
Mini-Batch SGD. Let us now discuss a variant of SGD that tries to reduce the approximation
error (gradient noise) (5.30) arising in the SGD step (5.29). The idea behind
this variant, referred to as mini-batch SGD, is quite simple. Instead of using only a single
randomly selected component rfi(w) (see (5.25)) for constructing a gradient approximation,
mini-batch SGD uses several randomly chosen components.
We summarize mini-batch SGD in Algorithm 4 which requires an integer batch size B as
input parameter. Algorithm 4 repeats the SGD step (5.27) using a gradient approximation
that is constructed from a randomly selected subset B = fi1; : : : ; iBg (a \batch"),
g
􀀀
w
 
= (1=B)
X
i02B
rfi0(w): (5.31)
For each new iteration of Algorithm 4, a new batch B is generated by a random generator.
Note that Algorithm 4 includes the basic SGD variant (5.29) as a special case for the batch
Algorithm 4 Mini-Batch SGD
Input: components fi(w), for i = 1; : : : ;m of objective function f(w) =
Pm
i=1 fi(w) ; batch
size B; learning rate schedule  r > 0.
Initialize: set w(0) :=0; set iteration counter r :=0
1: repeat
2: randomly select a batch B = fi1; : : : ; iBg   f1; : : : ;mg of indices that select a subset
of components fi
3: compute an approximate gradient g
􀀀
w(r)
 
using (5.31)
4: r := r + 1 (increase iteration counter)
5: w(r) := w(r􀀀1) 􀀀  rg
􀀀
w(r􀀀1)
 
6: until stopping criterion met
Output: w(r) (which approximates argminw2Rn f(w) ))
142
size B = 1. Another special case is B = m, where the SGD step 5 in Algorithm 4 becomes
an ordinary GD step (5.6).
Online Learning. A main motivation for the SGD step (5.29) is that a training set is
already collected but so large that the sum in (5.25) is computationally intractable. Another
variant of SGD is obtained for sequential (time-series) data. In particular, consider data
points that are gathered sequentially, one new data point
􀀀
x(t); y(t)
 
at each time instant
t = 1; 2; : : : :. With each new data point
􀀀
x(t); y(t)
 
we can access a new component ft(w) =
L
􀀀
(x(t); y(t)); h(w)
 
(see (5.1)). For sequential data, we can use a slight modi cation of the
SGD step (5.27) to obtain an online learning method (see Section 4.7). This online variant
of SGD computes, at each time instant t,
w(t) := w(t􀀀1) 􀀀  trft(w(t􀀀1)): (5.32)
5.8 Advanced Gradient-Based Methods
The main idea underlying GD and SGD is to approximate the objective function (5.1)
locally around a current guess or approximation w(r) for the optimal weights. This local
approximation is a tangent hyperplane whose normal vector is determined by the gradient
rf(w(r). We then obtain an updated (improved) approximation by minimizing this local
approximation, i.e., by doing a GD step (5.6).
The idea of advanced gradient methods [49, Ch. 8] is to exploit the information provided
by the gradients rf
􀀀
w(r0)
 
at previous iterations r0 = 1; : : : ; r, to build an improved local
approximation of f(w) around a current iterate w(r). Figure 5.4 indicates such an improved
local approximation of f(w) which is non-linear (e.g., quadratic). These improved local
approximations can be used to adapt the learning rate during the GD steps (5.6).
Advanced gradient-based methods use improved local approximations to modify the gradient
rf
􀀀
w(r0)
 
to obtain an improved update direction. Figure 5.5 depicts the contours
of an objective function f(w) for which the gradient rf
􀀀
w(r)
 
points only weakly towards
the optimal parameter vector w (minimizer of f(w)). The gradient history rf
􀀀
w(r0)
 
,
forr0 = 1; : : : ; r, allows to detect such an unfavourable geometry of the objective function.
Moreover, the gradient history can be used to \correct" the update direction rf
􀀀
w(r)
 
to
obtain an improved update direction towards the optimum parameter vector w.
143
w(r􀀀1)
f(w)
w(r)
Figure 5.4: Advanced gradient-based methods use the gradients at previous iterations to
construct an improved (non-linear) local approximation (dashed) of the objective function
f(w) (solid).
?
adapted
direction
GD
direction
􀀀5 f(w(r))
w(r)
Figure 5.5: Advanced gradient-based methods use improved (non-linear) local approximations
of the objective function f(w) to \nudge" the update direction towards the optimal
parameter vector w. The update direction of plain vanilla GD (5.6) is the negative gradient
􀀀rf
􀀀
w(r)
 
. For some objective functions the negative gradient might be only weakly
correlated with the straight direction from w(r) towards the optimal parameter vector (?).
144
5.9 Exercises
Exercise 5.1. Use Knowledge About Problem Class. Consider the space P of
sequences f = (f[0]; f[1]; : : :) that have the following properties
• for each sequence there is an index i(f) such that f is monotone increasing for indices
i0   i(f) and monotone decreasing for indices i0   i(f)
• any change point r of f, where f[r] 6= f[r+1] can only occrur at integer multiples of
100, e.g., r=100 or r=300.
Given a function f 2 P and starting point r0 our goal is to  nd the minimum value of
minr f[r] = f[r(f)] as quickly as possible. Can you constuct an iterative algorithm that can
access a given function f only by querying the values f[r]; f[r􀀀1] and f[r+1] for any given
index r.
Exercise 5.2. Learning rate Schedule for SGD Let us learn a linear hypothesis
h(x) = wTx using data points that arrive sequentially at discrete time instants t = 0; 1; : : :.
At time t, we gather a new data point
􀀀
x(r); y(r)
 
. The data points can be modelled as
realizations of iid copies of a RV
􀀀
x; y
 
. The probability distribution of the features x is
a standard multivariate normal distribution N(0; I). The label of a random2 data point is
related to its features via y = wTx + " with some  xed but unknown true parameter vector
w. The additive noise "   N(0; 1) follows a standard normal distribution. We use SGD to
learn the parameter vector w of a linear hypothesis,
w(t+1) = w(t) 􀀀  t
􀀀􀀀
w(t) T
x(t) 􀀀 y(t) 
x(t): (5.33)
with learning rate schedule  t =  =t
. Note that we compute a new SGD iteration (5.33)
for each new time instant t. What conditions on the hyper-parameters  ; 
 ensure that
lim
t!1
w(t) = w in distribution?
Exercise 5.3. ImageNet. The \ImageNet" database contains more than 106 images
[80]. These images are labeled according to their content (e.g., does the image show a dog?)
and stored as a  le of size at least 4 kilobytes. We want to learn a classi er that allows
to predict if an image shows a dog or not. To learn this classi er we run GD for logistic
2More precisely, a data point that is obtained as the realization of RV.
145
regression on a small computer that has 32 kilobytes memory and is connected to the internet
with bandwidth of 1 Mbit/s. Therefore, for each single GD update (5.6) it must essentially
download all images in ImageNet. How long would such a single GD update take ?
Exercise 5.4. Apple or No Apple? Consider data points being images of size of
1024   1024 pixels. Each image is characterized by the RGB pixel color intensities (value
range 0; : : : ; 255 resuling in 24 bits for each pixel), which we stack into a feature vector
x 2 Rn. We assign each image the label y = 1 if it shows an apple and y = 􀀀1 if it does not
show an apple. We use logistic regression to learn a linear hypothesis h(x) = wTx to classify
an image according to ^y = 1 if h(x)   0. The training set consists of m = 1010 labeled
images which are stored in the cloud. We implement the ML method on our own laptop
which is connected to the internet with a bandwidth of at most 100 Mbps. Unfortunately
we can only store at most  ve images on our computer. How long does it take at least to
complete one single GD step ?
Exercise 5.5. Feature Normalization To Speed Up GD Consider the dataset
with feature vectors x(1) = (100; 0)T 2 R2 and x(2) = (0; 1=10)T 2 R2 which we stack into
the matrix X = (x(1); x(2))T . What is the condition number of XTX? What is the condition
number of
􀀀bX
 TbX
with the matrix bX
= (bx
(1);bx
(2))T constructed from the normalized feature
vectors bx
(i) delivered by Algorithm 3.
Exercise 5.6. Convergence of GD Steps Consider a di erentiable objective function
f(w) whose argument is a parameter vector w 2 Rn. We make no assumption about
smoothness or convexity. Thus, the function f(w) might be non-convex and might also be
not   smooth. However, the gradient rf(w) is uniformly upper bounded krf(w)k   100
for every w. Starting from some initial vector w(0) we construct a sequence of parameter
vectors using GD steps,
w(r+1) = w(r) 􀀀  rrf
􀀀
w(r) 
: (5.34)
The learning rate  r in (5.34) is allowed to vary between di erent iterations. Can you
provide su cient conditions on the evolution of the learning rate  r, as iterations proceed,
that ensure convergence of the sequence w(0);w(1); : : : .
146
Chapter 6
Model Validation and Selection
training error validation error
baseline or benchmark
(e.g., Bayes risk, existing
ML methods or human
performance)
Figure 6.1: We can diagnose a ML method by comparing its training error with its validation
error. Ideally both are on the same level as a baseline (or benchmark error level).
Chapter 4 discussed ERM as a principled approach to learning a good hypothesis out of
a hypothesis space or model. ERM based methods learn a hypothesis ^h 2 H that incurs
minimum average loss on a set of labeled data points that serve as the training set. We refer
to the average loss incurred by a hypothesis on the training set as the training error. The
minimum average loss achieved by a hypothesis that solves the ERM might be referred to
as the training error of the overall ML method. This overall ML method is de ned by the
choice of hypothesis space (or model) and loss function (see Chapter 3).
ERM is sensible only if the training error of a hypothesis is an reliable approximation
for its loss incurred on data points outside the training set. Whether the training error of
147
a hypothesis is a reliable approximation for its loss on data points outside the training set
depends on both, the statistical properties of the data points generated by an ML application
and on the hypothesis space used by the ML method.
ML methods often use hypothesis spaces with a large e ective dimension (see Section
2.2). As an example consider linear regression (see Section 3.1) with data points having a
large number n of features (this setting is referred to as the high-dimensional regime). The
e ective dimension of the linear hypothesis space (3.1), which is used by linear regression,
is equal to the number n of features. Modern technology allows to collect a huge number
of features about individual data points which implies, in turn, that the e ective dimension
of (3.1) is large. Another example of a high-dimensional hypothesis space arises in deep
learning methods using a hypothesis space are constituted by all maps represented by an
ANN with billions of tunable parameters.
A high-dimensional hypothesis space is very likely to contain a hypothesis that perfectly
 ts any given training set. Such a hypothesis achieves a very small training error but might
incur a large loss when predicting the labels of a data point that is not included in training
set. Thus, the (minimum) training error achieved by a hypothesis learnt by ERM can be
misleading. We say that a ML method, such as linear regression using too many features,
over ts the training set when it learns a hypothesis (e.g., via ERM) that has small training
error but incurs much larger loss outside the training set.
Section 6.1 shows that linear regression will over t a training set as soon as the number
of features of a data point exceeds the size of the training set. Section 6.2 demonstrates
how to validate a learnt hypothesis by computing its average loss on data points which are
not contained in the training set. We refer to the set of data points used to validate the
learnt hypothesis as a validation set. If a ML method over ts the training set, it learns a
hypothesis whose training error is much smaller than its validation error. We can detect if
a ML method over ts by comparing its training error with its validation error (see Figure
6.1).
We can use the validation error not only to detect if a ML method over ts. The validation
error can also be used as a quality measure for the hypothesis space or model used by the
ML method. This is analogous to the concept of a loss function that allows us to evaluate
the quality of a hypothesis h 2 H. Section 6.3 shows how to select between ML methods
using di erent models by comparing their validation errors.
Section 6.4 uses a simple probabilistic model for the data to study the relation between
the training error of a learnt hypothesis and its expected loss (see (4.1)). This probabilis-
148
tic analysis reveals the interplay between the data, the hypothesis space and the resulting
training error and validation error of a ML method.
Section 6.5 discusses the bootstrap as a simulation based alternative to the probabilistic
analysis of Section 6.4. While Section 6.4 assumes a speci c probability distribution of the
data points, the bootstrap does not require the speci cation of a probability distribution
underlying the data.
As indicated in Figure 6.1, for some ML applications, we might have a baseline (or benchmark)
for the achievable performance of ML methods. Such a baseline might be obtained
from existing ML methods, human performance levels or from a probabilistic model (see
Section 6.4). Section 6.6 details how the comparison between training error, validation error
and (if available) a baseline informs possible improvements for a ML method. These improvements
might be obtained by collecting more data points, using more features of data
points or by changing the hypothesis space (or model).
Having a baseline for the expected loss, such as the Bayes risk, allows to tell if a ML
method already provides satisfactory results. If the training error and the validation error
of a ML method are close to the baseline, there might be little point in trying to further
improve the ML method.
6.1 Over tting
We now have a closer look at the occurrence of over tting in linear regression methods. As
discussed in Section 3.1, linear regression methods learn a linear hypothesis h(x) = wTx
which is parametrized by the parameter vector w 2 Rn. The learnt hypothesis is then used
to predict the numeric label y 2 R of a data point based on its feature vector x 2 Rn. Linear
regression aims at  nding a parameter vector bw
with minimum average squared error loss
incurred on a training set
D =
 􀀀
x(1); y(1) 
; : : : ;
􀀀
x(m); y(m)  
:
The training set D consists of m data points
􀀀
x(i); y(i)
 
, for i = 1; : : : ;m, with known
label values y(i). We stack the feature vectors x(i) and labels y(i), respectively, of the data
points in the training set into the feature matrix X = (x(1); : : : ; x(m))T and label vector
y = (y(1); : : : ; y(m))T .
The ERM (4.13) of linear regression is solved by any parameter vector bw
that solves
149
(4.11). The (minimum) training error of the hypothesis h(bw
) is obtained as
bL
(h(bw
) j D)
(4.4)
= min
w2Rn
bL
(h(w)jD)
(4.13)
=



(I 􀀀 P)y

 
2
2: (6.1)
Here, we used the orthogonal projection matrix P on the linear span
spanfXg =
 
Xa : a 2 Rn 
  Rm;
of the feature matrix X = (x(1); : : : ; x(m))T 2 Rm n.
In many ML applications we have access to a huge number of individual features to
characterize a data point. As a point in case, consider a data point which is a snapshot
obtained from a modern smartphone camera. These cameras have a resolution of several
megapixels. Here, we can use millions of pixel colour intensities as its features. For such
applications, it is common to have more features for data points than the size of the training
set,
n   m: (6.2)
Whenever (6.2) holds, the feature vectors x(1); : : : ; x(m) 2 Rn of the data points in D are
typically linearly independent. As a case in point, if the feature vectors x(1); : : : ; x(m) 2
Rn are realizations of iid RVs with a continuous probability distribution, these vectors are
linearly independent with probability one [100].
If the feature vectors x(1); : : : ; x(m) 2 Rn are linearly independent, the span of the feature
matrix X = (x(1); : : : ; x(m))T coincides with Rm which implies, in turn, P = I. Inserting
P = I into (4.13) yields
bL
(h(bw
) j D) = 0: (6.3)
As soon as the number m = jDj of training data points does not exceed the number n of
features that characterize data points, there is (with probability one) a linear predictor h(bw
)
achieving zero training error(!).
While the hypothesis h(bw
) achieves zero training error, it will typically incur a non-zero
average prediction error y􀀀h(bw
)(x) on data points (x; y) outside the training set (see Figure
6.2). Section 6.4 will make this statement more precise by using a probabilistic model for
the data points within and outside the training set.
Note that (6.3) also applies if the features x and labels y of data points are completely
unrelated. Consider an ML problem with data points whose labels y and features are real-
150
izations of a RV that are statistically independent. Thus, in a very strong sense, the features
x contain no information about the label of a data point. Nevertheless, as soon as the number
of features exceeds the size of the training set, such that (6.2) holds, linear regression
methods will learn a hypothesis with zero training error.
We can easily extend the above discussion about the occurrence of over tting in linear
regression to other methods that combine linear regression with a feature map. Polynomial
regression, using data points with a single feature z, combines linear regression with the
feature map z 7!  (z) :=
􀀀
z0; : : : ; zn􀀀1
 T
as discussed in Section 3.2.
It can be shown that whenever (6.2) holds and the features z(1); : : : ; z(m) of the training
set are all di erent, the feature vectors x(1) :=  
􀀀
z(1)
 
; : : : ; x(m) :=  
􀀀
z(m)
 
are linearly
independent. This implies, in turn, that polynomial regression is guaranteed to  nd a hypothesis
with zero training error whenever m   n and the data points in the training set
have di erent feature values.
Figure 6.2: Polynomial regression learns a polynomial map with degree n􀀀1 by minimizing its
average loss on a training set (blue crosses). Using high-degree polynomials (large n) results
in a small training error. However, the learnt high-degree polynomial performs poorly on
data points outside the training set (orange dots).
6.2 Validation
Consider an ML method that uses ERM (4.3) to learn a hypothesis ^h 2 H out of the
hypothesis space H. The discussion in Section 6.1 revealed that the training error of a learnt
151
􀀀
x(1); y(1)
 
􀀀
x(2); y(2)
 
􀀀
x(3); y(3)
 
D(train)
􀀀
x(4); y(4)
 
􀀀
x(5); y(5)
 
D(val)
Figure 6.3: We split the dataset D into two subsets, a training set D(train) and a validation
set D(val). We use the training set to learn ( nd) the hypothesis ^h with minimum empirical
risk bL
(^ hjD(train)) on the training set (4.3). We then validate ^h by computing its average
loss bL
(^ hjD(val)) on the validation set D(val). The average loss bL
(^ hjD(val)) obtained on the
validation set is the validation error. Note that ^h depends on the training set D(train) but is
completely independent of the validation set D(val).
152
hypothesis ^h can be a poor indicator for the performance of ^h for data points outside the
training set. The hypothesis ^h tends to \look better" on the training set over which it has
been tuned within ERM. The basic idea of validating the predictor ^h is simple:
•  rst we learn a hypothesis ^h using ERM on a training set and
• then we compute the average loss of ^h on data points that do not belong to the training
set.
Thus, validation means to compute the average loss of a hypothesis using data points that
have not been used in ERM to learn that hypothesis.
Assume we have access to a dataset of m data points,
D =
 􀀀
x(1); y(1) 
; : : : ;
􀀀
x(m); y(m)  
:
Each data point is characterized by a feature vector x(i) and a label y(i). Algorithm 5
outlines how to learn and validate a hypothesis h 2 H by splitting the dataset D into a
training set and a validation set. The random shu ing in step 1 of Algorithm 5 ensures the
i.i.d. assumption for the shu ed data. Section 6.2.1 shows next how the i.i.d. assumption
ensures that the validation error (6.6) approximates the expected loss of the hypothesis ^h.
The hypothesis ^h is learnt via ERM on the training set during step 4 of Algorithm 5.
6.2.1 The Size of the Validation Set
The choice of the split ratio     mt=m in Algorithm 5 is often based on trial and error.
We try out di erent choices for the split ratio and pick the one with the smallest validation
error. It is di cult to make a precise statement on how to choose the split ratio which applies
broadly [83]. This di culty stems from the fact that the optimal choice for   depends on
the precise statistical properties of the data points.
One approach to determine the required size of the validation set is to use a probabilistic
model for the data points. The i.i.d. assumption is maybe the most widely used probabilistic
model within ML. Here, we interpret data points as the realizations of iid RVs. These iid RVs
have a common (joint) probability distribution p(x; y) over possible features x and labels
y of a data point. Under the i.i.d. assumption, the validation error Ev (6.6) also becomes
a realization of a RV. The expectation (or mean) EfEvg of this RV is precisely the risk
EfL
􀀀
(x; y);^h
 
g of ^h (see (4.1)).
153
Algorithm 5 Validated ERM
Input: model H, loss function L, dataset D =
 􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
  
; split ratio  
1: randomly shu e the data points in D
2: create the training set D(train) using the  rst mt=d me data points,
D(train) =
 􀀀
x(1); y(1) 
; : : : ;
􀀀
x(mt); y(mt)  
:
3: create the validation set D(val) by the mv = m 􀀀 mt remaining data points,
D(val) =
 􀀀
x(mt+1); y(mt+1) 
; : : : ;
􀀀
x(m); y(m)  
:
4: learn hypothesis ^h via ERM on the training set,
^h
:= argmin
h2H
bL
􀀀
hjD(train) 
(6.4)
5: compute the training error
Et := bL
􀀀
^ hjD(train) 
= (1=mt)
Xmt
i=1
L
􀀀
(x(i); y(i));^h
 
: (6.5)
6: compute the validation error
Ev := bL
􀀀
^ hjD(val)  = (1=mv)
Xm
i=mt+1
L
􀀀
(x(i); y(i));^h
 
: (6.6)
Output: learnt hypothesis ^h, training error Et, validation error Ev
154
Within the above i.i.d. assumption, the validation error Ev becomes a realization of a RV
that 
uctuates around its mean EfEvg. We can quantify this 
uctuation using the variance
 2E
v := E
 􀀀
Ev 􀀀 EfEvg
 2 
:
Note that the validation error is the average of the realizations L
􀀀
(x(i); y(i));^h
 
of iid RVs.
The probability distribution of the RV L
􀀀
(x; y);^h
 
is determined by the probability distribution
p(x; y), the choice of loss function and the hypothesis ^h. In general, we do not know
p(x; y) and, in turn, also do not know the probability distribution of L
􀀀
(x; y);^h
 
.
If we know an upper bound U on the variance of the (random) loss L
􀀀
(x(i); y(i));^h
 
, we
can bound the variance of Ev as
 2E
v
  U=mv:
We can then, in turn, ensure that the variance  2E
v of the validation error Ev does not exceed
a given threshold  , say   = (1=100)E2
t , by using a validation set of size
mv   U= : (6.7)
The lower bound (6.7) is only useful if we can determine an upper bound U on the variance
of the RV L
􀀀
(x; y);^h
 
where
􀀀
x; y
 
is a RV with probability distribution p(x; y). An upper
bound on the variance of L
􀀀
(x; y);^h
 
can be derived using probability theory if we know an
accurate probabilistic model p(x; y) for the data points. Such a probabilistic model might
be provided by application-speci c scienti c  elds such as biology or psychology. Another
option is to estimate the variance of L
􀀀
(x; y);^h
 
using the sample variance of the actual loss
values L
􀀀
(x(1); y(1));^h
 
; : : : ;L
􀀀
(x(m); y(m));^h
 
obtained for the dataset D.
6.2.2 k-Fold Cross Validation
Algorithm 5 uses the most basic form of splitting a given dataset D into a training set and
a validation set. Many variations and extensions of this basic splitting approach have been
proposed and studied (see [33] and Section 6.5). One very popular extension of the single
split into training set and validation set is known as k-fold cross-validation (k-fold CV) [58,
Sec. 7.10]. We summarize k-fold CV in Algorithm 6 below.
Figure 6.4 illustrates the key principle behind k-fold CV. First, we divide the entire
dataset evenly into k subsets which are referred to as \folds". The learning (via ERM)
and validation of a hypothesis out of a given hypothesis space H is then repeated k times.
155
fold 1 D(val)=D1
fold 2 D(val)=D2
fold 3 D(val)=D3
fold 4 D(val)=D4
fold 5 D(val)=D5
dataset D =
 􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
  
Figure 6.4: Illustration of k-fold CV for k = 5. We evenly partition the entire dataset D
into k = 5 subsets (or folds) D1; : : : ;D5. We then repeat the validated ERM Algorithm 5 for
k = 5 times. The bth repetition uses the bth fold Db as the validation set and the remaining
k􀀀1(= 4) folds as the training set for ERM (4.3).
During each repetition, we use one fold as the validation set and the remaining k􀀀1 folds as
a training set. We then average the values of the training error and validation error obtained
for each repetition (fold).
The average (over all k folds) validation error delivered by k-fold CV tends to better
estimate the expected loss or risk (4.1) compared to the validation error obtained from a
single split in Algorithm 5. Consider a dataset that consists of a relatively small number of
data points. If we use a single split of this small dataset into a training set and validation
set, we might be very unlucky and choose data points for the validation set which are outliers
and not representative for the statistical properties of most data points. The e ect of such
an unlucky split is typically averaged out when using k-fold CV.
6.2.3 Imbalanced Data
The simple validation approach discussed above requires the validation set to be a good
representative for the overall statistical properties of the data. This might not be the case
in applications with discrete valued labels and some of the label values being very rare. We
might then be interested in having a good estimate of the conditional risks EfL
􀀀
(x; y); h
 
jy =
y0g where y0 is one of the rare label values. This is more than requiring a good estimate for
the risk EfL
􀀀
(x; y); h
 
g.
Consider data points characterized by a feature vector x and binary label y 2 f􀀀1; 1g.
Assume we aim at learning a hypothesis h(x) = wTx to classify data points as ^y = 1 if
156
Algorithm 6 k-fold CV ERM
Input: model H, loss function L, dataset D =
 􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
  
; number k of
folds
1: randomly shu e the data points in D
2: divide the shu ed dataset D into k folds D1; : : : ;Dk of size B = dm=ke,
D1=
 􀀀
x(1); y(1) 
; : : : ;
􀀀
x(B); y(B) 
g; : : : ;Dk=
 􀀀
x((k􀀀1)B+1); y((k􀀀1)B+1) 
; : : : ;
􀀀
x(m); y(m) 
g (6.8)
3: for fold index b = 1; : : : ; k do
4: use bth fold as the validation set D(val) = Db
5: use rest as the training set D(train) = D n Db
6: learn hypothesis ^h via ERM on the training set,
^h
(b) := argmin
h2H
bL
􀀀
hjD(train) 
(6.9)
7: compute the training error
E(b)
t := bL
􀀀
^ hjD(train) 
= (1=
  
D(train)
  
)
X
i2D(train)
L
􀀀
(x(i); y(i)); h
 
: (6.10)
8: compute validation error
E(b)
v := bL
􀀀
^ hjD(val) 
= (1=
  
D(val)
  
)
X
i2D(val)
L
􀀀
(x(i); y(i));^h
 
: (6.11)
9: end for
10: compute average training and validation errors
Et := (1=k)
Xk
b=1
E(b)
t , and Ev := (1=k)
Xk
b=1
E(b)
v
11: pick a learnt hypothesis ^h := ^h(b) for some b 2 f1; : : : ; kg
Output: learnt hypothesis ^h; average training error Et; average validation error Ev
157
h(x)   0 while ^y = 􀀀1 otherwise. The learning is based on a dataset D which contains
only one single (!) data point with y = 􀀀1. If we then split the dataset into training
and validation set, it is with high probability that the validation set does not include any
data point with label value y = 􀀀1. This cannot happen when using k-fold CV since the
single data point must be in one of the validation folds. However, even the applicability of
k-fold CV for such an imbalanced dataset is limited since we evaluate the performance of
a hypothesis h(x) using only one single data point with y = 􀀀1. The resulting validation
error will be dominated by the loss of h(x) incurred on data points from the majority class
(those with true label value y = 1).
To learn and validate a hypothesis with imbalanced data, it might be useful to to generate
synthetic data points to enlarge the minority class. This can be done using data
augmentation techniques which we discuss in Section 7.3. Another option is to choose a loss
function that takes the di erent frequencies of label values into account. Let us illustrate
this approach in what follows by an illustrative example.
Consider an imbalanced dataset of size m = 100, which contains 90 data points with
label y = 1 but only 10 data points with label y = 􀀀1. We might want to put more weight
on wrong predictions obtained for data points from the minority class (with true label value
y = 􀀀1). This can be done by using a much larger value for the loss L
􀀀
(x; y = 􀀀1); h(x) = 1
 
than for the loss L
􀀀
(x; y = 1); h(x) = 􀀀1
 
incurred by incorrectly predicting the label of a
data point from the majority class (with true label value y = 1).
6.3 Model Selection
Chapter 3 illustrated how many well-known ML methods are obtained by di erent combinations
of a hypothesis space or model, loss function and data representation. While for many
ML applications there is often a natural choice for the loss function and data representation,
the right choice for the model is typically less obvious. We now discuss how to use the
validation methods of Section 6.2 to choose between di erent candidate models.
Consider data points characterized by a single numeric feature x 2 R and numeric label
y 2 R. If we suspect that the relation between feature x and label y is non-linear, we might
use polynomial regression which is discussed in Section 3.2. Polynomial regression uses the
hypothesis space H(n)
poly with some maximum degree n. Di erent choices for the maximum
degree n yield a di erent hypothesis space: H(1) = H(0)
poly;H(2) = H(1)
poly; : : : ;H(M) = H(M)
poly.
Another ML method that learns non-linear hypothesis map is Gaussian basis regression
158
(see Section 3.5). Here, di erent choices for the variance   and shifts   of the Gaussian basis
function (3.12) result in di erent hypothesis spaces. For example, H(1) = H(2)
Gauss with   = 1
and  1 = 1 and  2 = 2, H(2) = H(2)
Gauss with   = 1=10,  1 = 10,  2 = 20.
Algorithm 7 summarizes a simple method to choose between di erent candidate models
H(1);H(2); : : : ;H(M). The idea is to  rst learn and validate a hypothesis ^h(l) separately for
each model H(l) using Algorithm 6. For each model H(l), we learn the hypothesis^h(l) via ERM
(6.4) and then compute its validation error E(l)
v (6.6). We then choose the hypothesis ^h(^l
)
from those model H(^l
) which resulted in the smallest validation error E(^l
)
v = minl=1;:::;M E(l)
v .
The work
ow of Algorithm 7 is similar to the work
ow of ERM. Remember that the idea
of ERM is to learn a hypothesis out of a set of di erent candidates (the hypothesis space).
The quality of a particular hypothesis h is measured using the (average) loss incurred on
some training set. We use the same principle for model selection but on a higher level.
Instead of learning a hypothesis within a hypothesis space, we choose (or learn) a hypothesis
space within a set of candidate hypothesis spaces. The quality of a given hypothesis space
is measured by the validation error (6.6). To determine the validation error of a hypothesis
space, we  rst learn the hypothesis ^h 2 H via ERM (6.4) on the training set. Then, we
obtain the validation error as the average loss of ^h on the validation set.
The  nal hypothesis ^h delivered by the model selection Algorithm 7 not only depends
on the training set used in ERM (see (6.9)). This hypothesis ^h has also been chosen based
on its validation error which is the average loss on the validation set in (6.11). Indeed, we
compared this validation error with the validation errors of other models to pick the model
H(^l
) (see step 10) which contains ^h. Since we used the validation error (6.11) of ^h to learn
it, we cannot use this validation error as a good indicator for the general performance of ^h.
To estimate the general performance of the  nal hypothesis ^h delivered by Algorithm 7
we must try it out on a test set. The test set, which is constructed in step 3 of Algorithm 7,
consists of data points that are neither contained in the training set (6.9) nor the validation
set (6.11) used for training and validating the candidate models H(1); : : : ;H(M). The average
loss of the  nal hypothesis on the test set is referred to as the test error. The test error is
computed in the step 12 of Algorithm 7.
Sometimes it is bene cial to use di erent loss functions for the training and the validation
of a hypothesis. As an example, consider logistic regression and the SVM which have been
discussed in Sections 3.6 and 3.7, respectively. Both methods use the same model which is the
space of linear hypothesis maps h(x) = wTx. The main di erence between these two methods
is in their choice for the loss function. Logistic regression minimizes the (average) logistic
159
Algorithm 7 Model Selection
Input: list of candidate models  􀀀 H(1); : : : ;H(M), loss function L, dataset D =
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
  
; number k of folds, test set fraction  
1: randomly shu e the data points in D
2: determine size m0 := d me of test set
3: construct a test set
D(test) =
 􀀀
x(1); y(1) 
; : : : ;
􀀀
x(m0); y(m0)  
4: construct a training set and a validation set,
D(trainval) =
 􀀀
x(m0+1); y(m0+1) 
; : : : ;
􀀀
x(m); y(m)  
5: for model index l = 1; : : : ;M do
6: run Algorithm 6 using H = H(l), dataset D = D(trainval), loss function L and k folds
7: Algorithm 6 delivers hypothesis ^h and validation error Ev
8: store learnt hypothesis ^h(l) := ^h and validation error E(l)
v := Ev
9: end for
10: pick model H(^l
) with minimum validation error E(^l
)
v =minl=1;:::;M E(l)
v
11: de ne optimal hypothesis ^h = ^h(^l
)
12: compute test error
E(test) := bL
􀀀
^ hjD(test) 
= (1=
  
D(test)
  
)
X
i2D(test)
L
􀀀
(x(i); y(i));^h
 
: (6.12)
Output: hypothesis ^h; training error E(^l
)
t ; validation error E(^l
)
v , test error E(test).
160
loss (2.12) on the training set to learn the hypothesis h(1)(x) =
􀀀
w(1)
 T
x with a parameter
vector w(1). The SVM instead minimizes the (average) hinge loss (2.11) on the training set to
learn the hypothesis h(2)(x) =
􀀀
w(2)
 T
x with a parameter vector w(2). It is inconvenient to
compare the usefulness of the two hypotheses h(1)(x) and h(2)(x) using di erent loss functions
to compute their validation errors. This comparison is more convenient if we instead compute
the validation errors for h(1)(x) and h(2)(x) using the average 0=1 loss (2.9).
Algorithm 7 requires as one of its inputs a given list of candidate models. The longer this
list, the more computation is required from Algorithm 7. Sometimes it is possible to prune
the list of candidate models by removing models that are very unlikely to have minimum
validation error.
Consider polynomial regression which uses as the model the space H(r)
poly of polynomials
with maximum degree r (see (3.4)). For r = 1, H(r)
poly is the space of polynomials with
maximum degree one (which are linear maps), h(x) = w2x + w1. For r = 2, H(r)
poly is the
space of polynomials with maximum degree two, h(x) = w3x2 + w2x + w1:
The polynomial degree r parametrizes a nested set of models,
H(r)
poly   H(r)
poly   : : : :
For each degree r, we learn a hypothesis h(r) 2 H(r)
poly with minimum average loss (training
error) E(r)
t on a training set (see (6.5)). To validate the learnt hypothesis h(r), we compute
its average loss (validation error) E(r)
v on a validation set (see (6.6)).
Figure 6.5 depicts the typical dependency of the training and validation errors on the
polynomial degree r. The training error E(r)
t decreases monotonically with increasing polynomial
degree r. To illustrate this monotonic decrease, we consider the two speci c choices
r = 3 and r = 5 with corresponding models H(r)
poly and H(r)
poly. Note that H(3)
poly   H(5)
poly since
any polynomial with degree not exceeding 3 is also a polynomial with degree not exceeding
5. Therefore, the training error (6.5) obtained when minimizing over the larger model H(5)
poly
can only decrease but never increase compared to (6.5) using the smaller model H(3)
poly
Figure 6.5 indicates that the validation error E(r)
v (see (6.6)) behaves very di erent compared
to the training error E(r)
t . Starting with degree r = 0, the validation error  rst
decreases with increasing degree r. As soon as the degree r is increased beyond a critical
value, the validation error starts to increase with increasing r. For very large values of r,
the training error becomes almost negligible while the validation error becomes very large.
In this regime, polynomial regression over ts the training set.
Figure 6.6 illustrates the over tting of polynomial regression when using a maximum
161
degree that is too large. In particular, Figure 6.6 depicts a learnt hypothesis which is a
degree 9 polynomial that  ts very well the training set, resulting in a very small training
error. To achieve this low training error the resulting polynomial has an unreasonable high
rate of change for feature values x   0. This results in large prediction errors for data points
with feature values x   0.
0 1 2 3 4 5 6 7 8 9
polyn. degree r
0.0
0.2
0.4
0.6
0.8
1.0
training
validation
Figure 6.5: The training error and validation error obtained from polynomial regression
using di erent values r for the maximum polynomial degree.
162
0.0 0.2 0.4 0.6 0.8 1.0
feature x
8
7
6
5
4
3
2
label y
learnt hypothesis for r=9
training set
validation set
Figure 6.6: A hypothesis ^h which is a polynomial with degree not larger than r = 9. The
hypothesis has been learnt by minimizing the average loss on the training set. Note the rapid
change of ^h for feature values x   0.
6.4 A Probabilistic Analysis of Generalization
More Data Beats Clever Algorithms ?; More Data Beats Clever Feature Selection?
A key challenge in ML is to ensure that a hypothesis that predicts well the labels on a
training set (which has been used to learn that hypothesis) will also predict well the labels
of data points outside the training set. We say that a ML method generalizes well if it learns
a hypothesis ^h that performs not signi cantly worse on data points outside the training set.
In other words, the loss incurred by ^h for data points outside the training set is not much
larger than the average loss of ^h incurred on the training set.
We now study the generalization of linear regression methods (see Section 3.1) using an
i.i.d. assumption. In particular, we interpret data points as iid realizations of RVs that have
the same distribution as a random data point z = (x; y). The feature vector x is then a
realization of a standard Gaussian RV with zero mean and covariance being the identity
matrix, i.e., x   N(0; I).
The label y of a random data point is related to its features x via a linear Gaussian model
y = wTx + ", with noise "   N(0;  2): (6.13)
We assume the noise variance  2  xed and known. This is a simplifying assumption and in
163
practice we would need to estimate the noise variance from data [25]. Note that, within our
probabilistic model, the error component " in (6.13) is intrinsic to the data and cannot be
overcome by any ML method. We highlight that the probabilistic model for the observed data
points is just a modelling assumption. This assumption allows us to study some fundamental
behaviour of ML methods. There are principled methods (\statistical tests") that allow to
determine if a given dataset can be accurately modelled using (6.13) [62].
We predict the label y from the features x using a linear hypothesis h(x) that depends
only on the  rst l features x1; : : : ; xl. Thus, we use the hypothesis space
H(l) = fh(w)(x) = (wT ; 0T )x with w 2 Rlg: (6.14)
Note that each element h(w) 2 H(l) corresponds to a particular choice of the parameter vector
w 2 Rl.
The model parameter l 2 f0; : : : ; ng coincides with the e ective dimension of the hypothesis
space H(l). For l < n, the hypothesis space H(l) is a proper (strict) subset of the space
of linear hypothesis maps (2.4) used within linear regression (see Section 3.1). Moreover, the
parameter l indexes a nested sequence of models,
H(0)   H(1)   : : :   H(n):
The quality of a particular predictor h(w) 2 H(l) is measured via the average squared
error bL
(h(w) j D(train)) incurred on the labeled training set
D(train) = f
􀀀
x(1); y(1) 
; : : : ;
􀀀
x(mt); y(mt) 
g: (6.15)
We interpret data points in the training set D(train) as well as any other data point outside
the training set as realizations of iid RVs with a common probability distribution. This
common probability distribution is a multivariate normal (Gaussian) distribution,
x; x(i) iid with x; x(i)   N(0; I): (6.16)
The labels y(i); y are related to the features of data points via (see (6.13))
y(i) = wTx(i) + "(i), and y = wTx + ": (6.17)
Here, the noise terms "; "(i)   N(0;  2) are realizations of iid Gaussian RVs with zero mean
and variance  2.
164
Chapter 4 showed that the training error bL
(h(w) j D(train)) is minimized by the predictor
h(bw
)(x) = bw
T Il nx, that uses the parameter vector
bw
=
􀀀􀀀
X(l) T
X(l) 􀀀1􀀀
X(l) T
y: (6.18)
Here we used the (restricted) feature matrix X(l) and the label vector y de ned as, respectively,
X(l)=(x(1); : : : ; x(mt))T In l2Rmt l, and
y=
􀀀
y(1); : : : ; y(mt) T
2Rmt : (6.19)
It will be convenient to tolerate a slight abuse of notation and denote both, the length-l
vector (6.18) as well as the zero-padded parameter vector
􀀀
bw
T ; 0T
 T
2 Rn, by bw. This
allows us to write
h(bw
)(x) = bw
Tx: (6.20)
We highlight that the formula (6.18) for the optimal weight vector bw
is only valid
if the matrix
􀀀
X(l)
 T
X(l) is invertible. Within our toy model (see (6.16)), this is true
with probability one whenever mt   l. Indeed, for mt   l the truncated feature vectors
Il nx(1); : : : ; Il nx(mt), which are iid realizations of a Gaussian RV, are linearly independent
with probability one [10, 42].
In what follows, we consider the case mt > l such that the formula (6.18) is valid (with
probability one). The more challenging high-dimensional regime mt   l will be studied in
Chapter 7.
The optimal parameter vector bw
(see (6.18)) depends on the training set D(train) via the
feature matrix X(l) and label vector y (see (6.19)). Therefore, since we model the data points
in the training set as realizations of RVs, the parameter vector bw
(6.18) is the realization of
a RV. For each speci c realization of the training set D(train), we obtain a speci c realization
of the optimal parameter vector bw
.
The probabilistic model (6.13) relates the features x of a data point to its label y via
some (unknown) true parameter vector w. Intuitively, the best linear hypothesis would be
h(x) = bw
Tx with parameter vector bw
= w. However, in general this will not be achievable
since we have to compute bw
based on the features x(i) and noisy labels y(i) of the data points
in the training set D.
The parameter vector bw
delivered by ERM (4.5) typically results in a non-zero estimation
165
error
 w := bw
􀀀 w: (6.21)
The estimation error (6.21) is the realization of a RV since the learnt parameter vector bw
(see (6.18)) is itself a realization of a RV.
The Bias and Variance Decomposition. The prediction accuracy of h(bw
), using
the learnt parameter vector (6.18), depends crucially on the mean squared estimation error
(MSEE)
Eest := Ef



 w

 
2
2
g
(6.21)
= E
 


bw
􀀀 w

 
2
2
 
: (6.22)
We will next decompose the MSEE Eest into two components, which are referred to as a
variance term and a bias term. The variance term quanti es the random 
uctuations or the
parameter vector obtained from ERM on the training set (6.15). The bias term characterizes
the systematic (or expected) deviation between the true parameter vector w (see (6.13)) and
the (expectation of the) learnt parameter vector bw
.
Let us start with rewriting (6.22) using elementary manipulations as
Eest
(6.22)
= E
 


bw
􀀀 w

 
2
2
  
= E
 


􀀀
bw
􀀀 E
 
bw
  
􀀀
􀀀
w 􀀀 E
 
bw
  
 
2
2
 
:
We can develop the last expression further by expanding the squared Euclidean norm,
Eest = E
 


bw
􀀀 Efbw
g

 
2
2
 
􀀀 2E
 􀀀
bw
􀀀 E
 
bw
  T 􀀀
w 􀀀 E
 
bw
   
+ E
 


w 􀀀 E
 
bw
 
 
2
2
 
= E
 


bw
􀀀 Efbw
g

 
2
2
 
􀀀 2
􀀀
E
 
bw
 
􀀀 E
 
bw 
| {z }
=0
 T 􀀀
w 􀀀 E
 
bw
   
+ E
 


w 􀀀 E
 
bw
 
 
2
2
 
= E
 


bw
􀀀 Efbw
g

 
2
2
 
| {z }
variance V
+E
 

w 􀀀 Efbw
g

 
2
2
 
| {z }
bias B2
: (6.23)
The  rst component in (6.23) represents the (expected) variance of the learnt parameter
vector bw
(6.18). Note that, within our probabilistic model, the training set (6.15) is the
realization of a RV since it is constituted by data points that are iid realizations of RVs (see
(6.16) and (6.13)).
The second component in (6.23) is referred to as a bias term. The parameter vector
bw
is computed from a randomly 
uctuating training set via (6.18) and is therefore itself
166

uctuating around its expectation E
 
bw
g. The bias term is the Euclidean distance between
this expectation E
 
bw
g and the true parameter vector w relating features and label of a data
point via (6.13).
The bias term B2 and the variance V in (6.23) both depend on the model complexity
parameter l but in a fundamentally di erent manner. The bias term B2 typically decreases
with increasing l while the variance V increases with increasing l. In particular, the bias
term is given as
B2 =



w 􀀀 Efbw
g

 
2
2 =
Xn
j=l+1
w2j
; (6.24)
The bias term (6.24) is zero if and only if
wj = 0 for any index j = l + 1; : : : ; n: (6.25)
The necessary and su cient condition (6.25) for zero bias is equivalent to h(w) 2 H(l).
Note that the condition (6.25) depends on both, the model parameter l and the true parameter
vector w. While the model parameter l is under control, the true parameter vector w is
not under our control but determined by the underlying data generation process. The only
way to ensure (6.25) for every possible parameter vector w in (6.13) is to use l = n, i.e., to
use all available features x1; : : : ; xn of a data point.
When using the model H(l) with l < n, we cannot guarantee a zero bias term since we
have no control over the true underlying parameter vector w in (6.13). In general, the bias
term decreases with an increasing model size l (see Figure 6.7). We highlight that the bias
term does not depend on the variance  2 of the noise " in our toy model (6.13).
Let us now consider the variance term in (6.23). Using the statistical independence of
the features and labels of data points (see (6.13), (6.16) and (6.17)), one can show that1
V = E
 


bw
􀀀 Efbw
g

 
2
2
 
=
􀀀
B2 +  2 
tr
n
E
n􀀀􀀀
X(l) T
X(l) 􀀀1
oo
: (6.26)
By (6.16), the matrix
 􀀀
X(l)
 T
X(l)
 􀀀1
is a realization of a (matrix-valued) RV with an
inverse Wishart distribution [91]. For mt > l + 1, its expectation is given as
Ef
􀀀􀀀
X(l) T
X(l) 􀀀1
g = 1=(mt 􀀀 l 􀀀 1)I: (6.27)
1This derivation is not very di cult but rather lengthy. For more details about the derivation of (6.26)
we refer to the literature [10, 88].
167
bias
variance
model complexity l
Eest
Figure 6.7: The MSEE Eest incurred by linear regression can be decomposed into a bias
term B2 and a variance term V (see (6.23)). These two components depend on the model
complexity l in an opposite manner which results in a bias-variance trade-o .
By inserting (6.27) and trfIg = l into (6.26),
V = E
n


bw
􀀀 Efbw
g

 
2
2
o
=
􀀀
B2 +  2 
l=(mt 􀀀 l 􀀀 1): (6.28)
The variance (6.28) typically increases with increasing model complexity l (see Figure 6.7).
In contrast, the bias term (6.24) decreases with increasing l.
The opposite dependence of variance and bias on the model complexity results in a biasvariance
trade-o . Choosing a model (hypothesis space) with small bias will typically result
in large variance and vice versa. In general, the choice of model must balance between a
small variance and a small bias.
Generalization. Consider a linear regression method that learns the linear hypothesis
h(x) = bw
Tx using the parameter vector (6.18). The parameter vector bw
T (6.18) results in
a linear hypothesis with minimum training error, i.e., minimum average loss on the training
set. However, the ultimate goal of ML is to  nd a hypothesis that predicts well the label
of any data point. In particular, we want the hypothesis h(x) = bw
Tx to generalize well to
data points outside the training set.
We quantify the generalization capability of h(x) = bw
Tx by its expected prediction loss
Epred = E
 􀀀
y 􀀀 bw
|{Tzx}
=^y
 2 
: (6.29)
Note that Epred is a measure for the performance of a ML method and not of a speci c
hypothesis. Indeed, the learnt parameter vector bw
is not  xed but depends on the data
points in the training set. These data points are modelled as realizations of iid RVs and, in
168
turn, the learnt parameter vector bw
becomes a realization of a RV. Thus, in some sense, the
expected prediction loss (6.29) characterizes the overall ML method that reads in a training
set and delivers (learn) a linear hypothesis with parameter vector bw
(6.18). In contrast,
the risk (4.1) introduced in Chapter 4 characterizes the performance of a speci c ( xed)
hypothesis h without taking into account a learning process that delivered h based on data.
Let us now relate the expected prediction loss (6.29) of the linear hypothesis h(x) = bw
Tx
to the bias and variance of (6.18),
Epred
(6.13)
= Ef wTxxT wg +  2
(a)
= EfEf wTxxT w j D(train)gg +  2
(b)
= Ef wT wg +  2
(6.21);(6.22)
= Eest +  2
(6.23)
= B2 + V +  2: (6.30)
Here, step (a) uses the law of iterated expectation (see, e.g., [10]). Step (b) uses that
the feature vector x of a \new" data point is a realization of a RV which is statistically
independent of the data points in the training set D(train). We also used our assumption that
x is the realization of a RV with zero mean and covariance matrix EfxxT g = I (see (6.16)).
According to (6.30), the average (expected) prediction error Epred is the sum of three
components: (i) the bias B2, (ii) the variance V and (iii) the noise variance  2. Figure 6.7
illustrates the typical dependency of the bias and variance on the model (6.14), which is
parametrized by the model complexity l. Note that the model complexity parameter l in
(6.14) coincides with the e ective model dimension de 
􀀀
H(l)
 
(see Section 2.2).
The bias and variance, whose sum is the estimation error Eest, can be in
uenced by
varying the model complexity l which is a design parameter. The noise variance  2 is the
intrinsic accuracy limit of our toy model (6.13) and is not under the control of the ML
engineer. It is impossible for any ML method (no matter how computationally expensive) to
achieve, on average, a prediction error smaller than the noise variance  2. Carefully note that
this statement only applies if the data points arising in a ML application can be (reasonably
well) modelled as realizations of iid RVs.
We highlight that our statistical analysis, resulting the formulas for bias (6.24), variance
(6.28) and the average prediction error (6.30), applies only if the observed data points can
be well modelled using the probabilistic model speci ed by (6.13), (6.16) and (6.17). The
169
validity of this probabilistic model can to be veri ed by principled statistical model validation
techniques [158, 145]. Section 6.5 discusses a fundamentally di erent approach to analyzing
the statistical properties of a ML method. Instead of a probabilistic model, this approach
uses random sampling techniques to synthesize iid copies of given (small) data points. We
can approximate the expectation of some relevant quantity, such as the loss L
􀀀􀀀
x; y
 
; h
 
,
using an average over synthetic data [58].
The qualitative behaviour of estimation error in Figure 6.7 depends on the de nition for
the model complexity. Our concept of e ective dimension (see Section 2.2) coincides with
most other notions of model complexity for the linear hypothesis space (6.14). However, for
more complicated models such as deep nets it is often not obvious how e ective dimension is
related to more tangible quantities such as total number of tunable weights or the number of
arti cial neurons. Indeed, the e ective dimension might also depend on the speci c learning
algorithm such as SGD. Therefore, for deep nets, if we would plot estimation error against
number of tunable weights we might observe a behaviour of estimation error fundamentally
di erent from the shape in Figure 6.7. One example for such un-intuitive behaviour is known
as \double descent phenomena" [9].
6.5 The Bootstrap
basic idea of bootstrap: use histogram of dataset as the underlying probability distribution;
generate new data points by random sampling (with replacement) from that distribution.
Consider learning a hypothesis ^h 2 H by minimizing the average loss incurred on a
dataset D = f
􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
 
g. The data points
􀀀
x(i)); y(i)
 
are modelled as
realizations of iid RVs. Let use denote the (common) probability distribution of these RVs
by p(x; y).
If we interpret the data points
􀀀
x(i)); y(i)
 
as realizations of RVs, also the learnt hypothesis
^h
is a realization of a RV. Indeed, the hypothesis ^h is obtained by solving an optimization
problem (4.3) that involves realizations of RVs. The bootstrap is a method for estimating
(parameters of) the probability distribution p(^h) [58].
Section 6.4 used a probabilistic model for data points to derive (the parameters of) the
probability distribution p(^h). Note that the analysis in Section 6.4 only applies to the speci c
probabilistic model (6.16), (6.17). In contrast, the bootstrap can be used for data points
drawn from an arbitrary probability distribution.
The core idea behind the bootstrap is to use the histogram ^p(z) of the data points in D
170
to generate B new datasets D(1); : : : ;D(B). Each dataset is constructed such that is has the
same size as the original dataset D. For each dataset D(b), we solve a separate ERM (4.3) to
obtain the hypothesis ^h(b). The hypothesis ^h(b) is a realization of a RV whose distribution is
determined by the histogram ^p(z) as well as the hypothesis space and the loss function used
in the ERM (4.3).
6.6 Diagnosing ML
diagnose ML methods by comparing training error with validation error and (if available)
some baseline; baseline can be obtained via the Bayes risk when using a probabilistic model
(such as the i.i.d. assumption) or human performance or the performance of existing ML
methods ("experts" in regret framework)
In what follows, we tacitly assume that data points can (to a good approximation) be
interpreted as realizations of iid RVs (see Section 2.1.4). This \i.i.d. assumption" underlies
ERM (4.3) as the guiding principle for learning a hypothesis with small risk (4.1). This
assumption also motivates to use the average loss (6.6) on a validation set as an estimate for
the risk. More fundamentally, we need the i.i.d. assumption to de ne the concept of risk as
a measure for how well a hypothesis predicts the labels of arbitrary data points.
Consider a ML method which uses Algorithm 5 (or Algorithm 6) to learn and validate
the hypothesis ^h 2 H. Besides the learnt hypothesis ^h, these algorithms also deliver the
training error Et and the validation error Ev. As we will see shortly, we can diagnose ML
methods to some extent just by comparing training with validation errors. This diagnosis is
further enabled if we know a baseline E(ref) .
One important source of a baseline E(ref) are probabilistic models for the data points (see
Section 6.4). Given a probabilistic model, which speci es the probability distribution p(x; y)
of the features and label of data points, we can compute the minimum achievable risk (4.1).
Indeed, the minimum achievable risk is precisely the expected loss of the Bayes estimator
^h
(x) of the label y, given the features x of a data point. The Bayes estimator ^h(x) is fully
determined by the probability distribution p(x; y) of the features and label of a (random)
data point [85, Chapter 4].
A further potential source for a baseline E(ref) is an existing, but for some reason unsuitable,
ML method. This existing ML method might be computationally too expensive to be
used for the ML application at end. However, we might still use its statistical properties as
a benchmark. The
171
We might also use the performance of human experts as a baseline. If we want to develop
a ML method that detects certain type of skin cancers from images of the skin, a benchmark
might be the current classi cation accuracy achieved by experienced dermatologists [37].
We can diagnose a ML method by comparing the training error Et with the validation
error Ev and (if available) the benchmark E(ref).
• Et   Ev   E(ref): The training error is on the same level as the validation error and
the benchmark error. There is not much to improve here since the validation error is
already on the desired error level. Moreover, the training error is not much smaller
than the validation error which indicates that there is no over tting. It seems we have
obtained a ML method that achieves the benchmark error level.
• Ev   Et: The validation error is signi cantly larger than the training error. It seems
that the ERM (4.3) results in a hypothesis ^h that over ts the training set. The loss
incurred by ^h on data points outside the training set, such as those in the validation
set, is signi cantly worse. This is an indicator for over tting which can be addressed
either by reducing the e ective dimension of the hypothesis space or by increasing the
size of the training set. To reduce the e ective dimension of the hypothesis space we
have di erent options depending on the used model. We might use a small number of
features in a linear model (3.1), a smaller maximum depth of decision trees (Section
3.10) or a fewer layers in an ANN (Section 3.11). One very elegant means for reducing
the e ective dimension of a hypothesis space is by limiting the number of GD steps
used in gradient-based methods. This optimization based shrinking of a hypothesis
space is referred to as early stopping. More generally, we can reduce the e ective
dimension of a hypothesis space via regularization techniques (see Chapter 7).
• Et   Ev   E(ref): The training error is on the same level as the validation error and
both are signi cantly larger than the baseline. Since the training error is not much
smaller than the validation error, the learnt hypothesis seems to not over t the training
set. However, the training error achieved by the learnt hypothesis is signi cantly larger
than the benchmark error level. There can be several reasons for this to happen. First,
it might be that the hypothesis space used by the ML method is too small, i.e., it
does not include a hypothesis that provides a good approximation for the relation
between features and label of a data point. The remedy for this situation is to use a
larger hypothesis space, e.g., by including more features in a linear model, using higher
polynomial degrees in polynomial regression, using deeper decision trees or having
172
larger ANNs (deep nets). Another reason for the training error being too large is that
the optimization algorithm used to solve ERM (4.3) is not working properly.
When using gradient-based methods (see Section 5.4) to solve ERM, one reason for
Et   E(ref) could be that the learning rate   in the GD step (5.6) is chosen too
small or too large (see Figure 5.3-(b)). This can be solved by adjusting the learning
rate by trying out several di erent values and using the one resulting in the smallest
training error. Another option is derive optimal values for the learning rate based on
a probabilistic model for how the data points are generated. One example for such a
probabilistic model is the i.i.d. assumption that has been used in Section 6.4 to analyze
linear regression methods.
• Et   Ev: The training error is signi cantly larger than the validation error (see
Exercise 6.2). The idea of ERM (4.3) is to approximate the risk (4.1) of a hypothesis by
its average loss on a training set D = f(x(i); y(i))gmi
=1. The mathematical underpinning
for this approximation is the law of large numbers which characterizes the average of
(realizations of) iid RVs. The quality and usefulness of this approximation depends on
the validity of two conditions. First, the data points used for computing the average
loss should be such that they would be typically obtained as realizations of iid RVs
with a common probability distribution. Second, the number of data points used for
computing the average loss must be su ciently large.
Whenever the data points behave di erent than the the realizations of iid RVs or if the
size of the training set or validation set is too small, the interpretation (comparison)
of training error and validation errors becomes more di cult. As an extreme case,
it might then be that the validation error consists of data points for which every
hypothesis incurs small average loss. Here, we might try to increase the size of the
validation set by collecting more labeled data points or by using data augmentation
(see Section 7.3). If the size of training set and validation set are large but we still
obtain Et   Ev, one should verify if data points in these sets conform to the i.i.d.
assumption. There are principled statistical methods that allow to test if an i.i.d.
assumption is satis ed (see [88] and references therein).
6.7 Exercises
Exercise 6.1. Validation Set Size. Consider a linear regression problem with data
173
points (x; y) characterized by a scalar feature x and a numeric label y. Assume data points
are realizations of iid RVs whose common probability distribution is multivariate normal
with zero-mean and covariance matrix C =
 
 2x
 x;y
 x;y  2
y
!
. The entries of this covariance
matrix are the variance  2x
of the (zero-mean) feature, the variance  2x
of the (zero-mean)
label and the covariance between feature and label of a random data point. How many data
points do we need to include in a validation set such that with probability of at least 0:8 the
validation error of a given hypothesis h does not deviate by more than 20 percent from its
expected loss?
Exercise 6.2. Validation Error Smaller Than Training Error? Linear regression
learns a linear hypothesis map ^h having minimal average squared error on a training set. The
learnt hypothesis ^h is then validated on a validation set which is di erent from the training
set. Can you construct a training set and validation set such that the validation error of ^h
is strictly smaller than the training error of ^h?
Exercise 6.3. When is Validation Set Too Small? The usefulness of the validation
error as an indicator for the performance of a hypothesis depends on the size of the validation
set. Experiment with di erent ML methods and datasets to  nd out the minimum required
size for the validation set.
Exercise 6.4. Too many Features? Consider data points that are characterized by
n = 1000 numeric features x1; : : : ; xn 2 R and a numeric label y 2 R. We want to learn
a linear hypothesis map h(x) = wTx for predicting the label of a data point based on its
features. Could it be bene cial to constrain the learnt hypothesis by requiring it to only
depend on the  rst 5 features of a data point?
Exercise 6.5. Benchmark via Probability Theory. Consider data points that are
characterized with single numeric feature x and label y. We model the feature and label of a
data point as iid realizations of a Gaussian random vector z   N(0;C) with zero mean and
covariance matrix C. The optimal hypothesis ^h(x) to predict the label y given the feature
x is the conditional expectation of the (unobserved) label y given the (observed) feature
x. How is the expected squared error loss of this optimal hypothesis (which is the Bayes
estimator) related to the covariance matrix C of the Gaussian random vector z.
174
Chapter 7
Regularization
Keywords: Data Augmentation. Robustness. Semi-Supervised Learning. Transfer Learning.
Multitask Learning.
label y
feature x
(x(1); y(1))
(x(2); y(2))
^h
(x)
Figure 7.1: The non-linear hypothesis map ^h perfectly predicts the labels of four data points
in a training set and therefore has vanishing training error. Despite perfectly  tting the
training set, the hypothesis ^h delivers the trivial (and useless) prediction ^y = ^h(x) = 0 for
data points outside the training set. Regularization techniques help to prevent ML methods
from learning such a map ^h.
Many ML methods use the principle of ERM (see Chapter 4) to learn a hypothesis out
of a hypothesis space by minimizing the average loss (training error) on a set of labeled data
points (which constitute a training set). Using ERM as a guiding principle for ML methods
makes sense only if the training error is a good indicator for its loss incurred outside the
training set.
Figure 7.1 illustrates a typical scenario for a modern ML method which uses a large
175
hypothesis space. This large hypothesis space includes highly non-linear maps which can
perfectly resemble any dataset of modest size. However, there might be non-linear maps for
which a small training error does not guarantee accurate predictions for the labels of data
points outside the training set.
Chapter 6 discussed validation techniques to verify if a hypothesis with small training
error will predict also well the labels of data points outside the training set. These validation
techniques, including Algorithm 5 and Algorithm 6, probe the hypothesis ^h 2 H delivered
by ERM on a validation set. The validation set consists of data points which have not been
used in the training set of ERM (4.3). The validation error, which is the average loss of the
hypothesis on the data points in the validation set, serves as an estimate for the average
error or risk (4.1) of the hypothesis ^h.
This chapter discusses regularization as an alternative to validation techniques. In contrast
to validation, regularization techniques do not require having a separate validation set
which is not used for the ERM (4.3). This makes regularization attractive for applications
where obtaining a separate validation set is di cult or costly (where labelled data is scarce).
Instead of probing a hypothesis ^h on a validation set, regularization techniques estimate
(or approximate) the loss increase when applying ^h to data points outside the training set.
The loss increase is estimated by adding a regularization term to the training error in ERM
(4.3).
Section 7.1 discusses the resulting regularized ERM, which we will refer to as SRM. It
turns out that the SRM is equivalent to ERM using a smaller (pruned) hypothesis space.
The amount of pruning depends on the weight of the regularization term relative to the
training error. For an increasing weight of the regularization term, we obtain a stronger
pruning resulting in a smaller e ective hypothesis space.
Section 7.2 constructs regularization terms by requiring the resulting ML method to be
robust against (small) random perturbations of the data points in a training set. Here, we
replace each data point of a training set by the realization of a RV that 
uctuates around
this data point. This construction allows to interpret regularization as a (implicit) form of
data augmentation.
Section 7.3 discusses data augmentation methods as a simulation-based implementation
of regularization. Data augmentation adds a certain number of perturbed copies to each
data point in the training set. One way to construct perturbed copies of a data point is to
add the realization of a RV to its features.
Section 7.4 analyzes the e ect of regularization for linear regression using a simple prob-
176
abilistic model for data points. This analysis parallels our previous study of the validation
error of linear regression in Section 6.4. Similar to Section 6.4, we reveal a trade-o  between
the bias and variance of the hypothesis learnt by regularized linear regression. This
trade- o  was traced out by a discrete model parameter (the e ective dimension) in Section
6.4. In contrast, regularization o ers a continuous trade-o  between bias and variance via a
continuous regularization parameter.
Semi-supervised learning (SSL) uses (large amounts of) unlabeled data points to support
the learning of a hypothesis from (a small number of) labeled data points [22]. Section
7.5 discusses SSL methods that use the statistical properties of unlabeled data points to
construct useful regularization terms. These regularization terms are then used in SRM
with a (typically small) set of labeled data points.
Multitask learning exploits similarities between di erent but related learning tasks [20].
We can formally de ne a learning task by a particular choice for the loss function (see
Section 2.3) . The primary role of a loss function is to score the quality of a hypothesis
map. However, the loss function also encapsulates the choice for the label of a data point.
For learning tasks de ned for a single underlying data generation process it is reasonable to
assume that the same subset of features is relevant for those learning tasks. One example
for a ML application involving several related learning tasks is multi-label classi cation (see
Section 2.1.2). Indeed, each individual label of a data point represents an separate learning
task. Section 7.6 shows how multitask learning can be implemented using regularization
methods. The loss incurred in di erent learning tasks serves mutual regularization terms in
a joint SRM for all learning tasks.
Section 7.7 shows how regularization can be used for transfer learning. Like multitask
learning also transfer learning exploits relations between di erent learning tasks. In contrast
to multitask learning, which jointly solves the individual learning tasks, transfer learning
solves the learning tasks sequentially. The most basic form of transfer learning is to  ne tune
a pre-trained model. A pre-trained model can be obtained via ERM (4.3) in a (\source")
learning task for which we have a large amount of labeled training data. The  ne-tuning is
then obtained via ERM (4.3) in the (\target") learning task of interest for which we might
have only a small amount of labeled training data.
177
7.1 Structural Risk Minimization
Section 2.2 de ned the e ective dimension de  (H) of a hypothesis space H as the maximum
number of data points that can be perfectly  t by some hypothesis h 2 H. As soon as the
e ective dimension of the hypothesis space in (4.3) exceeds the number m of training data
points, we can  nd a hypothesis that perfectly  ts the training data. However, a hypothesis
that perfectly  ts the training data might deliver poor predictions for data points outside
the training set (see Figure 7.1).
Modern ML methods typically use a hypothesis space with large e ective dimension
[150, 18]. Two well-known examples for such methods is linear regression (see Section 3.1)
using a large number of features and deep learning with ANNs using a large number (billions)
of arti cial neurons (see Section 3.11). The e ective dimension of these methods can be easily
on the order of billions (109) if not larger [126]. To avoid over tting during the naive use of
ERM (4.3) we would require a training set containing at least as many data points as the
e ective dimension of the hypothesis space. However, in practice we often do not have access
to a training set consisting of billions of labeled data points. The challenge is typically in
the labelling process which often requires human labour.
It seems natural to combat over tting of a ML method by pruning its hypothesis space
H. We prune H by removing some of the hypothesis in H to obtain the smaller hypothesis
space H0   H. We then replace ERM (4.3) with the restricted (or pruned) ERM
^h
= argmin
h2H0
bL
(hjD) with pruned hypothesis space H0 H: (7.1)
The e ective dimension of the pruned hypothesis space H0 is typically much smaller than
the e ective dimension of the original (large) hypothesis space H, de  (H0)   de  (H). For a
given size m of the training set, the risk of over tting in (7.1) is much smaller than the risk
of over tting in (4.3).
Let us illustrate the idea of pruning for linear regression using the hypothesis space (3.1)
constituted by linear maps h(x) = wTx. The e ective dimension of (3.1) is equal to the
number of features, de  (H) = n. The hypothesis space H might be too large if we use a
large number n of features, leading to over tting. We prune (3.1) by retaining only linear
hypotheses h(x) =
􀀀
w0
 T
x with parameter vectors w0 satisfying w03
= w04
= : : : = w0n = 0.
Thus, the hypothesis space H0 is constituted by all linear maps that only depend on the  rst
two features x1; x2 of a data point. The e ective dimension of H0 is dimension is de  (H0) = 2
178
instead of de  (H) = n.
Pruning the hypothesis space is a special case of a more general strategy which we refer to
as SRM [144]. The idea behind SRM is to modify the training error in ERM (4.3) to favour
hypotheses which are more smooth or regular in a speci c sense. By enforcing a smooth
hypothesis, a ML methods becomes less sensitive, or more robust, to small perturbations
of data points in the training set. Section 7.2 discusses the intimate relation between the
robustness (against perturbations of the data points in the training set) of a ML method
and its ability to generalize to data points outside the training set.
We measure the smoothness of a hypothesis using a regularizer R(h) 2 R+. Roughly
speaking, the value R(h) measures the irregularity or variation of a predictor map h. The
(design) choice for the regularizer depends on the precise de nition of what is meant by
regularity or variation of a hypothesis. Section 7.3 discusses how a particular choice for the
regularizer R(h) arises naturally from a probabilistic model for data points.
We obtain SRM by adding the scaled regularizer  R(h) to the ERM (4.3) ,
^h
= argmin
h2H
 bL
(hjD) +  R(h)
 
(2.16)
= argmin
h2H
 
(1=m)
Xm
i=1
L
􀀀
(x(i); y(i)); h
 
+  R(h)
 
: (7.2)
We can interpret the penalty term  R(h) in (7.2) as an estimate (or approximation) for the
increase, relative to the training error on D, of the average loss of a hypothesis ^h when it is
applied to data points outside D. Another interpretation of the term  R(h) will be discussed
in Section 7.3.
The regularization parameter   allows us to trade between a small training errorbL
(h(w)jD)
and small regularization term R(h), which enforces smoothness or regularity of h. If we
choose a large value for  , irregular or hypotheses h, with large R(h), are heavily \punished"
in (7.2). Thus, increasing the value of   results in the solution (minimizer) of (7.2)
having smaller R(h). On the other hand, choosing a small value for   in (7.2) puts more
emphasis on obtaining a hypothesis h incurring a small training error. For the extreme case
  = 0, the SRM (7.2) reduces to ERM (4.3).
The pruning approach (7.1) is intimately related to the SRM (7.2). They are, in a certain
sense, dual to each other. First, note that (7.2) reduces to the pruning approach (7.1) when
using the regularizer R(h) = 0 for all h 2 H0 , and R(h) = 1 otherwise, in (7.2). In the
other direction, for many important choices for the regularizer R(h), there is a restriction
179
  = 0
H( =0)
  = 1
H( =1)
  = 10
H( =10)
Figure 7.2: Adding the scaled regularizer  R(h) to the training error in the objective function
of SRM (7.2) is equivalent to solving ERM (7.1) with a pruned hypothesis space H( ).
H( )   H such that the solutions of (7.1) and (7.2) coincide (see Figure 7.2). The relation
between the optimization problems (7.1) and (7.2) can be made precise using the theory of
convex duality (see [15, Ch. 5] and [11]).
For a hypothesis space H whose elements h 2 H are parametrized by a parameter vector
w 2 Rn, we can rewrite SRM (7.2) as
bw
( ) = argmin
w2Rn
 bL
(h(w)jD) +  R(w)
 
= argmin
w2Rn
 
(1=m)
Xm
i=1
L
􀀀
(x(i); y(i)); h(w) 
+  R(w)
 
: (7.3)
For the particular choice of squared error loss (2.8), linear hypothesis space (3.1) and regularizer
R(w) = kwk22
, SRM (7.3) specializes to
bw
( ) = argmin
w2Rn
 
(1=m)
Xm
i=1
􀀀
y(i) 􀀀 wTx(i) 2
+  kwk22
 
: (7.4)
The special case (7.4) of SRM (7.3) is known as ridge regression [58].
Ridge regression (7.4) is equivalent to (see [11, Ch. 5])
bw
( ) = argmin
h(w)2H( )
(1=m)
Xm
i=1
􀀀
y(i) 􀀀 h(w)(x(i))
 2
(7.5)
180
with the restricted hypothesis space
H( ) := fh(w) : Rn ! R : h(w)(x) = wTx, with some w 2 Rn; kwk22
  C( )g   H(n): (7.6)
For any given value   of the regularization parameter in (7.4), there is a number C( ) such
that solutions of (7.4) coincide with the solutions of (7.5). Thus, ridge regression (7.4) is
equivalent to linear regression with a pruned version H( ) of the linear hypothesis space (3.1).
The size of the pruned hypothesis space H( ) (7.6) varies continuously with  .
Another popular special case of ERM (7.3) is obtained for the regularizer R(w) = kwk1
and known as the Lasso [59]
bw
( ) = argmin
w2Rn
 
(1=m)
Xm
i=1
􀀀
y(i) 􀀀 wTx(i) 2
+  kwk1
 
: (7.7)
Ridge regression (7.4) and the Lasso (7.7) have fundamentally di erent computational and
statistical properties. Ridge regression (7.4) uses a smooth and convex objective function that
can be minimized using e cient GD methods. The objective function of Lasso (7.7) is also
convex but non-smooth and therefore requires more advanced optimization methods. The
increased computational complexity of Lasso (7.7) comes at the bene t of typically delivering
a hypothesis with a smaller expected loss than those obtained from ridge regression [18, 59].
7.2 Robustness
Section 7.1 motivates regularization as a soft variant of model selection. Indeed, the regularization
term in SRM (7.2) is equivalent to ERM (7.1) using a pruned (reducing) hypothesis
space. We now discuss an alternative view on regularization as a means to make ML methods
robust.
The ML methods discussed in Chapter 4 rest on the idealizing assumption that we have
access to the true label values and feature values of labeled data points (that form a training
set). These methods learn a hypothesis h 2 H with minimum average loss (training error)
incurred for data points in the training set. In practice, the acquisition of label and feature
values might be prone to errors. These errors might stem from the measurement device itself
(hardware failures or thermal noise in electronic devices) or might be due to human mistakes
such as labelling errors.
Let us assume for the sake of exposition that the label values y(i) in the training set are
181
accurate but that the features x(i) are a perturbed version of the true features of the ith
data point. Thus, instead of having observed the data point
􀀀
x(i); y(i)
 
we could have equally
well observed the data point
􀀀
x(i) + "; y(i)
 
in the training set. Here, we have modelled the
perturbations in the features using a RV ". The probability distribution of the perturbation
" is a design parameter that controls robustness properties of the overall ML method. We
will study a particular choice for this distribution in Section 7.3.
A robust ML method should learn a hypothesis that incurs a small loss not only for a
speci c data point
􀀀
x(i); y(i)
 
but also for perturbed data points
􀀀
x(i) + "; y(i)
 
. Therefore,
it seems natural to replace the loss L
􀀀􀀀
x(i); y(i)
 
; h
 
, incurred on the ith data point in the
training set, with the expectation
E
 
L
􀀀􀀀
x(i) + "; y(i) 
; h
  
: (7.8)
The expectation (7.8) is computed using the probability distribution of the perturbation
". We will show in Section 7.3 that minimizing the average of the expectation (7.8), for
i = 1; : : : ;m, is equivalent to the SRM (7.2).
Using the expected loss (7.8) is not the only possible approach to make a ML method
robust. Another approach to make a ML method robust is known as bootstrap aggreation
(bagging). The idea of bagging is to use the bootstrap method (see Section 6.5 and [58, Ch.
8]) to construct a  nite number of perturbed copies D(1); : : : ;D(B) of the original training
set D.
We then learn (e.g, using ERM) a separate hypothesis h(b) for each perturbed copy D(b),
b = 1; : : : ;B. This results in a whole ensemble of di erent hypotheses h(b) which might even
belong to di erent hypothesis spaces. For example, one the hypothesis h(1) could be a linear
map (see Section 3.1) and the hypothesis h(2) could be obtained from an ANN (see Section
3.11).
The  nal hypothesis delivered by bagging is obtained by combining or aggregating (e.g.,
using the average) the predictions h(b)
􀀀
x
 
delivered by each hypothesis h(b), for b = 1; : : : ;B
in the ensemble. The ML method referred to as random forest uses bagging to learn an
ensemble of decision trees (see Chapter 3.10). The individual predictions obtained from the
di erent decision trees forming a random forest are then combined (e.g., using an average
for numeric labels or a majority vote for  nite-valued labels), to obtain a  nal prediction
[58].
182
7.3 Data Augmentation
ML methods using ERM (4.3) are prone to over tting as soon as the e ective dimension of
the hypothesis space H exceeds the number m of data points in the training set. Section
6.3 and Section 7.1 approached this by modifying either the model or the loss function by
adding a regularization term. Both approaches prune the hypothesis space H underlying a
ML method to reduce the e ective dimension de  (H). Model selection does this reduction
in a discrete fashion while regularization implements a soft \shrinking" of the hypothesis
space.
Instead of trying to reduce the e ective dimension we could also try to increase the
number m of data points in the training set used for ERM (4.3). We now discuss how to
synthetically generate new labeled data points by exploiting statistical symmetries of data.
The data arising in many ML applications exhibit intrinsic symmetries and invariances at
least in some approximation. The rotated image of a cat still shows a cat. The temperature
measurement taken at a given location will be similar to another measurement taken 10
milliseconds later. Data augmentation exploits such symmetries and invariances to augment
the raw data with additional synthetic data.
Let us illustrate data augmentation using an application that involves data points characterized
by features x 2 Rn and number labels y 2 R. We assume that the data generating
process is such that data points with close feature values have the same label. Equivalently,
this assumption is requiring the resulting ML method to be robust against small perturbations
of the feature values (see Section 7.2). This suggests to augment a data point
􀀀
x; y
 
by several synthetic data points
􀀀
x + "(1); y
 
; : : : ;
􀀀
x + "(B); y
 
; (7.9)
with "(1); : : : ; "(B) being realizations of iid random vectors with the same probability distribution
p(").
Given a (raw) dataset D =
 􀀀
x(1); y(1)
 
; : : : ;
􀀀
x(m); y(m)
 
g we denote the associated aug-
183
mented dataset by
D0 =
 􀀀
x(1;1); y(1) 
; : : : ;
􀀀
x(1;B); y(1) 
;
􀀀
x(2;1); y(2) 
; : : : ;
􀀀
x(2;B); y(2) 
;
: : :
􀀀
x(m;1); y(m) 
; : : : ;
􀀀
x(m;B); y(m) 
g: (7.10)
The size of the augmented dataset D0 is m0 = B   m. For a su ciently large augmentation
parameter B, the augmented sample size m0 is larger than the e ective dimension n of the
hypothesis space H. We then learn a hypothesis via ERM on the augmented dataset,
^h
= argmin
h2H
bL
(hjD0)
(7.10)
= argmin
h2H
(1=m0)
Xm
i=1
XB
b=1
L
􀀀
(x(i;b); y(i;b)); h
 
(7.9)
= argmin
h2H
(1=m)
Xm
i=1
(1=B)
XB
b=1
L
􀀀
(x(i) + "(b); y(i)); h
 
: (7.11)
We can interpret data-augmented ERM (7.11) as a data-driven form of regularization (see
Section 7.1). The regularization is implemented by replacing, for each data point
􀀀
x(i); y(i)
 
2
D, the loss L
􀀀
(x(i); y(i)); h
 
with the average loss (1=B)
PB
b=1 L
􀀀
(x(i) + "(b); y(i)); h
 
over the
augmented data points that accompany
􀀀
x(i); y(i)
 
2 D.
Note that in order to implement (7.11) we need to  rst generate B realizations "(b) 2 Rn
of iid random vectors with common probability distribution p("). This might be computationally
costly for a large B; n. However, when using a large augmentation parameter B, we
might use the approximation
(1=B)
XB
b=1
L
􀀀
(x(i) + "(b); y(i)); h
 
  E
 
L
􀀀
(x(i) + "; y(i)); h
  
: (7.12)
This approximation is made precise by a key result of probability theory, known as the law
of large numbers. We obtain an instance of ERM by inserting (7.12) into (7.11),
^h
= argmin
h2H
(1=m)
Xm
i=1
E
 
L
􀀀
(x(i) + "; y(i)); h
  
: (7.13)
184
The usefulness of (7.13) as an approximation to the augmented ERM (7.11) depends
on the di culty of computing the expectation E
 
L
􀀀
(x(i) + "; y(i)); h
  
. The complexity of
computing this expectation depends on the choice of loss function and the choice for the
probability distribution p(").
Let us study (7.13) for the special case linear regression with squared error loss (2.8) and
linear hypothesis space (3.1),
^h
= argmin
h(w)2H(n)
(1=m)
Xm
i=1
E
 􀀀
y(i) 􀀀 wT 􀀀
x(i) + "
  2 
: (7.14)
We use perturbations " drawn a multivariate normal distribution with zero mean and covariance
matrix  2I,
"   N(0;  2I): (7.15)
We develop (7.14) further by using
Ef
􀀀
y(i) 􀀀 wTx(i) 
"g = 0: (7.16)
The identity (7.16) uses that the data points
􀀀
x(i); y(i)
 
are  xed and known (deterministic)
while " is a zero-mean random vector. Combining (7.16) with (7.14),
E
 􀀀
y(i) 􀀀 wT 􀀀
x(i) + "
  2 
=
􀀀
y(i) 􀀀 wTx(i) 2
+



w

 
2
2 E
 


"

 
2
2
 
=
􀀀
y(i) 􀀀 wTx(i) 2
+ n



w

 
2
2 2: (7.17)
where the last step used E
 


"

 
2
2
  (7.15)
= n 2. Inserting (7.17) into (7.14),
^h
= argmin
h(w)2H(n)
(1=m)
Xm
i=1
􀀀
y(i) 􀀀 wTx(i) 2
+ n



w

 
2
2 2: (7.18)
We have obtained (7.18) as an approximation of the augmented ERM (7.11) for the special
case of squared error loss (2.8) and the linear hypothesis space (3.1). This approximation
uses the law of large numbers (7.12) and becomes more accurate for increasing augmentation
parameter B.
Note that (7.18) is nothing but ridge regression (7.4) using the regularization parameter
  = n 2. Thus, we can interpret ridge regression as implicit data augmentation (7.10) by
applying random perturbations (7.9) to the feature vectors in the original training set D.
185
The regularizer R(w) =



w

 
2
2 in (7.18) arose naturally from the speci c choice for the
probability distribution (7.15) of the random perturbation "(i) in (7.9) and using the squared
error loss. Other choices for this probability distribution or the loss function result in di erent
regularizers.
Augmenting data points with random perturbations distributed according (7.15) treat the
features of a data point independently. For application domains that generate data points
with highly correlated features it might be useful to augment data points using random
perturbations " (see (7.9)) distributed as
"   N(0;C): (7.19)
The covariance matrix C of the perturbation " can be chosen using domain expertise or
estimated (see Section 7.5). Inserting the distribution (7.19) into (7.13),
^h
= argmin
h(w)2H(n)
 
(1=m)
Xm
i=1
􀀀
y(i) 􀀀 wTx(i) 2
+ wTCw
 
: (7.20)
Note that (7.20) reduces to ordinary ridge regression (7.18) for the choice C =  2I.
7.4 Statistical and Computational Aspects of Regular-
ization
The goal of this section is to develop a better understanding for the e ect of the regularization
term in SRM (7.3). We will analyze the solutions of ridge regression (7.4) which is the special
case of SRM using the linear hypothesis space (3.1) and squared error loss (2.8). Using the
feature matrix X =
􀀀
x(1); : : : ; x(m)
 T
and label vector y = (y(1); : : : ; y(m))T , we can rewrite
(7.4) more compactly as
bw
( ) = argmin
w2Rn
 
(1=m)



y 􀀀 Xw

 
2
2 +  



w

 
2
2
 
: (7.21)
The solution of (7.21) is given by
bw
( ) = (1=m)
􀀀
(1=m)XTX +  I
 􀀀1
XTy: (7.22)
186
For  =0, (7.22) reduces to the formula (6.18) for the optimal weights in linear regression
(see (7.4) and (4.5)). Note that for   > 0, the formula (7.22) is always valid, even when
XTX is singular (not invertible). For   > 0 the optimization problem (7.21) (and (7.4)) has
the unique solution (7.22).
To study the statistical properties of the predictor h(bw
( ))(x) =
􀀀
bw
( )
 T
x (see (7.22)) we
use the probabilistic toy model (6.13), (6.16) and (6.17) that we used already in Section
6.4. We interpret the training data D(train) = f(x(i); y(i))gmi
=1 as realizations of iid RVs whose
distribution is de ned by (6.13), (6.16) and (6.17).
We can then de ne the average prediction error of ridge regression as
E( )
pred := E
  
y 􀀀 h(bw
( ))(x)
 2 
: (7.23)
As shown in Section 6.4, the error E( )
pred is the sum of three components: the bias, the
variance and the noise variance  2 (see (6.30)). The bias of bw
( ) is
B2 = E
 


(I 􀀀 (XTX + m I)􀀀1XTX)w

 
2
2
 
: (7.24)
For su ciently large size m of the training set, we can use the approximation
XTX   mI (7.25)
such that (7.24) can be approximated as
B2  



(I􀀀(I+ I)􀀀1)w

 
2
2
=
Xn
j=1
 
1 +  
w2j
: (7.26)
Let us compare the (approximate) bias term (7.26) of ridge regression with the bias term
(6.24) of ordinary linear regression (which is the extreme case of ridge regression with   = 0).
The bias term (7.26) increases with increasing regularization parameter   in ridge regression
(7.4). Sometimes the increase in bias is outweighed by the reduction in variance. The
variance typically decreases with increasing   as shown next.
187
bias of bw
( )
variance of bw
( )
regularization parameter  
Figure 7.3: The bias and variance of ridge regression (7.4) depend on the regularization
parameter   in an opposite manner resulting in a bias-variance trade-o .
The variance of ridge regression (7.4) satis es
V = ( 2=m2) 
tr
 
Ef((1=m)XTX+ I)􀀀1XTX((1=m)XTX+ I)􀀀1g
 
: (7.27)
Inserting the approximation (7.25) into (7.27),
V   ( 2=m2)tr
 
Ef(I+ I)􀀀1XTX(I+ I)􀀀1g
 
=  2(1=m)(n=(1+ )2): (7.28)
According to (7.28), the variance of bw
( ) decreases with increasing regularization parameter
  of ridge regression (7.4). This is the opposite behaviour as observed for the bias (7.26),
which increases with increasing  . By comparing the variance approximation (7.28) with the
variance (6.28) of linear regression suggests to interpret the ratio n=(1+ )2 as an e ective
number of features used by ridge regression. Increasing the regularization parameter  
decreases the e ective number of features.
Figure 7.3 illustrates the trade-o  between the bias B2 (7.26) of ridge regression, which
increases for increasing  , and the variance V (7.28) which decreases with increasing  . Note
that we have seen another example for a bias-variance trade-o  in Section 6.4. This trade-o 
was traced out by a discrete (model complexity) parameter l 2 f1; 2; : : :g (see (6.14)). In
stark contrast to discrete model selection, the bias-variance trade-o  for ridge regression is
traced out by the continuous regularization parameter   2 R+.
The main statistical e ect of the regularization term in ridge regression is to balance
the bias with the variance to minimize the average prediction error of the learnt hypothesis.
There is also a computational e ect or adding a regularization term. Roughly speaking, the
188
regularization term serves as a pre-conditioning of the optimization problem and, in turn,
reduces the computational complexity of solving ridge regression (7.21).
The objective function in (7.21) is a smooth (in nitely often di erentiable) convex function.
We can therefore use GD to solve (7.21) e ciently (see Chapter 5). Algorithm 8
summarizes the application of GD to (7.21). The computational complexity of Algorithm
8 depends crucially on the number of GD iterations required to reach a su ciently small
neighbourhood of the solutions to (7.21). Adding the regularization term  kwk22
to the
objective function of linear regression speeds up GD. To verify this claim, we  rst rewrite
(7.21) as the quadratic problem
min
w2Rn
(1=2)wTQw 􀀀 qT | {z w}
=f(w)
with Q = (1=m)XTX +  I; q = (1=m)XTy: (7.29)
This is similar to the quadratic optimization problem (4.9) underlying linear regression but
with a di erent matrix Q. The computational complexity (number of iterations) required by
GD (see (5.6)) to solve (7.29) up to a prescribed accuracy depends crucially on the condition
number  (Q)   1 of the psd matrix Q [68]. The smaller the condition number  (Q), the
fewer iterations are required by GD. We refer to a matrix with a small condition number as
being \well-conditioned".
The condition number of the matrix Q in (7.29) is given by
 (Q) =
 max((1=m)XTX) +  
 min((1=m)XTX) +  
: (7.30)
According to (7.30), the condition number  (Q) tends to one for increasing regularization
parameter  ,
lim
 !1
 max((1=m)XTX) +  
 min((1=m)XTX) +  
= 1: (7.31)
Thus, the number of required GD iterations in Algorithm 8 decreases with increasing regularization
parameter  .
7.5 Semi-Supervised Learning
Consider the task of predicting the numeric label y of a data point z =
􀀀
x; y
 
based on
its feature vector x =
􀀀
x1; : : : ; xn
 T
2 Rn. At our disposal are two datasets D(u) and D(l).
189
Algorithm 8 Regularized Linear regression via GD
Input: dataset D = f(x(i); y(i))gmi
=1; GD learning rate   > 0.
Initialize:set w(0) :=0; set iteration counter r :=0
1: repeat
2: r := r + 1 (increase iteration counter)
3: w(r) := (1 􀀀   )w(r􀀀1) +  (2=m)
Pm
i=1(y(i) 􀀀
􀀀
w(r􀀀1))Tx(i))x(i) (do a GD step (5.6))
4: until stopping criterion met
Output: w(r) (which approximates bw
( ) in (7.21))
For each datapoint in D(u) we only know the feature vector. We therefore refer to D(u) as
\unlabelled data". For each datapoint in D(l) we know both, the feature vector x and the
label y. We therefore refer to D(l) as \labeled data".
SSL methods exploit the information provided by unlabelled data D(u) to support the
learning of a hypothesis based on minimizing its empirical risk on the labelled (training)
data D(l). The success of SSL methods depends on the statistical properties of the data
generated within a given application domain. Loosely speaking, the information provided
by the probability distribution of the features must be relevant for the ultimate task of
predicting the label y from the the features x [22].
Let us design a SSL method, summarized in Algorithm 9 below, using the data augmentation
perspective from Section 7.3. The idea is the augment the (small) labeled dataset
D(l) by adding random perturbations fo the features vectors of data point in D(l). This is
reasonable for applications where feature vectors are subject to inherent measurement or
modelling errors. Given a data point with vector x we could have equally well observed a
feature vector x+" with some small random perturbation "   N(0;C). To estimate the covariance
matrix C, we use the sample covariance matrix of the feature vectors in the (large)
unlabelled dataset D(u). We then learn a hypothesis using the augmented (regularized) ERM
(7.20).
7.6 Multitask Learning
Consider a speci c learning task of  nding a hypothesis h with minimum (expected) loss
L
􀀀
(x; y); h
 
. Note that the loss incurred by h for a speci c data point depends on the
de nition for the label of a data point. We can obtain di erent learning tasks for the same
data points by using di erent choices or de nitions for the label of a data point. Multitask
learning exploits the similarities between di erent learning tasks to jointly solve them. Let
190
Algorithm 9 A Semi-Supervised Learning Algorithm
Input: labeled dataset D(l) = f(x(i); y(i))gmi
=1; unlabeled dataset D(u) = fex
(i)gm0
i=1
1: compute C via sample covariance on D(u),
C := (1=m0)
m0 X
i=1
􀀀
ex
(i)􀀀bx
 􀀀
ex
(i)􀀀bx
 T
with bx
:= (1=m0)
m0 X
i=1
ex
(i): (7.32)
2: compute (e.g. using GD)
bw
:= argmin
w2Rn
 
(1=m)
Xm
i=1
􀀀
y(i) 􀀀 wTx(i) 2
+ wTCw
 
: (7.33)
Output: hypothesis ^h(x) =
􀀀
bw
)Tx
us next discuss a simple example of a multitask learning problem.
Consider a data point z representing a hand-drawing that is collected via the online
game https://quickdraw.withgoogle.com/. The features of a data point are the pixel
intensities of the bitmap which is used to store the hand-drawing. As label we could use the
fact if a hand-drawing shows an apple or not. This results in the learning task T (1). Another
choice for the label of a hand-drawing could be the fact if a hand-drawing shows a fruit at
all or not. This results in another learning task T (2) which is similar but di erent from the
task T (1).
The idea of multitask learning is that a reasonable hypothesis h for a learning task
should also do well for a related learning tasks. Thus, we can use the loss incurred on similar
learning tasks as a regularization term for learning a hypothesis for the learning task at
hand. Algorithm 10 is a straightforward implementation of this idea for a given dataset that
gives rise to T related learning tasks T (1); : : : ; T (T). For each individual learning task T (t0)
it uses the loss on the remaining learning tasks T (t), with t 6= t0, as regularization term in
(7.34).
The applicability of Algorithm 10 is somewhat limited as it aims at  nding a single
hypothesis that does well for all T learning tasks simultaneously. For certain application
domains it might be more reasonable to not learn a single hypothesis for all learning tasks
but to learn a separate hypothesis h(t) for each learning task t = 1; : : : ; T. However, these
separate hypotheses typically might still share some structural similarities.1 We can enforce
1One important example for such a structural similarity in the case of linear predictors h(t)(x) = 􀀀
w(t)
 T
xis that the parameter vectors w(T) have a small joint support. Requiring the parameter vectors to
191
Algorithm 10 A Multitask Learning Algorithm
Input: dataset D = fz(1); : : : ; z(m)g; T learning tasks with loss functions L(1); : : : ;L(T),
hypothesis space H
1: learn a hypothesis ^h via
^h
:= argmin
h2H
XT
t=1
Xm
i=1
L(t)􀀀
z(i); h
 
: (7.34)
Output: hypothesis ^h
di erent notion of similarities between the hypotheses h(t) by adding a regularization term
to the loss functions of the tasks.
Algorithm 11 generalizes Algorithms 10 by learning a separate hypothesis for each task
t while requiring these hypotheses to be structurally similar. The structural (dis-)similarity
between the hypotheses is measured by a regularization term R in (7.35).
Algorithm 11 A Multitask Learning Algorithm
Input: dataset D = fz(1); : : : ; z(m)g with T associated learning tasks with loss functions
L(1); : : : ;L(T), hypothesis space H
1: learn a hypothesis ^h via
^h
(1); : : : ;^h(T) := argmin
h(1);:::;h(T)2H
XT
t=1
Xm
i=1
L(t)􀀀
z(i); h(t) 
+  R
􀀀
h(1); : : : ; h(T) 
: (7.35)
Output: hypotheses ^h(1); : : : ;^h(T)
7.7 Transfer Learning
Regularization is also instrumental for transfer learning to capitalize on synergies between
di erent related learning tasks [108, 61]. Transfer learning is enabled by constructing regularization
terms for a learning task by using the result of a previous leaning task. While
multitask learning methods solve many related learning tasks simultaneously, transfer learning
methods operate in a sequential fashion.
have a small joint support is equivalent to requiring the stacked vector ew
=
􀀀
w(1); : : : ;w(T)
 
to be block
(group) sparse [35].
192
Let us illustrate the idea of transfer learning using two learning tasks which di er signifcantly
in their intrinsic di culty. Informally, we consider a learning task to be easy if
we can easily gather large amounts of labeled (training) data for that task. Consider the
learning task T (1) of predicting whether an image shows a cat or not. For this learning task
we can easily gather a large training set D(1) using via image collections of animals. Another
(related) learning task T (2) is to predict whether an image shows a cat of a particular breed,
with a particular body height and with a speci c age. The learning task T (2) is more di cult
than T (1) since we have only a very limited amount of cat images for which we know the
particular breed, body height and precise age of the depicted cat.
7.8 Exercises
Exercise 7.1. Ridge Regression is a Quadratic Problem. Consider the linear
hypothesis space consisting of linear maps parameterized by weights w. We try to  nd the
best linear map by minimizing the regularized average squared error loss (empirical risk)
incurred on a training set
D :=
 
(x(1); y(1)); (x(2); y(2)); : : : ; (x(m); y(m))
 
:
Ridge reression augments the average squared error loss on D by the regularizer kwk2,
yielding the following learning problem
min
w2Rn
f(w) = (1=m)
Xm
i=1
􀀀
y(i) 􀀀 wTx(i) 
+  kwk22
:
Is it possible to rewrite the objective function f(w) as a convex quadratic function f(w) =
wTCw + bw + c? If this is possible, how are the matrix C, vector b and constant c related
to the feature vectors and labels of the training data ?
Exercise 7.2. Regularization or Model Selection. Consider data points, each
characterized by n = 10 features x 2 Rn and a single numeric label y. We want to learn a
linear hypothesis h(x) = wTx by minimizing the average squared error on the training set D
of size m = 4. We could learn such a hypothesis by two approaches. The  rst approach is to
split the dataset into a training set and a validation set. Then we consider all models that
consists of linear hypotheses with weight vectors having at most two non-zero weights. Each
193
of these models corresponds to a di erent subset of two weights that might be non-zero.
Find the model resulting in the smallest validation errors (see Algorithm 5). Compute the
average loss of the resulting optimal linear hypothesis on some data points that have neither
been used in the training set nor the validation set. Compare this average loss (\test error")
with the average loss obtained on the same data points by the hypothesis learnt by ridge
regression (7.4).
194
Chapter 8
Clustering
x(3)
x(4)
x(2)
x(1)
x(5)
x(6)
x(7)
xg
xr
Figure 8.1: Each circle represents an image which is characterized by its average redness
xr and average greenness xg. The i-th image is depicted by a circle located at the point
x(i) =
􀀀
x(i)
r ; x(i)
g
 T
2 R2. It seems that the images can be grouped into two clusters.
So far we focused on ML methods that use the ERM principle and lean a hypothesis
by minimizing the discrepancy between its predictions and the true labels on a training set.
These methods are referred to as supervised methods as they require labeled data points
for which the true label values have been determined by some human (who serves as a
\supervisor"). This and the following chapter discus ML methods which do not require to
know the label of any data point. These methods are often referred to as \unsupervised"
since they do not require a \supervisor" to provide the label values for any data point.
One important family of unsupervised ML methods aim at clustering a given set of data
195
points such as those depicted in Figure 8.1. The basic idea of clustering is to decompose a set
of data points into few subsets or clusters that consist of similar data points. For the dataset
in Figure 8.1 it seems reasonable to de ne two clusters, one cluster
 
x(1); x(5); x(6); x(7)
 
and
a second cluster
 
x(2); x(3); x(4); x(8)
 
.
Formally, clustering methods learn a hypothesis that assign each data point either to
precisely one cluster (see Section 8.1) or several clusters with di erent degrees of belonging
(see Section 8.2). Di erent clustering methods use di erent measures for the similarity between
data points. For data points characterized by (numeric) Euclidean feature vectors,
the similarity between data points can be naturally de ned in terms of the Euclidean distance
between feature vectors. Section 8.3 discusses clustering methods that use notions of
similarity that are not based on a Euclidean space.
There is a strong conceptual link between clustering methods and the classi cation methods
discussed in Chapter 3. Both type of methods learn a hypothesis that reads in the features
of a data point and delivers a prediction for some quantity of interest. In classi cation
methods, this quantity of interest is some generic label of a data point. For clustering methods,
this quantity of interest for a data point is the cluster assignment (for hard clustering)
of the degree of belonging (for soft clustering). A main di erence between clustering and
classi cation is that clustering methods do not require the true label (cluster assignment or
degree of belonging) of a single data point.
Classi cation methods learn a good hypothesis via minimizing their average loss incurred
on a training set of labeled data points. In contrast, clustering methods do not have access
to a single labeled data point. To  nd the correct labels (cluster assignments) clustering
methods rely solely on the intrinsic geometry of the data points. We will see that clustering
methods use this intrinsic geometry to de ne an empirical risk incurred by a candidate
hypothesis. Like classi cation methods, also clustering methods use an instance of the ERM
principle (see Chapter 4) to  nd a good hypothesis (clustering).
This chapter discusses two main 
avours of clustering methods:
• hard clustering (see Section 8.1)
• and soft clustering methods (see Section 8.2).
Hard clustering methods learn a hypothesis h that reads in the feature vector x of a data
point and delivers a predicted cluster assignment ^y = h(x) 2 f1; : : : ; kg. Thus, assigns each
data point to one single cluster. Section 8.1 will discuss one of the most widely-used hard
clustering algorithms which is known as k-means.
196
In contrast to hard clustering methods, soft clustering methods assign each data point
to several clusters with varying degree of belonging. These methods learn a hypothesis
that delivers a vector ^y =
􀀀
^y1; : : : ; ^yk
 T
with entry ^yc 2 [0; 1] being the predicted degree
by which the data point belongs to the c-th cluster. Hard clustering is an extreme case
of soft clustering where we enforce each degree of belonging to take only values in f0; 1g.
Moreover, hard clustering requires that for each data point only of the corresponding degree
of belonging (one for each cluster) is non-zero.
The main focus of this chapter is on methods that require data points being represented
by numeric feature vectors (see Sections 8.1 and 8.2). These methods de ne the similarity
between data points using the Euclidean distance between their feature vectors. Some applications
generate data points for which it is not obvious how to obtain numeric feature
vectors such that their Euclidean distances re
ect the similarity between data points. It is
then desirable to use a more 
exible notion of similarity which does not require to determine
(useful) numeric feature vectors of data points.
Maybe the most fundamental concept to represent similarities between data points is
a similarity graph. The nodes of the similarity graph are the individual data points of a
dataset. Similar data points are connected by edges (links) that might be assigned some
weight that quantities the amount of similarity. Section 8.3 discusses clustering methods
that use a graph to represent similarities between data points.
197
8.1 Hard Clustering with k-means
Consider a dataset D which consists of m data points that are indexed by i = 1; : : : ;m. The
data points are characterized via their numeric feature vectors x(i) 2 Rn, for i = 1; : : : ;m.
It will be convenient for the following discussion if we identify a data point with its feature
vector. In particular, we refer by x(i) to the i-th data point. Hard clustering methods
decompose (or cluster) the dataset into a given number k of di erent clusters C(1); : : : ; C(k).
These methods assign each data point x(i) to one and only one cluster C(c) with the cluster
index c 2 f1; : : : ; kg.
Let us de ne for each data point its label y(i) 2 f1; : : : ; kg as the index of the cluster
to which the ith data point actually belongs to. The c-th cluster consists of all data points
with y(i) = c,
C(c) :=
 
i 2 f1; : : : ;mg : y(i) = c
 
: (8.1)
We can interpret hard clustering methods as ML methods that compute predictions ^y(i)
for the (\correct") cluster assignments y(i). The predicted cluster assignments result in the
predicted clusters
b C(c) :=
 
i 2 f1; : : : ;mg : ^y(i) = c
 
, for c = 1; : : : ; k: (8.2)
We now discuss a hard clustering method which is known as k-means. This method does
not require the knowledge of the label or (true) cluster assignment y(i) for any data point in
D. This method computes predicted cluster assignments ^y(i) based solely from the intrinsic
geometry of the feature vectors x(i) 2 Rn for all i = 1; : : : ;m. Since it does not require
any labeled data points, k-means is often referred to as being an unsupervised method.
However, note that k-means requires the number k of clusters to be given as an input (or
hyper-) parameter.
The k-means method represents the c-th cluster b C(c) by a representative feature vector
 (c) 2 Rn. It seems reasonable to assign data points in D to clusters b C(c) such that they are
well concentrated around the cluster representatives  (c). We make this informal requirement
precise by de ning the clustering error
bL􀀀
f (c)gkc
=1; f^y(i)gmi
=1 j D
 
= (1=m)
Xm
i=1


 
x(i) 􀀀  (^y(i))




2
: (8.3)
Note that the clustering error bL
(8.3) depends on both, the cluster assignments ^y(i), which
198
de ne the cluster (8.2), and the cluster representatives  (c), for c = 1; : : : ; k.
Finding the optimal cluster means f (c)gkc
=1 and cluster assignments f^y(i)gmi
=1 that minimize
the clustering error (8.3) is computationally challenging. The di culty stems from the
fact that the clustering error is a non-convex function of the cluster means and assignments.
While jointly optimizing the cluster means and assignments is hard, separately optimizing
either the cluster means for given assignments or vice-versa is easy. In what follows, we
present simple closed-form solutions for these sub-problems. The k-means method simply
combines these solutions in an alternating fashion.
It can be shown that for given predictions (cluster assignments) ^y(i), the clustering error
(8.3) is minimized by setting the cluster representatives equal to the cluster means [13]
 (c) :=
􀀀
1=j b C(c)j
  X
^y(i)=c
x(i): (8.4)
To evaluate (8.4) we need to know the predicted cluster assignments ^y(i). The crux is that the
optimal predictions ^y(i), in the sense of minimizing clustering error (8.3), depend themselves
on the choice for the cluster representatives  (c). In particular, for given cluster representative
 (c) with c = 1; : : : ; k, the clustering error is minimized by the cluster assignments
^y(i) 2 argmin
c2f1;:::;kg



x(i) 􀀀  (c)



: (8.5)
Here, we denote by argmin
c02f1;:::;kg
kx(i) 􀀀  (c0)k the set of all cluster indices c 2 f1; : : : ; kg such
that kx(i) 􀀀  (c)k = minc02f1;:::;kg kx(i) 􀀀  (c0)k.
Note that (8.5) assigns the ith datapoint to those cluster C(c) whose cluster mean  (c) is
nearest (in Euclidean distance) to x(i). Thus, if we knew the optimal cluster representatives,
we could predict the cluster assignments using (8.5). However, we do not know the optimal
cluster representatives unless we have found good predictions for the cluster assignments ^y(i)
(see (8.4)).
To recap: We have characterized the optimal choice (8.4) for the cluster representatives
for given cluster assignments and the optimal choice (8.5) for the cluster assignments for
given cluster representatives. It seems natural, starting from some initial guess for the
cluster representatives, to alternate between the cluster assignment update (8.5) and the
update (8.4) for the cluster means. This alternating optimization strategy is illustrated in
Figure 8.2 and summarized in Algorithm 12. Note that Algorithm 12, which is maybe the
most basic variant of k-means, simply alternates between the two updates (8.4) and (8.5)
199
initial choice for
cluster means
update cluster
assignment (8.6)
update cluster
means (8.7)
\k-Means"
Figure 8.2: The work
ow of k-means. Starting from an initial guess or estimate for the
cluster means, the cluster assignments and cluster means are updated (improved) in an
alternating fashion.
until some stopping criterion is satis ed.
Algorithm 12 requires the speci cation of the number k of clusters and initial choices
for the cluster means  (c), for c = 1; : : : ; k. Those quantities are hyper-parameters that
must be tuned to the speci c geometry of the given dataset D. This tuning can be based on
probabilistic models for the dataset and its cluster structure (see Section 2.1.4 and [81, 149]).
Alternatively, if Algorithm 12 is used as pre-processing within an overall supervised ML
method (see Chapter 3), the validation error (see Section 6.3) of the overall method might
guide the choice of the number k of clusters.
Choosing Number of Clusters. The choice for the number k of clusters typically
depends on the role of the clustering method within an overall ML application. If the
clustering method serves as a pre-processing for a supervised ML problem, we could try
out di erent values of the number k and determine, for each choice k, the corresponding
validation error. We then pick the value of k which results in the smallest validation error.
If the clustering method is mainly used as a tool for data visualization, we might prefer a
small number of clusters. The choice for the number k of clusters can also be guided by
the so-called \elbow-method". Here, we run the k-means Algorithm 12 for several di erent
choices of k. For each value of k, Algorithm 12 delivers a clustering with clustering error
E(k) = bL
􀀀
f (c)gkc
=1; f^y(i)gmi
=1 j D
 
:
200
Algorithm 12 \k-means"
Input: dataset D = fx(i)gmi
=1; number k of clusters; initial cluster means  (c) for c =
1; : : : ; k.
1: repeat
2: for each datapoint x(i), i=1; : : : ;m, do
^y(i) := argmin
c02f1;:::;kg
kx(i) 􀀀  (c0)k (update cluster assignments) (8.6)
3: for each cluster c=1; : : : ; k do
 (c) :=
1
jfi : ^y(i) = cgj
X
i:^y(i)=c
x(i) (update cluster means) (8.7)
4: until stopping criterion is met
5: compute  nal clustering error E(k) := (1=m)
Pm
i=1


 
x(i) 􀀀  (^y(i))




2
Output: cluster means  (c), for c = 1; : : : ; k, cluster assignments ^y(i) 2 f1; : : : ; kg, for
i = 1; : : : ;m,  nal clustering error E(k)
We then plot the minimum empirical error E(k) as a function of the number k of clusters.
Figure 8.3 depicts an example for such a plot which typically starts with a steep decrease
for increasing k and then 
attening out for larger values of k. Note that for k   m we can
achieve zero clustering error since each datapoint x(i) can be assigned to a separate cluster
C(c) whose mean coincides with that datapoint, x(i) =  (c).
Cluster-Means Initialization. We brie
y mention some popular strategies for choosing
the initial cluster means in Algorithm 12. One option is to initialize the cluster means
with realizations of iid random vectors whose probability distribution is matched to the
dataset D = fx(i)gmi
=1 (see Section 3.12). For example, we could use a multivariate normal
distribution N(x;b ;b 
) with the sample mean b 
= (1=m)
Pm
i=1 x(i) and the sample covariance
b 
= (1=m)
Pm
i=1(x(i)􀀀b 
)(x(i)􀀀b 
)T . Alternatively, we could choose the initial cluster
means  (c) by selecting k di erent data points x(i) from D. This selection process might
combine random choices with an optimization of the distances between cluster means [4]. Finally,
the cluster means might also be chosen by evenly partitioning the principal component
of the dataset (see Chapter 9).
Interpretation as ERM. For a practical implementation of Algorithm 12 we need to
decide when to stop updating the cluster means and assignments (see (8.6) and (8.7)). To
this end it is useful to interpret Algorithm 12 as a method for iteratively minimizing the
201
2 4 6 8 10
2
4
6
8
k
E(k)
Figure 8.3: The clustering error E(k) achieved by k-means for increasing number k of clusters.
clustering error (8.3). As can be veri ed easily, the updates (8.6) and (8.7) always modify
(update) the cluster means or assignments in such a way that the clustering error (8.3) is
never increased. Thus, each new iteration of Algorithm 12 results in cluster means and
assignments with a smaller (or the same) clustering error compared to the cluster means
and assignments obtained after the previous iteration. Algorithm 12 implements a form of
ERM (see Chapter 4) using the clustering error (8.3) as the empirical risk incurred by the
predicted cluster assignments ^y(i). Note that after completing a full iteration of Algorithm
12, the cluster means
 
 (c)
 k
c=1 are fully determined by the cluster assignments
 
^y(i)
 m
i=1
via (8.7). It seems natural to terminate Algorithm 12 if the decrease in the clustering error
achieved by the most recent iteration is below a prescribed (small) threshold.
Clustering and Classi cation. There is a strong conceptual link between Algorithm
12 and classi cation methods (see e.g. Section 3.13). Both methods essentially learn a
hypothesis h(x) that maps the feature vector x to a predicted label ^y = h(x) from a  nite
set. The practical meaning of the label values is di erent for Algorithm 12 and classi cation
methods. For classi cation methods, the meaning of the label values is essentially de ned by
the training set (of labeled data points) used for ERM (4.3). On the other hand, clustering
methods use the predicted label ^y = h(x) as a cluster index.
Another main di erence between Algorithm 12 and most classi cation methods is the
choice for the empirical risk used to evaluate the quality or usefulness of a given hypothesis
h( ). Classi cation methods typically use an average loss over labeled data points in a
202
training set as empirical risk. In contrast, Algorithm 12 uses the clustering error (8.3) as
a form of empirical risk. Consider a hypothesis that resembles the cluster assignments ^y(i)
obtained after completing an iteration in Algorithm 12, ^y(i) = h
􀀀
x(i)
 
. Then we can rewrite
the resulting clustering error achieved after this iteration as
bL
􀀀
hjD
 
= (1=m)
Xm
i=1






x(i) 􀀀
P
i02D(i) x(i0)
  
D(i)
  






2
: with D(i) :=
 
i0 : h
􀀀
x(i) 
= h
􀀀
x(i0)  
: (8.8)
Note that the i-th summand in (8.8) depends on the entire dataset D and not only on (the
features of) the i-th data point x(i).
Some Practicalities. For a practical implementation of Algorithm 12 we need to  x
three issues.
• Issue 1 (\tie-breaking"): We need to specify what to do if several di erent cluster
indices c 2 f1; : : : ; kg achieve the minimum value in the cluster assignment update
(8.6) during step 2.
• Issue 2 (\empty cluster"): The cluster assignment update (8.6) in step 3 of Algorithm
12 might result in a cluster c with no datapoints associated with it, jfi : ^y(i) = cgj = 0.
For such a cluster c, the update (8.7) is not well-de ned.
• Issue 3 (\stopping criterion"): We need to specify a criterion used in step 4 of Algorithm
12 to decide when to stop iterating.
Algorithm 13 is obtained from Algorithm 12 by  xing those three issues [52]. Step 3 of
Algorithm 13 solves the  rst issue mentioned above (\tie breaking"), arising when there
are several cluster clusters whose means have minimum distance to a data point x(i), by
assigning x(i) to the cluster with smallest cluster index (see (8.9)). Step 4 of Algorithm 13
resolves the \empty cluster" issue by computing the variables b(c) 2 f0; 1g for c = 1; : : : ; k.
The variable b(c) indicates if the cluster with index c is active (b(c) = 1) or the cluster c is
inactive (b(c) = 0). The cluster c is de ned to be inactive if there are no data points assigned
to it during the preceding cluster assignment step (8.9). The cluster activity indicators b(c)
allows to restrict the cluster mean updates (8.10) only to the clusters c with at least one data
point x(i). To obtain a stopping criterion, step 7 Algorithm 13 monitors the clustering error
Er incurred by the cluster means and assignments obtained after r iterations. Algorithm 13
continues updating cluster assignments (8.9) and cluster means (8.10) as long as the decrease
is above a given threshold "   0.
203
Algorithm 13 \k-Means II" (slight variation of \Fixed Point Algorithm" in [52])
Input: dataset D = fx(i)gmi
  =1; number k of clusters; tolerance "   0; initial cluster means
 (c)
 k
c=1
1: Initialize. set iteration counter r := 0; E0 := 0
2: repeat
3: for all datapoints i=1; : : : ;m,
^y(i) := minf argmin
c02f1;:::;kg
kx(i) 􀀀  (c0)kg (update cluster assignments) (8.9)
4: for all clusters c=1; : : : ; k, update the activity indicator
b(c) :=
(
1 if jfi : ^y(i) = cgj > 0
0 else.
5: for all c=1; : : : ; k with b(c) = 1,
 (c) :=
1
jfi : ^y(i) = cgj
X
fi:^y(i)=cg
x(i) (update cluster means) (8.10)
6: r := r + 1 (increment iteration counter)
7: Er := bL
􀀀
f (c)gkc
=1; f^y(i)gmi
=1 j D
 
(evaluate clustering error (8.3))
8: until r > 1 and Er􀀀1 􀀀 Er   " (check for su cient decrease in clustering error)
9: E(k) := (1=m)
Pm
i=1


 
x(i) 􀀀  (^y(i))




2
(compute  nal clustering error)
Output: cluster assignments ^y(i)2f1; : : : ; kg, cluster means  (c), clustering error E(k).
204
For Algorithm 13 to be useful we must ensure that the stopping criterion is met within a
 nite number of iterations. In other words, we must ensure that the clustering error decrease
can be made arbitrarily small within a su ciently large (but  nite) number of iterations. To
this end, it is useful to represent Algorithm 13 as a  xed-point iteration
f^y(i)gmi
=1 7! Pf^y(i)gm m=1: (8.11)
The operator P, which depends on the dataset D, reads in a list of cluster assignments and
delivers an improved list of cluster assignments aiming at reducing the associated clustering
error (8.3). Each iteration of Algorithm 13 updates the cluster assignments ^y(i) by applying
the operator P. Representing Algorithm 13 as a  xed-point iteration (8.11) allows for an
elegant proof of the convergence of Algorithm 13 within a  nite number of iterations (even
for " = 0) [52, Thm. 2].
Figure 8.4 depicts the evolution of the cluster assignments and cluster means during
the iterations Algorithm 13. Each subplot corresponds to one iteration of Algorithm 13 and
depicts the cluster means before that iteration and the clustering assignments (via the marker
symbols) after the corresponding iteration. In particular, the upper left subplot depicts the
cluster means before the  rst iteration (which are the initial cluster means) and the cluster
assignments obtained after the  rst iteration of Algorithm 13.
−100 −50 0 50 100
−0.02
−0.01
0.00
0.01
0.02
first iteration
mean of cluster 1
mean of cluster 2
−100 −50 0 50 100
−0.02
−0.01
0.00
0.01
0.02
second iteration
−100 −75 −50 −25 0 25 50 75 100
−0.02
−0.01
0.00
0.01
0.02
third iteration
−100 −75 −50 −25 0 25 50 75 100
−0.02
−0.01
0.00
0.01
0.02
fourth iteration
Figure 8.4: The evolution of cluster means (8.7) and cluster assignments (8.6) (depicted as
large dot and large cross) during the  rst four iterations of k-means Algorithm 13.
Consider running Algorithm 13 with tolerance " = 0 (see step 8) such that the iterations
205
are continued until there is no decrease in the clustering error E(r) (see step 7 of Algorithm
13). As discussed above, Algorithm 13 will terminate after a  nite number of iterations.
Moreover, for " = 0, the delivered cluster assignments
 
^y(i)
 m
i=1 are fully determined by the
delivered clustered means
 
 (c)
 k
c=1,
^y(i) = minf argmin
c02f1;:::;kg
kx(i) 􀀀  (c0)kg: (8.12)
Indeed, if (8.12) does not hold one can show the  nal iteration r would still decrease the
clustering error and the stopping criterion in step 8 would not be met.
If cluster assignments and cluster means satisfy the condition (8.12), we can rewrite the
clustering error (8.3) as a function of the cluster means solely,
bL
􀀀 
 (c) k
c=1
jD
 
:= (1=m)
Xm
i=1
min
c02f1;:::;kg
kx(i) 􀀀  (c0)k2: (8.13)
Even for cluster assignments and cluster means that do not satisfy (8.12), we can still use
(8.13) to lower bound the clustering error (8.3),
bL
􀀀 
 (c) k
c=1
jD
 
  bL
􀀀
f (c)gkc
=1; f^y(i)gmi
=1 j D
 
.
Algorithm 13 iteratively improves the cluster means in order to minimize (8.13). Ideally,
we would like Algorithm 13 to deliver cluster means that achieve the global minimum of
(8.13) (see Figure 8.5). However, for some combination of dataset D and initial cluster means,
Algorithm 13 delivers cluster means that form only a local optimum of bL
􀀀 
 (c)
 k
c=1
jD
 
which
is strictly worse (larger) than its global optimum (see Figure 8.5).
The tendency of Algorithm 13 to get trapped around a local minimum of (8.13) depends
on the initial choice for cluster means. It is therefore useful to repeat Algorithm 13 several
times, with each repetition using a di erent initial choice for the cluster means. We then
pick the cluster assignments f^y(i)gmi
=1 obtained for the repetition that resulted in the smallest
clustering error E(k) (see step 9).
206
bL
􀀀
f (c)gkc
=1 j D
 
local minimum
global minimum
Figure 8.5: The clustering error (8.13) is a non-convex function of the cluster means f (c)gkc
=1.
Algorithm 13 iteratively updates cluster means to minimize the clustering error but might
get trapped around one of its local minimum.
8.2 Soft Clustering with Gaussian Mixture Models
Consider a dataset D = fx(1); : : : ; x(m)g that we wish to group into a given number of
k di erent clusters. The hard clustering methods discussed in Section 8.1 deliver cluster
assignments ^y(i), for i = 1; : : : ;m. The cluster assignment ^y(i) is the index of the cluster
to which the ith data point x(i) is assigned to. These cluster assignments ^y provide rather
coarse-grained information. Two data points x(i); x(i0) might be assigned to the same cluster
c although their distances to the cluster mean  (c) might di er signi cantly. Intuitively,
these two data points have a di erent degree of belonging to the cluster c.
For some clustering applications it is desirable to quantify the degree by which a data
point belongs to a cluster. Soft clustering methods use a continues range, such as the closed
interval [0; 1], of possible values for the degree of belonging. In contrast, hard clustering
methods use only two possible values for the degree of belonging to a speci c cluster, either
\full belonging" or no \belonging at all". While hard clustering methods assign a given data
point to precisely one cluster, soft clustering methods typically assign a data point to several
di erent clusters with non-zero degree of belonging.
This chapter discusses soft clustering methods that compute, for each data point x(i) in
the dataset D, a vector by
(i) =
􀀀
^y(i)
1 ; : : : ; ^y(i)
k
 T
. We can interpret the entry ^y(i)
c 2 [0; 1] as the
degree by which the data point x(i) belongs to the cluster C(c). For ^y(i)
c   1, we are quite
con dent in the data point x(i) belonging to cluster C(c). In contrast, for ^y(i)
c   0, we are
quite con dent that the data point x(i) is outside the cluster C(c).
A widely used soft clustering method uses a probabilistic model for the data points
D = fx(i)gmi
=1. Within this model, each cluster C(c), for c = 1; : : : ; k, is represented by a
207
multivariate normal distribution [10]
N
􀀀
x; (c); (c) 
=
1 p
detf2  g
exp
􀀀
􀀀 (1=2)
􀀀
x􀀀 (c) T 􀀀
 (c) 􀀀1􀀀
x􀀀 (c)  
: (8.14)
The probability distribution (8.14) is parametrized by a cluster-speci c mean vector  (c) and
an (invertible) cluster-speci c covariance matrix  (c).1 Let us interpret a speci c data point
x(i) as a realization drawn from the probability distribution (8.14) of a speci c cluster c(i),
x(i)   N
􀀀
x; (c); (c) 
with cluster index c = c(i): (8.15)
We can think of c(i) as the true index of the cluster to which the data point x(i) belongs to.
The variable c(i) selects the cluster distributions (8.14) from which the feature vector x(i)
has been generated (drawn). We will therefore refer to the variable c(i) as the (true) cluster
assignment for the ith data point. Similar to the feature vectors x(i) we also interpret the
cluster assignments c(i), for i = 1; : : : ;m as realizations of iid RVs.
In contrast to the feature vectors x(i), we do not observe (know) the true cluster indices
c(i). After all, the goal of soft clustering is to estimate the cluster indices c(i). We obtain a
soft clustering method by estimating the cluster indices c(i) based solely on the data points in
D. To compute these estimates we assume that the (true) cluster indices c(i) are realizations
of iid RVs with the common probability distribution (or probability mass function)
pc := p(c(i) = c) for c = 1; : : : ; k: (8.16)
The (prior) probabilities pc, for c = 1; : : : ; k, are either assumed known or estimated from
data [85, 10]. The choice for the probabilities pc could re
ect some prior knowledge about
di erent sizes of the clusters. For example, if cluster C(1) is known to be larger than cluster
C(2), we might choose the prior probabilities such that p1 > p2.
The probabilistic model given by (8.15), (8.16) is referred to as a a GMM. This name
is quite natural as the common marginal distribution for the feature vectors x(i), for i =
1; : : : ;m, is a (additive) mixture of multivariate normal (Gaussian) distributions,
p(x(i)) =
Xk
c=1
|p(c(i{)z= c})
pc
|p(x(i)j{cz(i) = c})
N(x(i); (c); (c))
: (8.17)
1Note that the expression (8.14) is only valid for an invertible (non-singular) covariance matrix  .
208
As already mentioned, the cluster assignments c(i) are hidden (unobserved) RVs. We thus
have to infer or estimate these variables from the observed data points x(i) which realizations
or iid RVs with the common distribution (8.17).
The GMM (see (8.15) and (8.16)) lends naturally to a rigorous de nition for the degree
y(i)
c by which data point x(i) belongs to cluster c.2 Let us de ne the label value y(i)
c as the
\a-posteriori" probability of the cluster assignment c(i) being equal to c 2 f1; : : : ; kg:
y(i)
c := p(c(i) = cjD): (8.18)
By their very de nition (8.18), the degrees of belonging y(i)
c always sum to one,
Xk
c=1
y(i)
c = 1 for each i = 1; : : : ; m: (8.19)
We emphasize that we use the conditional cluster probability (8.18), conditioned on the
dataset D, for de ning the degree of belonging y(i)
c . This is reasonable since the degree of
belonging y(i)
c depends on the overall (cluster) geometry of the dataset D.
The de nition (8.18) for the label values (degree of belongings) y(i)
c involves the GMM
parameters f (c); (c); pcgkc
=1 (see (8.17)). Since we do not know these parameters beforehand
we cannot evaluate the conditional probability in (8.18). A principled approach to solve this
problem is to evaluate (8.18) with the true GMM parameters replaced by some estimates
fb 
(c);b 
(c); ^pcgk c=1. Plugging in the GMM parameter estimates into (8.18) provides us with
predictions ^y(i)
c for the degrees of belonging. However, to compute the GMM parameter
estimates we would have already needed the degrees of belonging y(i)
c . This situation is similar
to hard clustering where ultime goals is to jointly optimize cluster means and assignments
(see Section 8.1).
Similar to the spirit of Algorithm 12 for hard clustering, we solve the above dilemma
of soft clustering by an alternating optimization scheme. This scheme, which is illustrated
in Figure 8.6, alternates between updating (optimizing) the predicted degrees of belonging
(or soft cluster assignments) ^y(i)
c , for i = 1; : : : ;m and c = 1; : : : ; k, given the current
GMM parameter estimates fb 
(c);b 
(c); ^pcgk c=1 and then updating (optimizing) these GMM
parameter estimates based on the updated predictions ^y(i)
c . We summarize the resulting soft
clustering method in Algorithm 14. Each iteration of Algorithm 14 consists of an update
2Remember that the degree of belonging y(i)
c is considered as the (unknown) label value of a data point.
The choice or de nition for the labels of data points is a design choice. In particular, we can de ne the labels
of data points using a hypothetical probabilistic model such as the GMM.
209
initial choice for
cluster means, cov.
and e ective size
update soft cluster
assignment
update cluster
means, cov. and e . size
^y(i)
c
fb (c);b 
(c); ^pcgkc
=1
Figure 8.6: The work
ow of the soft clustering Algorithm 14. Starting from an initial guess
or estimate for the cluster parameters, the soft cluster assignments and cluster parameters
are updated (improved) in an alternating fashion.
(8.22) for the degrees of belonging followed by an update (step 3) for the GMM parameters.
To analyze Algorithm 14 it is helpful to interpret (the features of) data points x(i) as realizations
of iid RVs distributed according to a GMM (8.15)-(8.16). We can then understand
Algorithm 14 as a method for estimating the GMM parameters based on observing realizations
drawn from the GMM (8.15)-(8.16). We can estimate the parameters of a probability
distribution using the maximum likelihood method (see Section 3.12 and [76, 85]). As its
name suggests, maximum likelihood methods estimate the GMM parameters by maximizing
the probability (density)
p
􀀀
D; f (c); (c); pcgkc
=1
 
(8.20)
of actually observing the data points in the dataset D.
It can be shown that Algorithm 14 is an instance of a generic approximate maximum
likelihood technique referred to as expectation maximization expectation maximization (EM)
(see [58, Chap. 8.5] for more details). In particular, each iteration of Algorithm 14 updates
the GMM parameter estimates such that the corresponding probability density (8.20) does
not decrease [156]. If we denote the GMM parameter estimate obtained after r iterations of
Algorithm 14 by  (r) [58, Sec. 8.5.2],
p
􀀀
D;  (r+1) 
  p
􀀀
D;  (r) 
(8.21)
210
 (1)
 (1)
 (2)
 (2)
 (3)
 (3)
Figure 8.7: The GMM (8.15), (8.16) results in a probability distribution (8.17) for (feature
vectors of) data points which is a weighted sum of multivariate normal distributions
N( (c); (c)). The weight of the c-th component is the cluster probability p(c(i) = c).
Algorithm 14 \A Soft-Clustering Algorithm" [13]
Input: dataset D = fx(i)gmi
=1; number k of clusters, initial GMM parameter estimates
fb 
(c);b 
(c); ^pcgkc
=1
1: repeat
2: for each i = 1; : : : ;m and c = 1; : : : ; k, update degrees of belonging
^y(i)
c :=
^pcN(x(i);b 
(c);b 
(c))
Pk
c0=1 ^pc0N(x(i);b 
(c0);b 
(c0))
(8.22)
3: for each c 2 f1; : : : ; kg, update GMM parameter estimates:
• ^pc :=mc=m with e ective cluster size mc :=
mP
i=1
^y(i)
c (cluster probability)
• b 
(c) := (1=mc)
mP
i=1
^y(i)
c x(i) (cluster mean)
• b 
(c) := (1=mc)
mP
i=1
^y(i)
c
􀀀
x(i)􀀀b 
(c)
 􀀀
x(i)􀀀b 
(c)
 T
(cluster covariance matrix)
4: until stopping criterion met
Output: predicted degrees of belonging by
(i) = (^y(i)
1 ; : : : ; ^y(i)
k )T for i = 1; : : : ;m.
211
As for Algorithm 12, we can also interpret Algorithm 14 as an instance of the ERM principle
discussed in Chapter 4. Indeed, maximizing the probability density (8.20) is equivalent
to minimizing the empirical risk
bL
􀀀
  j D
 
:= 􀀀log p
􀀀
D;  
 
with GMM parameters   := f (c); (c); pcgkc
=1 (8.23)
The empirical risk (8.23) is the negative logarithm of the probability (density) (8.20) of
observing the dataset D as iid realizations of the GMM (8.17). The monotone increase in
the probability density (8.21) achieved by the iterations of Algorithm 14 translate into a
monotone decrease of the empirical risk,
bL
􀀀
 (r) j D
 
  bL
􀀀
 (r􀀀1) j D
 
with iteration counter r: (8.24)
The monotone decrease (8.24) in the empirical risk (8.23) achieved by the iterations of
Algorithm 14 naturally lends to a stopping criterion. Let Er denote the empirical risk (8.23)
achieved by the GMM parameter estimates  (r) obtained after r iterations in Algorithm 14.
Algorithm 14 stops iterating as soon as the decrease Er􀀀1􀀀Er achieved by the r-th iteration
of Algorithm 14 falls below a given (positive) threshold " > 0.
Similar to Algorithm 12, also Algorithm 14 might get trapped in local minima of the
underlying empirical risk. The GMM parameters delivered by Algorithm 14 might only be
a local minimum of (8.23) but not the global minimum (see Figure 8.5 for the analogous
situation in hard clustering). As for hard clustering Algorithm 12, we typically repeat
Algorithm 14 several times. During each repetition of Algorithm 14, we use a di erent
(randomly chosen) initialization for the GMM parameter estimates   = fb 
(c);b 
(c); ^pcgkc
=1.
Each repetition of Algorithm 14 results in a potentially di erent set of GMM parameter
estimates and degrees of belongings ^y(i)
c . We then use the results for that repetition that
achieves the smallest empirical risk (8.23).
Let us point out an interesting link between soft clustering methods based on GMM (see
Algorithm 14) and hard clustering with k-means (see Algorithm 12). Consider the GMM
(8.15) with prescribed cluster covariance matrices
 (c) =  2I for all c 2 f1; : : : ; kg; (8.25)
with some given variance  2 > 0. We assume the cluster covariance matrices in the GMM to
be given by (8.25) and therefore can replace the covariance matrix updates in Algorithm 14
212
with the assignment b 
(c) :=  2I. It can be veri ed easily that for su ciently small variance
 2 in (8.25), the update (8.22) tends to enforce ^y(i)
c 2 f0; 1g. In other words, each data point
x(i) becomes then e ectively associated with exactly one single cluster c whose cluster mean
b 
(c) is nearest to x(i). For  2 ! 0, the soft clustering update (8.22) in Algorithm 14 reduces
to the (hard) cluster assignment update (8.6) in k-means Algorithm 12. We can interpret
Algorithm 12 as an extreme case of Algorithm 14 that is obtained by  xing the covariance
matrices in the GMM to  2I with a su ciently small  2.
Combining GMM with linear regression. Let us sketch how Algorithm 14 could
be combined with linear regression methods (see Section 3.1). The idea is to  rst compute
the degree of belongings to the clusters for each data point. We then learn separate linear
predictors for each cluster using the degree of belongings as weights for the individual loss
terms in the training error. To predict the label of a new data point, we  rst compute
the predictions obtained for each cluster-speci c linear hypothesis. These cluster-speci c
predictions are then averaged using the degree of belongings for the new data point as
weights.
8.3 Connectivity-based Clustering
The clustering methods discussed in Sections 8.1 and 8.2 can only be applied to data points
which are characterized by numeric feature vectors. These methods de ne the similarity
between data points using the Euclidean distance between the feature vectors of these data
points. As illustrated in Figure 8.8, these methods can only produce \Euclidean shaped"
clusters that are contained either within hyper-spheres (Algorithm 12) or hyper-ellipsoids
(Algorithm 14).
 (1)
(a)
 (1)
 (2)
 (3)  (2)
 (3)
 (1)
 (1)
 (2)
 (3)  (2)
(b)
 (3)
Figure 8.8: (a): Cartoon of typical cluster shapes delivered by k-means Algorithm 13. (b):
Cartoon of typical cluster shapes delivered by soft clustering Algorithm 14.
213
Some applications generate data points for which the construction of useful numeric features
is di cult. Even if we can easily obtain numeric features for data points, the Euclidean
distances between the resulting feature vectors might not re
ect the actual similarities between
data points. As a case in point, consider data points representing text documents.
We could use the histogram of a respeci ed list of words as numeric features for a text document.
In general, a small Euclidean distance between histograms of text documents does
not imply that the text documents have similar meanings. Moreover, clusters of similar text
documents might have highly complicated shapes in the space of feature vectors that cannot
be grouped within hyper-ellipsoids. For datasets with such \non-Euclidean" cluster shapes,
k-means or GMM are not suitable as clustering methods. We should then replace the Euclidean
distance between feature vectors with another concept to determine or measure the
similarity between data points.
Connectivity-based clustering methods do not require any numeric features of data points.
These methods cluster data points based on explicitly specifying for any two di erent data
points if they are similar and to what extend. A convenient mathematical tool to represent
similarities between the data points of a dataset D is a weighted undirected graph G =
􀀀
V; E
 
.
We refer to this graph as the similarity graph of the dataset D (see Figure 8.9). The nodes
V in this similarity graph G represent data points in D and the undirected edges connect
nodes that represent similar data points. The extend of the similarity is represented by the
weights Wi;i0 for each edge fi; i0g 2 E.
Given a similarity graph G of a dataset, connectivity-based clustering methods determine
clusters as subsets of nodes that are well connected within the cluster but weakly connected
between di erent clusters. Di erent concepts for quantifying the connectivity between nodes
in a graph yield di erent clustering methods. Spectral clustering methods use eigenvectors of
a graph Laplacian matrix to measure the connectivity between nodes [147, 106]. Flow-based
clustering methods measure the connectivity between two nodes via the amount of 
ow that
can be routed between them [73]. Note that we might use these connectivity measures to
construct meaningful numerical feature vectors for the nodes in the empirical graph. These
feature vectors can then be fed into the hard-clustering Algorithm 13 or the soft clustering
Algorithm 14 (see Figure 8.9).
The algorithm density-based spatial clustering of applications with noise (DBSCAN)
considers two data points i; i0 as connected if one of them (say i) is a core node and the other
214
node (i0) can be reached via a sequence (path) of connected core nodes
i(1); : : : ; i(r) , with fi; i(1)g; fi(1); i(2)g; : : : ; fi(r); i0g 2 E:
DBSCAN considers a node to be a core node if it has a su ciently large number of neighbours
[36]. The minimum number of neighbours required for a node to be considered a core node
is a hyper-parameter of DBSCAN. When DBSCAN is applied to data points with numeric
feature vectors, it de nes two data points as connected if the Euclidean distance between
their feature vectors does not exceed a given threshold " (see Figure 8.10).
In contrast to k-means and GMM, DBSCAN does not require the number of clusters to
be speci ed. The number of clusters is determined automatically by DBSCAN and depends
on its hyper-parameters. DBSCAN also performs an implicit outlier detection. The outliers
delivered by DBSCAN are those data points which do not belong to the same cluster as any
other data point.
1
2
3
4 5 6
7
8
x(1)
(a)
x(1)
x(2)
x(3)
x(4)
x(5)x(6)
x(7)
x(8)
x1
x2
(b)
Figure 8.9: Connectivity-based clustering can be obtained by constructing features x(i) that
are (approximately) identical for well-connected data points. (a): A similarity graph for
a dataset D consists of nodes representing individual data points and edges that connect
similar data points. (b) Feature vectors of well-connected data points have small Euclidean
distance.
215
x(1)
x(2)
< "
Figure 8.10: DBSCAN assigns two data points to the same cluster if they are reachable.
Two data points x(i); x(i0) are reachable if there is a path of data points from x(i0) to x(i).
This path consists of a sequence of data points that are within a distance of ". Moreover,
each data point on this path must be a core point which has at least a given number of
neighbouring data points within the distance ".
8.4 Clustering as Preprocessing
In applications it might be bene ial to combine clustering methods with supervised methods
such as linear regression. As a point in case consider a dataset that consists of data points
obtained from two di erent data generation processes. Let us denote the data points generated
by one process by D(1) and the other one by D(2). Each datapoint is characterzed by
features and a label. While there would be an accurate lienar hypothesis for predicting the
label of datapoints in D(1) and another linear hypothesis for D(2) these two are very di erent.
We could try to use clustering methods to assign any given data point to the corresponding
data generation process. If we are lucky, the resulting clusters resemble (approximately)
the two data generation processes D(1) and D(2). Once we have successfully clustered the
data points, we can learn a separate (tailored) hypothesis for ach cluster. More generally,
we can use the predicted cluster assignments obtained from the methods of Section 8.1 - 8.3
as additional features for each data point.
Let us illustrate the above ideas by combining Algorithm 12 with linear regression. We
 rst group data points into a given number k of clusters and then learn separate linear
predictors h(c)(x) =
􀀀
w(c)
 T
x for each cluster c = 1; : : : ; k. To predict the label of a new
data point with features x, we  rst assign to the cluster c0 with the nearest cluster mean.
We then use the linear predictor h(c0) assigned to cluster c0 to compute the predicted label
^y = h(c0)(x).
216
8.5 Exercises
Exercise 8.1. Monotonicity of k-means Updates. Show that the cluster means and
assignments updates (8.7) and (8.6) never increase the clustering error (8.3).
Exercise 8.2. How to choose k in k-means? Discuss and experiment with di erent
strategies for choosing the number k of clusters in k-means Algorithm 13.
Exercise 8.3. Local Minima. Apply the hard clustering Algorithm 13 to the dataset
(􀀀10; 1); (10; 1); (􀀀10;􀀀1); (10;􀀀1) with initial cluster means (0; 1); (0;􀀀1) and tolerance
" = 0. For this initialization, will Algorithm 13 get trapped in a local minimum of the
clustering error (8.13)?
Exercise 8.4. Image Compression with k-means. Apply k-means to image
compression. Consider image pixels as data points whose features are RGB intensities. We
obtain a simple image compression format by, instead of storing RGB pixel values, storing
the cluster means (which are RGB triplets) and the cluster index for each pixel. Try out
di erent values for the numberf k of clusters and discuss the resulting trade o  between
achievable reconstruction quality and storage size.
Exercise 8.5. Compression with k-means. Consider m = 10000 datapoints
x(1); : : : ; x(m) which are represented by numeric feature vectors of length two. We apply
k-means to cluster the data set into k = 5 clusters. How many bits do we need to store the
resulting cluster assignments?
217
Chapter 9
Feature Learning
\Solving Problems By Changing the Viewpoint."
Figure 9.1: Dimensionality reduction methods aim at  nding a map h which maximally
compresses the raw data while still allowing to accurately reconstruct the original datapoint
from a small number of features x1; : : : ; xn.
Chapter 2 de ned features as those properties of a data point that can be measured or
computed easily. Sometimes the choice of features follows naturally from the available hardand
software. For example, we might use the numeric measurement z 2 R delivered by a
sensing device as a feature. However, we could augment this single feature with new features
such as the powers z2 and z3 or adding a constant z+5. Each of these computations produces
a new feature. Which of these additional features are most useful?
Feature learning methods automate the choice of  nding good features. These methods
learn a hypothesis map that reads in some representation of a data point and transforms
it to a set of features. Feature learning methods di er in the precise format of the original
218
data representation as well as the format of the delivered features. This chapter mainly
discusses feature learning methods that require data points being represented by n0 numeric
raw features and deliver a set of n new numeric features. We will denote the raw features and
the learnt new features by z =
􀀀
z1; : : : ; zn0
 T
2 Rn0 and x =
􀀀
x1; : : : ; xn
 T
2 Rn, respectively.
Many ML application domains generate data points for which can access a huge number
of raw features. Consider data points being snapshots generated by a smartphone. It seems
natural to use the pixel colour intensities as the raw features of the snapshot. Since modern
smartphone have \Megapixel cameras", the pixel intensities would provide us with millions
of raw features. It might seem a good idea to use as many raw features of a data point as
possible since more features should o er more information about a data point and its label
y. There are, however, two pitfalls in using an unnecessarily large number of features. The
 rst one is a computational pitfall and the second one is a statistical pitfall.
Computationally, using very long feature vectors x 2 Rn (with n being billions), might
result in prohibitive computational resource requirements (bandwidth, storage, time) of the
resulting ML method. Statistically, using a large number of features makes the resulting ML
methods more prone to over tting. For example, linear regression will typically over t when
using feature vectors x 2 Rn whose length n exceeds the number m of labeled data points
used for training (see Chapter 7).
Both from a computational and a statistical perspective, it is bene cial to use only the
maximum necessary amount of features. The challenge is to select those features which carry
most of the relevant information required for the prediction of the label y. Finding the most
relevant features out of a huge number of raw features is the goal of dimensionality reduction
methods. Dimensionality reduction methods form an important sub-class of feature learning
methods. These methods learn a hypothesis h(z) that maps a long raw feature vector z 2 Rn0
to a new (short) feature vector x 2 Rn with n   n0.
Beside avoiding over tting and coping with limited computational resources, dimensionality
reduction can also be useful for data visualization. Indeed, if the resulting feature
vector has length n = 2, we depict data points in the two-dimensional plane in form of a
scatterplot.
We will discuss the basic idea underlying dimensionality reduction methods in Section
9.1. Section 9.2 presents one particular example of a dimensionality reduction method that
computes relevant features by a linear transformation of the raw feature vector. Section
9.4 discusses a method for dimensionality reduction that exploits the availability of labelled
data points. Section 9.6 shows how randomness can be used to obtain computationally cheap
219
dimensionality reduction .
Most of this chapter discusses dimensionality reduction methods that determine a small
number of relevant features from a large set of raw features. However, sometimes it might
be useful to go the opposite direction. There are applications where it might be bene cial to
construct a large (even in nite) number of new features from a small set of raw features. Section
9.7 will showcase how computing additional features can help to improve the prediction
accuracy of ML methods.
9.1 Basic Principle of Dimensionality Reduction
The e ciency of ML methods depends crucially on the choice of features that are used to
characterize data points. Ideally we would like to have a small number of highly relevant
features to characterize data points. If we use too many features we risk to waste computations
on exploring irrelevant features. If we use too few features we might not have
enough information to predict the label of a data point. For a given number n of features,
dimensionality reduction methods aim at learning an (in a certain sense) optimal map from
the data point to a feature vector of length n.
Figure 9.1 illustrates the basic idea of dimensionality reduction methods. Their goal is
to learn (or  nd) a \compression" map h( ) : Rn0 ! Rn that transforms a (long) raw feature
vector z 2 Rn0 to a (short) feature vector x = (x1; : : : ; xn)T := h(z) (typically n   n0).
The new feature vector x = h(z) serves as a compressed representation (or code) for the
original feature vector z. We can reconstruct the raw feature vector using a reconstruction
map r( ) : Rn ! Rn0 . The reconstructed raw features bz
:= r(x) = r(h(z)) will typically
di er from the original raw feature vector z. In general, we obtain a non-zero reconstruction
error
bz
|{z}
=r(h(z)))
􀀀z: (9.1)
Dimensionality reduction methods learn a compression map h( ) such that the reconstruction
error (9.1) is minimized. In particular, for a dataset D =
 
z(1); : : : ; z(m)
 
, we
measure the quality of a pair of compression map h and reconstruction map r by the average
reconstruction error
bL
􀀀
h; rjD
 
:= (1=m)
Xm
i=1
L
􀀀
z(i); r
􀀀
h
􀀀
z(i)   
: (9.2)
Here, L
􀀀
z; r
􀀀
h
􀀀
z(i)
  
denotes a loss function that is used to measure the reconstruction error
220
r
􀀀
h
􀀀
z(i)  
| {z }
bz
􀀀z. Di erent choices for the loss function in (9.2) result in di erent dimensionality
reduction methods. One widely-used choice for the loss is the squared Euclidean norm
L
􀀀
z; g
􀀀
h
􀀀
z
   
:=



z 􀀀 g
􀀀
h
􀀀
z
  
 
2
2: (9.3)
Practical dimensionality reduction methods have only  nite computational resources.
Any practical method must therefore restrict the set of possible compression and reconstruction
maps to small subsets H and H , respectively. These subsets are the hypothesis
spaces for the compression map h 2 H and the reconstruction map r 2 H . Feature learning
methods di er in their choice for these hypothesis spaces.
Dimensionality reduction methods learn a compression map by solving
^h
= argmin
h2H
min
r2H 
bL
􀀀
h; rjD
 
(9.2)
= argmin
h2H
min
r2H 
(1=m)
Xm
i=1
L
􀀀
z(i); r
􀀀
h
􀀀
z(i)   
: (9.4)
We can interpret (9.4) as a (typically non-linear) approximation problem. The optimal compression
map ^h is such that the reconstructionr(^h(z)), with a suitably chosen reconstruction
map r, approximates the original raw feature vector z as good as possible. Note that we use
a single compression map h( ) and a single reconstruction map r( ) for all data points in the
dataset D.
We obtain variety of dimensionality methods by using di erent choices for the hypothesis
spaces H;H  and loss function in (9.4). Section 9.2 discusses a method that solves (9.4) for
H;H  constituted by linear maps and the loss (9.3). Deep autoencoders are another family
of dimensionality reduction methods that solve (9.4) with H;H  constituted by non-linear
maps that are represented by deep neural networks [49, Ch. 14].
9.2 Principal Component Analysis
We now consider the special case of dimensionality reduction where the compression and
reconstruction map are required to be linear maps. Consider a data point which is characterized
by a (typically very long) raw feature vector z=
􀀀
z1; : : : ; zn0
 T
2Rn0 of length n0. The
length n0 of the raw feature vector might be easily of the order of millions. To obtain a small
set of relevant features x=
􀀀
x1; : : : ; xn
 T
2Rn, we apply a linear transformation to the raw
221
feature vector,
x = Wz: (9.5)
Here, the \compression" matrixW 2 Rn n0 maps (in a linear fashion) the (long) raw feature
vector z 2 Rn0 to the (shorter) feature vector x 2 Rn.
It is reasonable to choose the compression matrix W 2 Rn n0 in (9.5) such that the
resulting features x 2 Rn allow to approximate the original data point z 2 Rn0 as accurate
as possible. We can approximate (or recover) the data point z 2 Rn0 back from the features
x by applying a reconstruction operator R 2 Rn0 n, which is chosen such that
z   Rx
(9.5)
= RWz: (9.6)
The approximation error bL
􀀀
W;R j D
 
resulting when (9.6) is applied to each data point
in a dataset D = fz(i)gmi
=1 is then
bL
􀀀
W;R j D
 
= (1=m)
Xm
i=1
kz(i) 􀀀 RWz(i)k2: (9.7)
One can verify that the approximation error bL
􀀀
W;R j D
 
can only by minimal if the
compression matrix W is of the form
W =WPCA :=
􀀀
u(1); : : : ; u(n) T
2 Rn n0
; (9.8)
with n orthonormal vectors u(j), for j = 1; : : : ; n. The vectors u(j) are the eigenvectors
corresponding to the n largest eigenvalues of the sample covariance matrix
Q := (1=m)ZTZ 2 Rn0 n0
: (9.9)
Here we used the data matrix Z =
􀀀
z(1); : : : ; z(m)
 T
2 Rm n0 .1 It can be veri ed easily,
using the de nition (9.9), that the matrix Q is psd. As a psd matrix, Q has an eigenvalue
decomposition (EVD) of the form [134]
Q =
􀀀
u(1); : : : ; u(n0) 
0
BB@
 1 : : : 0
0
. . . 0
0 : : :  n0
1
CCA
􀀀
u(1); : : : ; u(n0) T
(9.10)
1Some authors de ne the data matrix as Z =
􀀀
ez
(1); : : : ;ez
(m)
 T
2 Rm D using \centered" data points
z(i) 􀀀 bmobtained by subtracting the average bm
= (1=m)
Pm
i=1 z(i).
222
with real-valued eigenvalues  1    2   : : :    n0   0 and orthonormal eigenvectors fu(j)gn0
j=1.
The feature vectors x(i) are obtained by applying the compression matrix WPCA (9.8)
to the raw feature vectors z(i). We refer to the entries of the learnt feature vector x(i) =
WPCAz(i) (see (9.8)) as the principal components (PC) of the raw feature vectors z(i). Algorithm
15 summarizes the overall procedure of determining the compression matrix (9.8)
and computing the learnt feature vectors x(i). This procedure is known as PCA. The length
Algorithm 15 PCA
Input: dataset D = fz(i) 2 Rn0gmi
=1; number n of PCs.
1: compute the EVD (9.10) to obtain orthonormal eigenvectors
􀀀
u(1); : : : ; u(n0)
 
corresponding
to (decreasingly ordered) eigenvalues  1    2   : : :    n0   0
2: construct compression matrix WPCA :=
􀀀
u(1); : : : ; u(n)
 T
2 Rn n0
3: compute feature vector x(i) =WPCAz(i) whose entries are PC of z(i)
4: compute approximation error bL
(PCA) =
Pn0
j=n+1  j (see (9.11)).
Output: x(i), for i = 1; : : : ;m, and the approximation error bL
(PCA).
n 2 f0; : : : ; n0) of the delivered feature vectors x(i), for i = 1; : : : ;m, is an input (or hyper-)
parameter of Algorithm 15. Two extreme cases are n = 0 (maximum compression) and
n = n0 (no compression). We  nally note that the choice for the orthonormal eigenvectors
in (9.8) might not be unique. Depending on the sample covariance matrix Q, there might
di erent sets of orthonormal vectors that correspond to the same eigenvalue of Q. Thus,
for a given length n of the new feature vectors, there might be several di erent matrices W
that achieve the same (optimal) reconstruction error bL
(PCA).
Computationally, Algorithm 15 essentially amounts to an EVD of the sample covariance
matrix Q (9.9). The EVD of Q provides not only the optimal compression matrix WPCA
but also the measure bL
(PCA) for the information loss incurred by replacing the original data
points z(i) 2 Rn0 with the shorter feature vector x(i) 2 Rn. We quantify this information
loss by the approximation error obtained when using the compression matrix WPCA (and
corresponding reconstruction matrix Ropt =WT
PCA),
bL
(PCA) := bL
􀀀
WPCA; |R{ozp}t
=WT
PCA
j D
 
=
n0 X
j=n+1
 j : (9.11)
As depicted in Figure 9.2, the approximation error bL
(PCA) decreases with increasing number n
223
of PCs used for the new features (9.5). For the extreme case n=0, where we completely ignore
the raw feature vectors z(i), the optimal reconstruction error is bL
(PCA) = (1=m)
Pm
i=1



z(i)

 
2
.
The other extreme case n = n0 allows to use the raw features directly as the new features
x(i) = z(i). This extreme case means no compression at all, and trivially results in a zero
reconstruction error bL
(PCA)=0.
n0
0
2
4
6
8
n
bL
(PCA)
Figure 9.2: Reconstruction error bL
(PCA) (see (9.11)) of PCA for varying number n of PCs.
9.2.1 Combining PCA with linear regression
One important use case of PCA is as a pre-processing step within an overall ML problem such
as linear regression (see Section 3.1). As discussed in Chapter 7, linear regression methods
are prone to over tting whenever the data points are characterized by raw feature vectors z
whose length n0 exceeds the number m of labeled data points used in ERM.
One simple but powerful strategy to avoid over tting is to preprocess the raw features
z(i) 2 Rn0 , for i = 1; : : : ;m by applying PCA. Indeed, PCA Algorithm 15 delivers feature
vectors x(i) 2 Rn of prescribed length n. Thus, choosing the parameter n such that n < m
will typically prevent the follow-up linear regression method from over tting.
9.2.2 How To Choose Number of PC?
There are several aspects which can guide the choice for the number n of PCs to be used as
features.
224
• To generate data visualizations we might use either n = 2 or n = 3.
• We should choose n su ciently small such that the overall ML method  ts the available
computational resources.
• Consider using PCA as a pre-processing for linear regression (see Section 3.1). In
particular, we can use the learnt feature vectors x(i) delivered by PCA as the feature
vectors of data points in plain linear regression methods. To avoid over tting, we
should choose n < m (see Chapter 7).
• Choose n large enough such that the resulting approximation error bL
(PCA) is reasonably
small (see Figure 9.2).
9.2.3 Data Visualisation
If we use PCA with n = 2, we obtain feature vectors x(i) = WPCAz(i) (see (9.5)) which
can be depicted as points in a scatterplot (see Section 2.1.3). As an example, consider data
points z(i) obtained from historic recordings of Bitcoin statistics. Each data point z(i) 2 Rn0
is a vector of length n0 = 6. It is di cult to visualise points in an Euclidean space Rn0 of
dimension n0 > 2. Therefore, we apply PCA with n = 2 which results in feature vectors
x(i) 2 R2. These new feature vectors (of length 2) can be depicted conveniently as the
scatterplot in Figure 9.3.
9.2.4 Extensions of PCA
We now brie
y discuss variants and extensions of the basic PCA method.
• Kernel PCA [58, Ch.14.5.4]: The PCA method is most e ective if the raw feature
vectors of data points are concentrated around a n-dimensional linear subspace of
Rn0 . Kernel PCA extends PCA to data points that are located near a low-dimensional
manifold which might be highly non-linear. This is achieved by applying PCA to transformed
feature vectors instead of the original raw feature vectors. Kernel PCA  rst
applies a (typically non-linear) feature map   to the raw feature vectors z(i) (see Section
3.9) and applies PCA to the transformed feature vectors  
􀀀
z(i)
 
, for i = 1; : : : ;m.
• Robust PCA [155]: The basic PCA Algorithm 15 is sensitive to outliers, i.e., a small
number of data points with signi cantly di erent statistical properties than the bulk
225
􀀀8;000􀀀6;000􀀀4;000􀀀2;000 2;000 4;000 6;000
􀀀400
􀀀200
200
400
second PC x2
 rst PC x1
Figure 9.3: A scatterplot of data points with feature vectors x(i) =
􀀀
x(i)
1 ; x(i)
2
 T
whose entries
are the  rst two PCs of the Bitcoin statistics z(i) of the i-th day.
of data points. This sensitivity might be attributed to the properties of the squared
Euclidean norm (9.3) which is used in PCA to measure the reconstruction error (9.1).
We have seen in Chapter 3 that linear regression (see Section 3.1 and 3.3) can be made
robust against outliers by replacing the squared error loss with another loss function.
In a similar spirit, robust PCA replaces the squared Euclidean norm with another norm
that is less sensitive to having very large reconstruction errors (9.1) for a small number
of data points (which are outliers).
• Sparse PCA [58, Ch.14.5.5]: The basic PCA method transforms the raw feature
vector z(i) of a data point to a new (shorter) feature vector x(i). In general each entry
x(i)
j of the new feature vector will depend on each entry of the raw feature vector
z(i). More precisely, the new feature x(i)
j depends on all raw features z(i)
j0 for which
the corresponding entry Wj;j0 of the matrix W = WPCA (9.8) is non-zero. For most
datasets, all entries of the matrix WPCA will typically be non-zero.
In some applications of linear dimensionality reduction we would like to construct new
features that depend only on a small subset of raw features. Equivalently we would
like to learn a linear compression map W (9.5) such that each row of W contains only
few non-zero entries. To this end, sparse PCA enforces the rows of the compression
matrix W to contain only a small number of non-zero entries. This enforcement can
be implement either using additional constraints on W or by adding a penalty term
226
to the reconstruction error (9.7).
• Probabilistic PCA [118, 137]: We have motivated PCA as a method for learning
an optimal linear compression map (matrix) (9.5) such that the compressed feature
vectors allows to linearly reconstruct the original raw feature vector with minimum
reconstruction error (9.7). Another interpretation of PCA is that of a method that
learns a subspace of Rn0 that best  ts the distribution of the raw feature vectors z(i),
for i = 1; : : : ;m. This optimal subspace is precisely the subspace spanned by the rows
of WPCA (9.8).
Probabilistic PCA (PPCA) interprets the raw feature vectors z(i) as realizations of iid
RVs. These realizations are modelled as
z(i) =WTx(i) + "(i), for i = 1; : : : ; m: (9.12)
Here, W 2 Rn n0 is some unknown matrix with orthonormal rows. The rows of W
span the subspace around which the raw features are concentrated. The vectors x(i)
in (9.12) are realizations of iid RVs whose common probability distribution is N(0; I).
The vectors "(i) are realizations of iid RVs whose common probability distribution is
N(0;  2I) with some  xed but unknown variance  2. Note thatW and  2 parametrize
the joint probability distribution of the feature vectors z(i) via (9.12). PPCA amounts
to maximum likelihood estimation (see Section 3.12) of the parametersWand  2. This
maximum likelihood estimation problem can be solved using computationally e cient
estimation techniques such as EM [137, Appendix B]. The implementation of PPCA
via EM also o ers a principled approach to handle missing data. Roughly speaking,
the EM method allows to use the probabilistic model (9.12) to estimate missing raw
features [137, Sec. 4.1].
9.3 Feature Learning for Non-Numeric Data
We have motivated dimensionality reduction methods as transformations of (very long) raw
feature vectors to a new (shorter) feature vector x such that it allows to reconstruct the
raw features z with minimum reconstruction error (9.1). To make this requirement precise
we need to de ne a measure for the size of the reconstruction error and specify the class of
possible reconstruction maps. PCA uses the squared Euclidean norm (9.7) to measure the
reconstruction error and only allows for linear reconstruction maps (9.6).
227
Alternatively, we can view dimensionality reduction as the generation of new feature vectors
x(i) that maintain the intrinsic geometry of the data points with their raw feature vectors
z(i). Di erent dimensionality reduction methods use di erent concepts for characterizing the
\intrinsic geometry" of data points. PCA de nes the intrinsic geometry of data points using
the squared Euclidean distances between feature vectors. Indeed, PCA produces feature vectors
x(i) such that for data points whose raw feature vectors have small squared Euclidean
distance, also the new feature vectors x(i) will have small squared Euclidean distance.
Some application domains generate data points for which the Euclidean distances between
raw feature vectors does not re
ect the intrinsic geometry of data points. As a point in
case, consider data points representing scienti c articles which can be characterized by the
relative frequencies of words from some given set of relevant words (dictionary). A small
Euclidean distance between the resulting raw feature vectors typically does not imply that
the corresponding text documents are similar. Instead, the similarity between two articles
might depend on the number of authors that are contained in author lists of both papers.
We can represent the similarities between all articles using a similarity graph whose nodes
represent data points which are connected by an edge (link) if they are similar (see Figure
8.9).
Consider a dataset D =
􀀀
z(1); : : : ; z(m)
 
whose intrinsic geometry is characterized by an
unweighted similarity graph G =
􀀀
V := f1; : : : ;mg; E
 
. The node i 2 V represents the i-th
data point z(i). Two nodes are connected by an undirected edge if the corresponding data
points are similar.
We would like to  nd short feature vectors x(i), for i = 1; : : : ;m, such that two data
points i; i0, whose feature vectors x(i); x(i0) have small Euclidean distance, are well-connected
to each other. This informal requirement must be made precise by a measure for how well
two nodes of an undirected graph are connected. We refer the reader to literature on network
theory for an overview and details of various connectivity measures [103].
Let us discuss a simple but powerful technique to map the nodes i 2 V of an undirected
graph G to (short) feature vectors x(i) 2 Rn. This map is such that the Euclidean distances
between the feature vectors of two nodes re
ect their connectivity within G. This technique
uses the Laplacian matrix L 2 R(i) which is de ned for an undirected graph G (with node
228
set V = f1; : : : ;mg) element-wise
Li;i0 :=
8>>><
>>>:
􀀀1 , if fi; i0g 2 E
d(i) , if i = i0
0 otherwise.
: (9.13)
Here, d(i) :=
  
fi0 : fi; i0g 2 Eg
  
denotes the number of neighbours (the degree) of node i 2 V.
It can be shown that the Laplacian matrix L is psd [147, Proposition 1]. Therefore we can
 nd a set of orthonormal eigenvectors
u(1); : : : ; u(m) 2 Rm (9.14)
with corresponding (ordered in a non-decreasing fashion) eigenvalues  1   : : :    m of L.
For a given number n, we construct the feature vector
x(i) :=
􀀀
u(1)
i ; : : : ; u(n)
i
 T
for the ith data point. Here, we used the entries of of the  rst n eigenvectors (9.14). It can
be shown that the Euclidean distances between the feature vectors x(i), for i = 1; : : : ;m,
re
ect the connectivities between data points i = 1; : : : ;m in the similarity graph G. For a
more precise statement of this informal claim we refer to the excellent tutorial [147].
To summarize, we can construct numeric feature vectors for (non-numeric ) data points
via the eigenvectors of the Laplacian matrix of a similarity graph for the data points. Algorithm
16 summarizes this feature learning method which requires as its input a similarity
graph for the data points and the desired number n of numeric features. Note that Algorithm
16 does not make any use of the Euclidean distances between raw feature vectors and uses
solely the similarity graph G to determine the intrinsic geometry of D.
9.4 Feature Learning for Labeled Data
We have discussed PCA as a linear dimensionality reduction method. PCA learns a compression
matrix that maps raw features z(i) of data points to new (much shorter) feature
vectors x(i). The feature vectors x(i) determined by PCA depend solely on the raw feature
vectors z(i) of a given dataset D. In particular, PCA determines the compression matrix such
that the new features allow for a linear reconstruction (9.6) with minimum reconstruction
229
Algorithm 16 Feature Learning for Non-Numeric Data
Input: dataset D = fz(i) 2 Rn0gmi
=1; similarity graph G; number n of features to be constructed
for each data point.
1: construct the Laplacian matrix L of the similarity graph (see (9.13))
2: compute EVD of L to obtain n orthonormal eigenvectors (9.14) corresponding to the
smallest eigenvalues of L
3: for each data point i, construct feature vector
x(i) :=
􀀀
u(1)
i ; : : : ; u(n)
i
 T
2 Rn (9.15)
Output: x(i), for i = 1; : : : ;m
error (9.7).
For some application domains we might not only have access to raw feature vectors
but also to the label values y(i) of the data points in D. Indeed, dimensionality reduction
methods might be used as pre-processing step within a regression or classi cation problem
that involves a labeled training set. However, in its basic form, PCA (see Algorithm 15) does
not allow to exploit the information provided by available labels y(i) of data points z(i). For
some datasets, PCA might deliver feature vectors that are not very relevant for the overall
task of predicting the label of a data point.
Let us now discuss a modi cation of PCA that exploits the information provided by
available labels of the data points. The idea is to learn a linear construction map (matrix)
W such that the new feature vectors x(i) = Wz(i) allow to predict the label y(i) as good as
possible. We restrict the prediction to be linear,
^y(i) := rTx(i) = rTWz(i); (9.16)
with some weight vector r 2 Rn.
While PCA is motivated by minimizing the reconstruction error (9.1), we now aim at
minimizing the prediction error ^y(i) 􀀀 y(i). In particular, we assess the usefulness of a given
230
pair of construction map W and predictor r (see (9.16)), using the empirical risk
bL
􀀀
W; r j D
 
:= (1=m)
Xm
i=1
􀀀
y(i) 􀀀 ^y(i) 2
(9.16)
= (1=m)
Xm
i=1
􀀀
y(i) 􀀀 rTWz(i) 2
: (9.17)
to guide the learning of a compressing matrixW and corresponding linear predictor weights
r ((9.16)).
The optimal matrix W that minimizes the empirical risk (9.17) can be obtained via the
EVD (9.10) of the sample covariance matrix Q (9.9). Note that we have used the EVD of
Q already for PCA in Section 9.2 (see (9.8)). Remember that PCA uses the n eigenvectors
u(1); : : : ; u(n) corresponding to the n largest eigenvalues of Q. In contrast, to minimize (9.17),
we need to use a di erent set of eigenvectors in the rows of W in general. To  nd the right
set of n eigenvectors, we need the sample cross-correlation vector
q := (1=m)
Xm
i=1
y(i)z(i): (9.18)
The entry qj of the vector q estimates the correlation between the raw feature z(i)
j and the
label y(i). We then de ne the index set
S := fj1; : : : ; jng such that
􀀀
qj
 2
= j  
􀀀
qj0
 2
= j0 for any j 2 S; j0 2 f1; : : : ; n0g =2 S:
(9.19)
It can then be shown that the rows of the optimal compression matrixWare the eigenvectors
u(j) with j 2 S. We summarize the overall feature learning method in Algorithm 17.
The main focus of this section is on regression problems that involve data points with
numeric labels (e.g., from the label space Y = R). Given the raw features and labels of the
data point in the dataset D, Algorithm 17 determines new feature vectors x(i) that allow to
linearly predict a numeric label with minimum squared error. A similar approach can be
used for classi cation problems involving data points with a  nite label space Y. Linear (or
Fisher) discriminant analysis aims at constructing a compression matrix W such that the
learnt features x = Wz of a data point allow to predict its label y as accurately as possible
[58].
231
Algorithm 17 Linear Feature Learning for Labeled Data
Input: dataset
􀀀
z(1); y(1)
 
; : : : ;
􀀀
z(m); y(m)
 
with raw features z(i) 2 Rn0 and numeric labels
y(i) 2 R ; length n of new feature vectors.
1: compute EVD (9.10) of the sample covariance matrix (9.9) to obtain orthonormal eigenvectors
􀀀
u(1); : : : ; u(n0)
 
corresponding to (decreasingly ordered) eigenvalues  1    2  
: : :    n0   0
2: compute the sample cross-correlation vector (9.18) and, in turn, the sequence
􀀀
q1
 2
= 1; : : : ;
􀀀
qn0
 2
= n0 (9.20)
3: determine indices j1; : : : ; jn of n largest elements in (9.20)
4: construct compression matrix W :=
􀀀
u(j1); : : : ; u(jn)
 T
2 Rn n0
5: compute feature vector x(i) = Wz(i)
Output: x(i), for i = 1; : : : ;m, and compression matrix W.
9.5 Privacy-Preserving Feature Learning
Many important application domains of ML involve sensitive data that is subject to data
protection law [148]. Consider a health-care provider (such as a hospital) holding a large
database of patient records. From a ML perspective this databases is nothing but a (typically
large) set of data points representing individual patients. The data points are characterized
by many features including personal identi ers (name, social security number), bio-physical
parameters as well as examination results . We could apply ML to learn a predictor for the
risk of particular disease given the features of a data point.
Given large patient databases, the ML methods might not be implemented locally at the
hospital but using cloud computing. However, data protection requirements might prohibit
the transfer of raw patient records that allow to match individuals with bio-physical properties.
In this case we might apply feature learning methods to construct new features for
each patient such that they allow to learn an accurate hypothesis for predicting a disease
but do not allow to identify sensitive properties of the patient such as its name or a social
security number.
Let us formalize the above application by characterizing each data point (patient in the
hospital database) using raw feature vector z(i) 2 Rn0 and a sensitive numeric property  (i).
We would like to  nd a compression map W such that the resulting features x(i) = Wz(i)
232
do not allow to accurately predict the sensitive property  (i). The prediction of the sensitive
property is restricted to be a linear ^ (i) := rTx(i) with some weight vector r.
Similar to Section 9.4 we want to  nd a compression matrix W that transforms, in a
linear fashion, the raw feature vector z 2 Rn0 to a new feature vector x 2 Rn. However the
design criterion for the optimal compression matrixWwas di erent in Section 9.4 where the
new feature vectors should allow for an accurate linear prediction of the label. In contrast,
here we want to construct feature vectors such that there is no accurate linear predictor of
the sensitive property  (i).
As in Section 9.4, the optimal compression matrixWis given row-wise by the eigenvectors
of the sample covariance matrix (9.9). However, the choice of which eigenvectors to use is
di erent and based on the entries of the sample cross-correlation vector
c := (1=m)
Xm
i=1
 (i)z(i): (9.21)
We summarize the construction of the optimal privacy-preserving compression matrix and
corresponding new feature vectors in Algorithm 18.
Algorithm 18 Privacy Preserving Feature Learning
Input: dataset
􀀀
z(1); y(1)
 
; : : : ;
􀀀
z(m); y(m)
 
; each data point characterized by raw features
z(i) 2 Rn0 and (numeric) sensitive property  (i) 2 R; number n of new features.
1: compute the EVD (9.10) of the sample covariance matrix (9.9) to obtain orthonormal
eigenvectors
􀀀
u(1); : : : ; u(n0)
 
corresponding to (decreasingly ordered) eigenvalues  1  
 2   : : :    n0   0
2: compute the sample cross-correlation vector (9.21) and, in turn, the sequence
􀀀
c1
 2
= 1; : : : ;
􀀀
cn0
 2
= n0 (9.22)
3: determine indices j1; : : : ; jn of n smallest elements in (9.22)
4: construct compression matrix W :=
􀀀
u(j1); : : : ; u(jn)
 T
2 Rn n0
5: compute feature vector x(i) = Wz(i)
Output: feature vectors x(i), for i = 1; : : : ;m, and compression matrix W.
Algorithm 18 learns a map W to extract privacy-preserving features out of the raw
feature vector of a data point. These new features are privacy-preserving as they do not
allow to accurately predict (in a linear fashion) a sensitive property   of the data point.
233
Another formalization for the preservation of privacy can be obtained using informationtheoretic
concepts. This information-theoretic approach interprets data points, their feature
vector and sensitive property, as realizations of RVs. It is then possible to use the mutual
information between new features x and the sensitive (private) property   as an optimization
criterion for learning a compression map h (Section 9.1). The resulting feature learning
method (referred to as privacy-funnel) di ers from Algorithm 18 not only in the optimization
criterion for the compression map but also in that it allows it to be non-linear [89, 128].
9.6 Random Projections
Note that PCA uses an EVD of the sample covariance matrix Q (9.9). The computational
complexity (e.g., measured by number of multiplications and additions) for computing this
EVD is lower bounded by n minfn02;m2g [47, 138]. This computational complexity can be
prohibitive for ML applications with n0 and m being of the order of millions or even billions.
There is a computationally cheap alternative to PCA (Algorithm 15) for  nding a useful
compression matrix W in (9.5). This alternative is to construct the compression matrix W
entry-wise
Wj;j0 := aj;j0 with aj;j0   p(a): (9.23)
The matrix entries (9.23) are realizations ai;j of iid RVs with some common probability
distribution p(a). Di erent choices for the probability distribution p(a) have been studied
in the literature [40]. The Bernoulli distribution is used to obtain a compression matrix
with binary entries. Another popular choice for p(a) is the multivariate normal (Gaussian)
distribution.
Consider data points whose raw feature vectors z are located near a s-dimensional subspace
of Rn0 . The feature vectors x obtained via (9.5) using a random matrix (9.23) allows
to reconstruct the raw feature vectors z with high probability whenever
n   Cs log n0: (9.24)
The constant C depends on the maximum tolerated reconstruction error   (such that kbz
􀀀
zk22
    for any data point) and the probability that the features x (see (9.23)) allow for a
maximum reconstruction error   [40, Theorem 9.27.].
234
9.7 Dimensionality Increase
The focus of this chapter is on dimensionality reduction methods that learn a feature map
delivering new feature vectors which are (signi cantly) shorter than the raw feature vectors.
However, it might sometimes be bene cial to learn a feature map that delivers new feature
vectors which are longer than the raw feature vectors. We have already discussed two
examples for such feature learning methods in Sections 3.2 and 3.9. Polynomial regression
maps a single raw feature z to a feature vector containing the powers of the raw feature
z. This allows to use apply linear predictor maps to the new feature vectors to obtain
predictions that depend non-linearly on the raw feature z. Kernel methods might even use a
feature map that delivers feature vectors belonging to an in nite-dimensional Hilbert space
[125].
Mapping raw feature vectors into higher-dimensional (or even in nite-dimensional) spaces
might be useful if the intrinsic geometry of the data points is simpler when looked at in the
higher-dimensional space. Consider a binary classi cation problem where data points are
highly inter-winded in the original feature space (see Figure 3.7). Loosely speaking, mapping
into higher-dimensional feature space might "
atten-out" a non-linear decision boundary
between data points. We can then apply linear classi ers to the higher-dimensional features
to achieve accurate predictions.
9.8 Exercises
Exercise 9.1. Computational Burden of Many Features Discuss the computational
complexity of linear regression. How much computation do we need to compute the linear
predictor that minimizes the average squared error on a training set?
Exercise 9.2. Power Iteration The key computational step of PCA amounts to an
EVD of the psd matrix (9.9). Consider an arbitrary initial vector u(r) and the sequence
obtained by iterating
u(r+1) := Qu(r)=



Qu(r)



: (9.25)
What (if any) conditions on the initialization u(r) ensure that the sequence u(r) converges to
the eigenvector u(1) of Q that corresponds to its largest eigenvalue  1?
Exercise 9.3. Linear Classi ers with High-Dimensional Features Consider a training
set D consisting of m = 1010 labeled data points
􀀀
z(1); y(1)
 
; : : : ;
􀀀
z(m); y(m)
 
with raw feature
235
vectors z(i) 2 R4000 and binary labels y(i) 2 f􀀀1; 1g. Assume we have used a feature learning
method to obtain the new features x(i) 2 f0; 1gn with n = m and such that the only nonzero
entry of x(i) is x(i)
i = 1, for i = 1; : : : ;m. Can you  nd a linear classi er that perfectly
classi es the training set?
236
Chapter 10
Transparent and Explainable ML
The successful deployment of ML methods depends on their transparency or explainability.
We formalize the notion of an explanation and its e ect using a simple probabilistic model
in Section 10.1. Roughly speaking, an explanation is any artefact. such as a list of relevant
features or a reference data point from a training set, that coneys information about a ML
method and its predictions. Put di erently, explaining a ML method should reduce the
uncertainty (of a human end-user) about its predictions.
Explainable ML is umbrella term for techniques that make ML method transparent or
explainable. Providing explanations for the predictions of a ML method is particularly
important when these predictions inform decision making [24]. It is increasingly becoming
a legal requirement to provide explanations for automated decision making systems [54].
Even for applications where predictions are not directly used to inform far-reaching decisions,
providing explanations is important. The human end users have an intrinsic desire for
explanations that resolve the uncertainty about the prediction. This is known as the \need
for closure" in psychology [30, 75]. Beside legal and psychological requirements, providing
explanations for predictions might also be useful for validating and verifying ML methods.
Indeed, the explanations of ML methods (and its predictions) can point the user (which
might be a \domain expert") to incorrect modelling assumptions used by the ML method
[38].
Explainable ML is challenging since explanations must be tailored (personalized) to human
end-users with varying backgrounds and in di erent contexts [87]. The user background
includes the formal education as well as the individual digital literacy. Some users might
have received university-level education in ML, while other users might have no relevant
formal training (such as an undergraduate course in linear algebra). Linear regression with
237
few features might be perfectly interpretable for the  rst group but be considered a \black
box" for the latter. To enable tailored explanations we need to model the user background
as relevant for understanding the ML predictions.
This chapter discusses explainable ML methods that have access to some user signal
or feedback for some data points. Such a user signal might be obtained in various ways,
including answers to surveys or bio-physical measurements collected via wearables or medical
diagnostics. The user signal is used to determine (to some extent) the end-user background
and, in turn, to tailor the delivered explanations for this end-user.
Existing explainable ML methods can be roughly divided into two categories. The  rst
category is referred to as \model-agnostic" [24]). Model-agnostic methods do not require
knowledge of the detailed work principles of a ML method. These methods do not require
knowledge of the hypothesis space used by a ML method but learn how to explain its
predictions by observing them on a training set [23].
A second category of explainable ML methods, sometimes referred to as \white-box"
methods [24], uses ML methods that are considered as intrinsically explainable. The intrinsic
explainability of a ML method depends crucially on its choice for the hypothesis space (see
Section 2.2). This chapter discusses one recent method from each of the two explainable ML
categories [72, 69]. The common theme of both methods is the use of information-theoretic
concepts to measure the usefulness of explanations [28].
Section 10.1 discusses a recently proposed model-agnostic approach to explainable ML
that constructs tailored explanations for the predictions of a given ML method [72]. This
approach does not require any details about the internal mechanism of a ML method whose
predictions are to be explained. Rather, this approach only requires a (su ciently large)
training set of data points for which the predictions of the ML method are known.
To tailor the explanations to a particular user, we use the values of a user (feedback)
signal provided for the data points in the training set. Roughly speaking, the explanations
are chosen such that they maximally reduce the \surprise" or uncertainty that the user has
about the predictions of the ML method.
Section 10.2 discusses an example for a ML method that uses a hypothesis space that is
intrinsically explainable [69]. We construct an explainable hypothesis space by appropriate
pruning of a given hypothesis space such as linear maps (see Section 3.1) or non-linear
maps represented by either an ANN (see Section 3.11) or decision trees (see Section 3.10).
This pruning is implemented via adding a regularization term to ERM (4.3), resulting in an
instance of SRM (7.2) which we refer to as explainable empirical risk minimization (EERM).
238
The regularization term favours hypotheses that are explainable to a user. Similar to the
method in Section 10.1, the explainability of a map is quanti ed by information theoretic
quantities. For example, if the original hypothesis space is the set of linear maps using a
large number of features, the regularization term might favour maps that depend only on
few features that are interpretable. Hence, we can interpret EERM as a feature learning
method that aims at learning relevant and interpretable features (see Chapter 9).
10.1 Personalized Explanations for ML Methods
Consider a ML application involving data points with features x =
􀀀
x1; : : : ; xn
 T
2 Rn and
label y 2 R. We use a ML method that reads in some labelled data points
􀀀
x(1); y(1) 
;
􀀀
x(2); y(2) 
; : : : ;
􀀀
x(m); y(m) 
; (10.1)
and learns a hypothesis
h( ) : Rn ! R : x 7! ^y = h(x): (10.2)
The precise working principle of this ML method for how to learn this hypothesis h is not
relevant in what follows.
user u consumig
prediction ^y
ML method
explanation e prediction y^
Figure 10.1: An explanation e provides additional information I(^y; eju) to a user u about
the prediction ^y.
The learnt predictor h(x) is applied to the features of a data point to obtain the predicted
label ^y := h(x). The prediction ^y is then delivered to a human end-user (see Figure 10.1).
Depending on the ML application, this end-user might be a streaming service subscriber
[48], a dermatologist [37] or a city planner [157].
239
Human users of ML methods often have some conception or model for the relation between
features x and label y of a data point. This intrinsic model might vary signi cantly
between users with di erent (social or educational) background. We will model the user
understanding of a data point by a \user summary" u 2 R. The summary is obtained by
a (possibly stochastic) map from the features x of a data point. For ease of exposition, we
focus on summaries obtained by a deterministic map
u( ) : Rn ! R : x 7! u := u(x): (10.3)
However, the resulting explainable ML method can be extended to user feedback u modelled
as a stochastic maps. In this case, the user feedback u is characterized by a probability
distribution p(ujx).
The user feedback u is determined by the features x of a data point. We might think
of the value u for a speci c data point as a signal that re
ects how the human end-user
interprets (or perceives) the data point, given her knowledge (including formal education)
and the context of the ML application. We do not assume any knowledge about the details
for how the signal value u is formed for a speci c data point. In particular, we do not know
any properties of the map u( ) : x 7! u.
The above approach is quite 
exible as it allows for very di erent forms of user summaries.
The user summary could be the prediction obtained from a simpli ed model, such as linear
regression using few features that the user anticipates as being relevant. Another example
for a user summary u could be a higher-level feature, such as eye spacing in facial pictures,
that the user considers relevant [67].
Note that, since we allow for an arbitrary map in (10.3), the user summary u(x) obtained
for a random data point with features x might be correlated with the prediction ^y = h(x).
As an extreme case, consider a very knowledgable user that is able to predict the label of
any data point from its features as well as the ML method itself. In this case, the maps
(10.2) and (10.3) might be nearly identical. However, in general the predictions delivered by
the learnt hypothesis (10.2) will be di erent from the user summary u(x).
We formalize the act of explaining a prediction ^y = h(x) as presenting some additional
quantity e to the user (see Figure 10.1). This explanation e can be any artefact that helps
the user to understand the prediction ^y, given her understanding u of the data point. Loosely
speaking, the aim of providing explanation e is to reduce the uncertainty of the user u about
the prediction ^y [75].
For the sake of exposition, we construct explanations e that are obtained via a determin-
240
istic map
e( ) : Rn ! R : x 7! e := e(x); (10.4)
from the features x of a data point. However, the explainable ML methods in this chapter
can be generalized without di culty to handle explanations obtained from a stochastic map.
In the end, we only require the speci cation of the conditional probability distribution p(ejx).
The explanation e (10.4) depends only on the features x but not explicitly on the prediction
^y. However, our method for constructing the map (10.4) takes into account the
properties of the predictor map h(x) (10.2). In particular, Algorithm 19 below requires as
input the predicted labels ^y(i) for a set of data points (that serve as a training set for our
method).
To obtain comprehensible explanations that can be computed e ciently, we must typically
restrict the space of possible explanations to a small subset F of maps (10.4). This is
conceptually similar to the restriction of the space of possible predictor functions in a ML
method to a small subset of maps which is known as the hypothesis space.
10.1.1 Probabilistic Data Model for XML
In what follows, we model data points as realizations of iid RVs with common (joint) probability
distribution p(x; y) of features and label (see Section 2.1.4). Modelling the data points
as realizations of RVs implies that the user summary u, prediction ^y and explanation e are
also realizations of RVs. The joint distribution p(u; ^y; e; x; y) conforms with the Bayesian
network [111] depicted in Figure 10.2. Indeed,
p(u; ^y; e; x; y) = p(ujx)   p(ejx)   p(^yjx)   p(x; y): (10.5)
We measure the amount of additional information provided by an explanation e for a
prediction ^y to some user u via the conditional mutual information (MI) [28, Ch. 2 and 8]
I(e; ^yju) := E
 
log
p(^y; eju)
p(^yju)p(eju)
 
: (10.6)
The conditional MI I(e; ^yju) can also be interpreted as a measure for the amount by which
the explanation e reduces the uncertainty about the prediction ^y which is delivered to some
user u. Providing the explanation e serves the apparent human need to understand observed
phenomena, such as the predictions from a ML method [75].
241
data point
(x; y)
some user
explanation
e
user
signal u prediction
^y = h(x)
Figure 10.2: A simple probabilistic graphical model (a Bayesian network [84, 79]) for explainable
ML. We interpret data points (with features x and label y) along with the user
summary u, e and predicted label ^y as realizations of RVs. These RVs satisfy conditional
independence relations encoded by the directed links of the graph [79]. Given the data
point, the predicted label ^y, the explanation e and the user summary u are conditionally
independent. This conditional independence is trivial if all these quantities are obtained
from deterministic maps applied to the features x of the data point.
10.1.2 Computing Optimal Explanations
Capturing the e ect of an explanation using the probabilistic model (10.6) o ers a principled
approach to computing an optimal explanation e. We require the optimal explanation
e  to maximize the conditional MI (10.6) between the explanation e and the prediction ^y
conditioned on the user summary u of the data point.
Formally, an optimal explanation e  solves
I(e ; ^yju) = sup
e2F
I(e; ^yju): (10.7)
The choice for the subset F of valid explanations o ers a trade-o  between comprehensibility,
informativeness and computational cost incurred by an explanation e  (solving (10.7)).
The maximization problem (10.7) for obtaining optimal explanations is similar to the
approach in [23]. However, while [23] uses the unconditional MI between explanation and
prediction, (10.7) uses the conditional MI given the user summary u. Therefore, (10.7)
delivers personalized explanations that are tailored to the user who is characterized by the
summary u.
It is important to note that the construction (10.7) allows for many di erent forms of
242
explanations. An explanation could be a subset of features of a data point (see [116] and
Section 10.1.2). More generally, explanations could be obtained from simple local statistics
(averages) of features that are considered related, such as nearby pixels in an image or
consecutive amplitude values of an audio signal. Instead of individual features, carefully
chosen data points from a training set can also serve as an explanation [92, 117].
Let us illustrate the concept of optimal explanations (10.7) using linear regression. We
model the features x as a realization of a multivariate normal random vector with zero mean
and covariance matrix Cx,
x   N(0;Cx): (10.8)
The predictor and the user summary are linear functions
^y := wTx, and u := vTx: (10.9)
We construct explanations via subsets of individual features xj that are considered most
relevant for a user to understand the prediction ^y (see [98, De nition 2] and [97]). Thus, we
consider explanations of the form
e := fxjgj2E with some subset E   f1; : : : ; ng: (10.10)
The complexity of an explanation e is measured by the number jEj of features that
contribute to it. We limit the complexity of explanations by a  xed (small) sparsity level,
jEj   s(  n): (10.11)
Modelling the feature vector x as Gaussian (10.8) implies that the prediction ^y and user
summary u obtained from (10.9) is jointly Gaussian for a given E (10.4). Basic properties
of multivariate normal distributions [28, Ch. 8], allow to develop (10.7) as
max
E f1;:::;ng
jEj s
I(e; ^yju)
= h(^yju) 􀀀 h(^yju; E)
= (1=2) logC^yju 􀀀 (1=2) log detC^yju;D(train)
= (1=2) log  2
^yju 􀀀 (1=2) log  2
^yju;D(train) : (10.12)
243
Here,  2
^yju denotes the conditional variance of the prediction ^y, conditioned on the user
summary u. Similarly,  2
^yju;E denotes the conditional variance of ^y, conditioned on the user
summary u and the subset fxjgj2E of features. The last step in (10.12) follows from the fact
that ^y is a scalar random variable.
The  rst component of the  nal expression of (10.12) does not depend on the index set E
used to construct the explanation e (see (10.10)). Therefore, the optimal choice for E solves
sup
jEj s
􀀀(1=2) log  2
^yju;E : (10.13)
The maximization (10.13) is equivalent to
inf
jEj s
 2
^yju;E : (10.14)
In order to solve (10.14), we relate the conditional variance  2
^yju;E to a particular decomposition
^y =  u +
X
j2E
 jxj + ": (10.15)
For an optimal choice of the coe cients   and  j , the variance of the error term in (10.15)
is given by  2
^yju;E . Indeed,
min
 ; j2R
E
 􀀀
^y 􀀀  u 􀀀
X
j2E
 jxj
 2 
=  2
^yju;e: (10.16)
Inserting (10.29) into (10.14), an optimal choice E (of feature) for the explanation of
prediction ^y to user u is obtained from
inf
jEj s
min
 ; j2R
E
 􀀀
^y 􀀀  u 􀀀
X
j2E
 jxj
 2 
(10.17)
= min
k k0 s
E
 􀀀
^y 􀀀  u 􀀀  Tx
 2 
: (10.18)
An optimal subset Eopt of features de ning the explanation e (10.10) is obtained from any
solution  opt of (10.18) via
Eopt = supp  opt: (10.19)
Section 10.1.2 uses the probabilistic model (10.8) to construct optimal explanations via
the (support of the) solutions  opt of the sparse linear regression problem (10.18). To obtain
244
a practical algorithm for computing (approximately) optimal explanations (10.19), we approximate
the expectation in (10.18) using an average over the training set
􀀀
x(i); ^y(i); u(i)
 
, for
i = 1; : : : ;m. This resulting method for computing personalized explanations is summarized
in Algorithm 19.
Algorithm 19 XML Algorithm
Input: explanation complexity s, training set
􀀀
x(i); ^y(i); u(i)
 
for i = 1; : : : ;m
1: compute b   by solving
b   2 argmin
 2R;k k0 s
(1=m)
Xm
i=1
􀀀
^y(i)􀀀 u(i)􀀀 Tx(i) 2
(10.20)
Output: feature set b E := supp b 
Algorithm 19 is interactive in the sense that the user has to provide a feedback signal
u(i) for the data points with features x(i). Based on the user feedback u(i), for i = 1; : : : ;m,
Algorithm 19 learns an optimal subset E of features (10.10) that are used for the explanation
of predictions.
The sparse regression problem (10.20) becomes intractable for large feature length n.
However, if the features are weakly correlated with each other and the user summary u, the
solutions of (10.20) can be found by e cient convex optimization methods. One popular
method to (approximately) solve sparse regression (10.20) is the Lasso (see Section 3.4),
b 2 argmin
 2R; 2Rn
(1=m)
Xm
i=1
􀀀
^y(i)􀀀 u(i)􀀀 Tx(i) 2
+ k k1: (10.21)
There is large body of work that studies the choice of Lasso parameter   in (10.21) such that
solutions (10.21) coincide with the solutions of (10.20) (see [59, 143] and references therein).
The proper choice for   typically requires knowledge of statistical properties of data. If such
a probabilistic model is not available, the choice of   can be guided by simple validation
techniques (see Section 6.2).
10.2 Explainable Empirical Risk Minimization
Section 7.1 discussed SRM (7.1) as a method for pruning the hypothesis space H used in
ERM (4.3). This pruning is implemented either via a (hard) constraint as in (7.1) or by
245
adding a regularization term to the training error as in (7.2). The idea of SRM is to avoid
(prune away) hypothesis maps that perform good on the training set but poorly outside, i.e.,
they do not generalize well. Here, we will use another criterion for steering the pruning and
construction of regularization terms. In particular, we use the (intrinsic) explainability of a
hypotheses map as a regularization term.
To make the notion of explainability precise we use again the probabilistic model of
Section 10.1.1. We interpret data points as realizations of iid RVs with common (joint)
probability distribution p(x; y) of features x and label y. A quantitative measure the intrinsic
explainability of a hypothesis h 2 H is the conditional (di erential) entropy [28, Ch. 2 and
8]
H(^yju) := 􀀀E
 
log p(^yju)
 
: (10.22)
The conditional entropy (10.22) indicates the uncertainty about the prediction ^y, given the
user summary ^u = u(x). Smaller values H(^y; u) correspond to smaller levels of uncertainty
in the predictions ^y that is experienced by user u.
We obtain EERM by requiring a su ciently small conditional entropy (10.22) of a hypothesis,
^h
2 argmin
h2H
bL
􀀀
h
 
s.t. H(^yj^u)    : (10.23)
The random variable ^y = h(x) in the constraint of (10.23) is obtained by applying the
predictor map h 2 H to the features. The constraint H(^yj^u)     in (10.23) enforces the
learnt hypothesis ^h to be su ciently explainable in the sense that the conditional entropy
H(^ hj^u)     does not exceed a prescribed level  .
Let us now consider the special case of EERM (10.23) for the linear hypothesis space
h(w)(x) := wTx with some parameter vector w 2 Rn: (10.24)
Moreover, we assume that the features x of a data point and its user summary u are jointly
Gaussian with mean zero and covariance matrix C,
􀀀
xT ; ^u
 T
  N(0;C): (10.25)
Under the assumptions (10.24) and (10.25) (see [28, Ch. 8]),
H(^uj^y) = (1=2) log  2
^yj^u: (10.26)
246
Here, we used the conditional variance  2
^yj^u of ^y given the random user summary u = u(x).
Inserting (10.26) into the generic form of EERM (10.23),
^h
2 argmin
h2H
bL
(h) s.t. log  2
^yj^u    : (10.27)
By the monotonicity of the logarithm, (10.27) is equivalent to
^h
2 argmin
h2H
bL
(h) s.t.  2
^yj^u   e( ): (10.28)
To further develop (10.29), we use the identity
min
 2R
E
 􀀀
^y 􀀀  u
 2 
=  2
^yj^u: (10.29)
The identity (10.29) relates the conditional variance  2
^yj^u to the minimum mean squared error
that can be achieved by estimating ^y using a linear estimator  ^u with some   2 R. Inserting
(10.29) and (10.24) into (10.28),
^h
2 argmin
w2Rn; 2R
bL
(h(w)) s.t. E
  􀀀
wTx | {z }
(10.24)
= ^y
􀀀 ^u
 2 
  e( ): (10.30)
The inequality constraint in (10.30) is convex [15, Ch. 4.2.]. For squared error loss, the
objective function bL
(h(w)) is also convex. Thus, for linear least squares regression, we can
reformulate (10.30) as an equivalent (dual) unconstrained problem [15, Ch. 5]
^h
2 argmin
w2Rn; 2R
E(h(w)) +  E
 􀀀
wTx 􀀀  ^u
 2 
: (10.31)
By convex duality, for a given threshold e( ) in (10.30), we can  nd a value for   in (10.31)
such that (10.30) and (10.31) have the same solutions [15, Ch. 5]. Algorithm 20 below is
obtained from (10.31) by approximating the expectation E
 􀀀
wTx 􀀀  ^u
 2 
with an average
over the data points
􀀀
x(i); ^y(i); ^u(i)
 
for i = 1; : : : ;m.
10.3 Exercises
Exercise 10.1. Convexity of Explainable Linear regression Rewrite the optimization
problem (10.32) as an equivalent quadratic optimization problem minv2Rn vTQv + vTq.
247
Algorithm 20 Explainable Linear Least Squares Regression
Input: explainability parameter  , training set
􀀀
x(i); ^y(i); ^ui)
 
for i = 1; : : : ;m
1: solve
bw
2 argmin
 2R;w2Rn
(1=m)
Xm
i=1
􀀀
^y(i)􀀀wTx(i) 2
| {z }
empirical risk
+  (wTx(i) 􀀀  ^u(i))2
| {z }
explainability
(10.32)
Output: weights bw
of explainable linear hypothesis
Identify the matrix Q 2 Rn n and the vector q 2 Rn.
248
Glossary
k-fold cross-validation (k-fold CV k-fold cross-validation divides a dataset evenly into
k folds. This algorithm consists of k repetitions, during which one of the folds as the
validation set and the remaining k 􀀀 1 folds as a training set. 155{158
k-means The k-means algorithm is a hard clustering method. It aims at assigning data
points to clusters such that they have minimum average distance from the cluster
centre. 13, 196, 198, 201, 202, 205, 212{215, 217, 249, 251
\density-based spatial clustering of applications with noise" (DBSCAN) A clustering
algorithm for data points that are characterized by numeric feature vectors.
Similar to k-means and soft clustering via GMM also DBSCAN uses the Euclidean
distances between feature vectors to determine the clusters. However, in contrast to
these other clustering methodes, DBSCAN uses a di erent notion of similarity between
data points. In particular, DBSCAN considers two data points as similar if they are
\connected" via a sequence (path) of close-by intermediate data points. Thus, DBSCAN
might consider two data points as similar (and therefore belonging to the same
cluster) even if their feature vectors have a large Euclidean distance. 214{216
activation function Each arti cial neuron within an ANN consists of an activation function
that maps the inputs of the neuron to a single output value. In general, an
activation function is a non-linear map of the weighted sum of neuron inputs (this
weighted sum is the activation of the neuron). 9, 97, 98, 104, 105, 264
arti cial intelligence Arti cial intelligence aims at developing systems that behave rational
in the sense of maximizing a long-term reward. 25{28, 30
arti cial neural network An arti cial neural network is a graphical (signal-
ow) representation
of a map from features of a data point at its input to a predicted label at its
249
output. 9, 10, 25, 33, 49, 52, 72, 78, 97{99, 104, 105, 111, 112, 132, 148, 172, 173, 178,
182, 238, 249, 253, 257, 262, 264
bagging bagging (or \bootstrap aggregation") is a generic technique to improve or robustify
a given ML method. The idea is to use the bootstrap to generate perturbed copy of a
given training set and then apply the original ML method to learn a separate hypothesis
for each perturbed copy of the training set. The resulting set of hypotheses is then
used to predict the label of a data point by combining or aggregating the individual
predictions of each individual hypothesis. For hypotheses that deliver numeric label
values (regression methods) this aggregation could be implemented by computing the
average of individual predictions. 182
baseline A reference value or benchmark for the average loss incurred by a hypothesis when
applied to the data points generated in a speci c ML application. Such a reference
value might be obtained from human performance (e.g., error rate of dermatologists
diagnosing cancer from visual inspection of skin areas) or other ML methods (\competitors")
108, 147, 149, 171, 172
Bayes estimator A hypothesis whose Bayes risk is minimal [85]. 65, 66, 68, 92, 101, 106,
108{111, 119, 121, 122, 171, 174, 250
Bayes risk We use the term Bayes risk as a synonym for the risk or expected loss of a
hypothesis. Some authors reserve the term Bayes risk for the risk of a hypothesis that
achieves minimum risk, such a hypothesis being referred to as a Bayes estimator [85].
65, 68, 108, 147, 149, 171, 250, 255
bias Consider some unknown quantity   w, e.g., the true weight in a linear model y =   wx+e
relating feature and label of a data point. We might use an ML method (e.g., based
on ERM) to compute an estimate ^ w for the   w based on a set of data points that are
realizations of RVs. The (squared) bias incurred by the estimate ^ w is typically de ned
as B2 :=
􀀀
Ef ^ wg􀀀   w
 2
. We extend this de nition to vector-valued quantities using the
squared Euclidean norm B2 :=



Efbw
g 􀀀 w

 
2
2. 9, 166{169, 187, 188
bootstrap Consider a probabilistic model that interprets a given set of data points D =  
z(1); : : : ; z(m)
 
as realizations of iid RVs with a common probability distribution p(z).
The bootstrap uses the histogram of D as the underlying proability distribution p(z).
149, 170
250
classi cation Classi cation is the task of determining a discrete-valued label y of a data
point based solely on its features x. The label y belongs to a  nite set, such as
y 2 f􀀀1; 1g, or y 2 f1; : : : ; 19g and represents a category to which the corresponding
data point belongs to. 43, 57, 62, 63, 196
classi er A classi er is a hypothesis h(x) that is used to predict a discrete-valued label.
Strictly speaking, a classi er is a hypothesis h(x) that can take only a  nite number
of di erent values. However, we are sometimes sloppy and use the term classi er
also for a hypothesis that delivers a real number which is thresholded to obtain the
predicted label value. For example, in a binary classi cation problem with label values
y 2 f􀀀1; 1g, we refer to a linear hypothesis h(x) = wTx as classi er if it is used to
predict the label value according to ^y = 1 when h(x)   0 and ^y = 􀀀1 otherwise. 48,
92
cluster A cluster is a subset of data points that are more similar to each other than to the
data points outside the cluster. The notion and measure of similarity between data
points is a design choice. If data points are characterized by numeric Euclidean feature
vectors it might be reasonable to de ne similarity between two data points using the
(inverse of th) Euclidean distance between the corresponding feature vectors 29, 30,
33, 39, 196, 197, 204, 207, 253, 265
clustering Clustering means to decompose a givne set of data points into few subsets, which
are referred to as clusters, that consist of similar data points. Di erent clustering
methods use di erent measurs for the similarity between data points and di erent
representation of cluters. The clustering method k-means uses the average feature
vector (\cluster means") of a cluster as its repsentative (see Section 8.1). A popular
soft clustering method based on GMM represents a cluster by a Gaussian (multivariate
normal) probability distribution (see Section 8.2). 29, 31, 33, 53, 196
condition number The condition number  (Q) of a psd matrix Q is the ratio of the largest
to the smallest eigenvalue of Q. 116, 137, 139, 140, 146, 189
confusion matrix Consider data points characterized by features x and label y having
value c 2 f1; : : : ; kg. The confusion matrix is k   k matrix with rows representing
di erent values c of the true label of a data point. The columns of a confusion matrix
correspond to di erent values c0 delivered by a hypothesis h(x). The (c; c0)-th entry of
251
the confusion matrix is the fraction of data points with label y=c and predicted label
^y=c0. 67
convex A set C in Rn is called convex if it contains the line segment between any two points
of that set. A function is called convex if its epigraph is a convex set [15]. 63, 91, 104,
113
data A set of data points. 32, 34, 92, 102, 111, 169, 263
data augmentation Data augmentation methods add synthetic data points to an existing
set of data points. These synthetic data points might be obtained by perturbations
(adding noise) or transformations (rotations of images) of the original data points. 41,
158, 173, 176, 183, 185, 190
data point A data point is any object that conveys information [28]. Data points might
be students, radio signals, trees, forests, images, RVs, real numbers or proteins. We
characterize data points using two types of properties. One type of property is referred
to as a feature. Features are properties of a data point that can be measured or
computed in an automated fashion. Another type of property is referred to as a label.
The label of a data point represents a higher-level facts or quantities of interest. In
contrast to features, determining the label of a data point typically requires human
experts (domain experts). Roughly speaking, ML aims at predicting the label of a
data point based solely on its features. 2{4, 6{10, 16, 19{25, 27{31, 33{48, 50, 52{77,
79{94, 98{112, 115, 116, 119, 120, 122{127, 130, 134, 135, 137, 139, 140, 142, 143,
145{151, 153{160, 162{179, 181{186, 189{191, 193{198, 201{203, 207{211, 213{226,
228{235, 237{243, 245{247, 249{267
dataset With a slight abuse of notation we use the terms \dataset\ or \set of datapoints"
to refer to an indexed list of data points z(1); : : : ;. Thus, there is a  rst data point z(1),
a second data point z(2) and so on. Strictly speaking a dataset is a list and not a set
[57]. By using indexed lists of data points we avoid some of the challenges arising in
concept of an abstract set. 7, 8, 37, 39, 42, 72, 74, 77, 170, 196, 203, 209, 226, 230, 258
decision region Consider a hypothesis map h that reads in a feature vector x 2 Rn and
delivers a value from a  nite set Y. The decision boundary induced by h is the set of
vectors x 2 Rn that lie between di erentdecision regions. More precisely, a vector x
belongs to the decision boundary if and only if each neighborhood fx0 : kx􀀀x0k   "g,
252
for any " > 0, contains at least two vectors with di erent function values. 52, 53, 90,
91
decision region Consider a hypothesis map h that can only take values from a  nite set
Y. We refer to the set of features x 2 X that result in the same output h(x) = a as a
decision region of the hypothesis h. 48, 94{96, 252
decision tree A decision tree is a 
ow-chart like representation of a hypothesis map h.
More formally, a decision tree is a directed graph which reads in the feature vector x
of a data point at its root node. The root node then forwards the data point to one
of its children nodes based on some elementary test on the features x. If the receiving
children node is not a leaf node, i.e., it has itself children nodes, it represents another
test. Based on the test result, the data point is further pushed to one of its neighbours.
This testing and forwarding of the data point is repeated until the data point ends up
in a leaf node (having no children nodes). The leaf nodes represent sets (decision
regions) constituted by feature vectors x that are mapped to the same function value
h(x). 10, 94{96, 109, 112, 117{119, 172, 182, 238, 263
deep net We refer to an ANN with a (relatively) large number of hidden layers as a deep
ANN or \deep net". Deep nets are used to represent the hypothesis spaces of deep
learning methods [49]. 99, 102, 128, 173
degree of belonging A number that indicats the extend by which a data point belongs to
a cluster. The degree of belonging can be interpreted as a soft cluster assignment. Soft
clustering methods typically represent the degree of belonging by a real number in the
interval [0; 1]. The boundary values 0 and 1 correspond to hard cluster assignments.
196, 197, 207, 209, 213, 265
di erentiable A function f : Rn ! R is di erentiable if it has a gradient rf(x) everywhere
(for every x 2 Rn). 91, 129
e ective dimension The e ective dimension de  (H) of an in nite hypothesis space H is
a measure of its size. Loosely speaking, the e ective dimension is equal to the number
of \independent" tunable parameters of the model. These parameters might be the
coe cients used in a linear map or the weights and bias terms of an ANN. 54, 55, 104,
148, 164, 170, 172, 177, 178, 183, 184, 257
253
eigenvalue We refer to a number   2 R as eigenvalue of a square matrix A 2 Rn n if there
is a non-zero vector x 2 Rn n f0g such that Ax =  x. 133, 134, 137, 139, 222, 223,
229, 230, 235, 254
eigenvalue decomposition The task of computing the eigenvalues and corresponding eigenvectors
of a matrix. 222, 223, 230{235
eigenvector An eigenvector of a matrix A is a non-zero vector x 2 Rn n f0g such that
Ax =  x with some eigenvalue  . 222, 223, 229{231, 233, 235
empirical risk The empirical risk of a given hypothesis on a given set of datapoints is the
average loss of the hypothesis computed over all datapoints in that set. 9, 65, 66, 73,
74, 87, 107, 108, 110, 112, 115, 117, 119, 123, 140, 152, 190, 193, 196, 202, 203, 212,
231, 248, 266
empirical risk minimization Empirical risk minimization is the optimization problem of
 nding the hypothesis with minimum average loss (empirical risk) on a given set of
data points (the training set). Many ML methods are special cases of empirical risk
minimization. 24, 32{35, 92, 100, 107{115, 117, 119, 121{129, 135, 137, 138, 147{149,
151, 153{157, 159, 165, 166, 171{173, 175{185, 195, 196, 201, 202, 212, 224, 238, 245,
250, 254, 255, 261, 264, 266, 267
estimation error Consider data points with feature vectors x and label y. In some applications
we can model the relation between features and label of a data point as
y =  h
(x) + ". Here we used some true hypothesis  h
and a noise term " which might
represent modelling or labelling errors. The estimation error incurred by a ML method
that learns a hypothesis ^h, e.g., using ERM, is de ned as ^h 􀀀  h
. For a parametrized
hypothesis space, consisting of hypothesis maps that are determined by a parameter
vector w, we de ne the estimation error in terms of parameter vectors as  w = bw
􀀀w.
 rst 165, 166, 261
Euclidean space The Euclidean space Rn of dimension n refers to the space of all vectors
x =
􀀀
x1; : : : ; xn
 
, with real-valued entries x1; : : : ; xn 2 R, whose geometry is de ned
by the inner product xTx0 =
Pn
j=1 xjx0
j between any two vectors x; x0 2 Rn [119]. 19,
20, 40, 42, 85, 93, 100, 196, 225, 256, 263
expectation maximization Expectation maximization is a generic technique for estimating
the parameters of a probabilistic model (a parametrized probability distribution)
254
p(z;w) from data [13, 58, 152]. Expectation maximization delivers an approximation
to the maximum likelihood estimate for the model parameters w. 210, 227, 263
expert ML aims at learning a hypothesis h that accurately predicts the label of a data
point based on its features. We measure the prediction error using some loss function.
Ideally we want to  nd a hypothesis that incurres minimum loss. One approach to
make this goal precise is to use the i.i.d. assumption and use the resulting Bayes risk
as the benchmark level for the (average) loss of a hypothesis. Alternatively we might
know a reference or benchmark hypothesis h0 which might be obtained by some existing
ML mehtod. We can then compare the loss incurred by h with the loss incurred by
h0. Such a reference or baseline hypothesis h0 is refered to as an expert. Note that
an expert might deliver very poor predictions. We typically compare against many
di erent experts and aim at incurring not much more loss than the best among those
experts (this is known as regret minimization) [21, 60].  rst 68, 255, 264
explainability We use the term explainability in a highly informal fashion as a measure for
the predicatability of a ML method (output). A ML method is perfectly explained to
a user if she can, upon receiving this explanation, perfectly anticipate the behaviour
of the ML method. 33, 237{239, 246, 248
explainable empirical risk minimization An instance of structural risk minimization
that adds a regularization term to the training error in ERM. The regularization term
is chosen to favour hypotheses that are intrinsically explainable for a user. 238, 239,
246, 247
explainable machine learning Explainable ML methods aim at complementing predictions
with explanations for how the prediction has been obtained. 3, 22, 33, 237, 238,
240{242
feature map A map that transforms some raw features into a new feature vector. The new
feature vector might be preferable over the raw features for several reasons. It might
be possible to use linear hypothesis with the new feature vectors. Another reason could
be that the new feature vector is much shorter and therefore avoids over tting or can
be used for a scatterplot 81, 84, 151, 235
feature space The feature space of a given ML application or method is constituted by all
potential values that the feature vector of a data point can take on. Within this book
255
the most frequently used choice for the feature space is the Euclidean space Rn with
dimension n being the number of individual features of a data point. 40{42, 54, 100,
258
features Features are those properties of a data point that can be measured or computed in
an automated fashion. For example, if a data point is a bitmap image, then we could
use the red-green-blue intensities of its pixels as features. Some widely used synonyms
for the term feature are \covariate",\explanatory variable", \independent variable",
\input (variable)", \predictor (variable)" or \regressor" [53, 31, 39]. However, this
book makes consequent use of the term features for low-level properties of data points
that can be measured easily. 7, 34, 36, 37, 42, 59, 79, 86, 91, 93, 126, 252, 260
federated learning (FL) Federated learning is an umbrella term for ML methods that
train models in a collaborative fashion using decentralized data and computation. 24,
111
Finnish Meteorological Institute The Finnish Meteorological Institute is a government
agency responsible for gathering and reporting weather data in Finland. 15, 16, 24,
72, 123
Gaussian mixture model Gaussian mixture models (GMM) are a family of probabilistic
models for data points. Within a GMM, the feature vector x of a data point is interpreted
as being drawn from one out of k di erent multivariate normal (Gaussian)
distributions indexed by c = 1; : : : ; k. The probability that the feature vector x is
drawn from the c-th Gaussian distribution is denoted pc. The GMM is parametrized
by the probability pc of x being drawn from the c-th Gaussian distribution as well as
the mean vectors  (c) and covariance matrices  (c) for c = 1; : : : ; k. 120, 208{215, 249,
251
gradient For a real-valued function f : Rn ! R : w 7! f(w), a vector a such that
limw!w0
f(w)􀀀
􀀀
f(w0)+aT (w􀀀w0)
 
kw􀀀w0k = 0 is referred to as the gradient of f at w0. If such a
vector exists it is denoted rf(w0) or rf(w)
  
w0 . 10, 32, 52, 63, 128, 129, 138, 141,
143, 144, 253, 265
gradient descent Gradient descent is an iterative method for  nding the minimum of a
di erentiable function f(w). 63, 116, 117, 128, 130{146, 172, 173, 181, 189{191, 259,
266
256
gradient-based method Gradient-based methods are iterative algorithms for  nding the
minimum (or maximum) of a di erentiable objective function of a parameter vector.
These algorithms construct a sequence of approximations to an optimal parameter vector
whose function value is minimal (or maximal). As their name indicates, gradientbased
methods use the gradients of the objective function evaluated during previous
iterations to construct a new (hopefully) improved approximation of an optimal parameter
vector. 3, 8, 32, 57, 58, 62, 63, 91, 109, 112, 113, 119, 128, 129, 131, 132, 143,
144, 172, 173, 265
hard clustering Hard clustering refers to the task of partitioning a given set of data points
into (few) non-overlapping clusters. Each data point is assigned to one speci c cluster.
196{198, 207, 212, 217
high-dimensional regime A ML method or problem belongs to the high-dimensional
regime if the e ective dimension of the model is larger than the number of available
(labeled) data points. For example, linear regression belongs to the high-dimensional
regime whenever the number n of features used to characterize datapoints is larger
than the number of data points in the training set. Another example for the highdimensional
regime are deep learning methods that use a hypothesis space generated
by a ANN with much more tunable weights than the number of data points in the
training set. The recent  eld of high-dimensional statistics uses probability theory to
analyze ML methods in the high-dimensional regime [150, 18]. 83, 84, 148, 165
Hilbert space A Hilbert space is a linear vector space that is equipped with an inner
product between pairs of vectors. One important example for a Hilbert space is the
Euclidean spaces Rn, for some dimension n, which consists of Euclidean vectors u = 􀀀
u1; : : : ; un
 T
along with the inner product uTv. 235
hinge loss Consider a data point that is characterized by a feature vector x 2 Rn and a
binary label y 2 f􀀀1; 1g. The hinge loss incurred by a speci c hypothesis h is de ned
as (2.11). A regularized variant of the hinge loss is used by the SVM to learn a linear
classi er with maximum margin between the two classes (see Figure 3.6). 62, 63, 89{91,
161, 266
histogram Consider a dataset D consisting of data points z(1); : : : ; z(m) that belong to some
box in Rn. We partition this hyper-rectangle evenly into small elementary boxes. The
257
histogram of D is the assignment of each elementary box to the corresponding fractions
of datapoints in D that belong to this elementary box. 170, 171
Huber loss The Huber loss is a mixture of the squared error loss and the absolute value of
the prediction error. 82, 83
hypothesis A map (or function) h : X ! Y from the feature space X to the label space
Y. Given a data point with features x we use a hypothesis map h to estimate (or
approximate) the label y using the predicted label ^y = h(x). ML is about learning (or
 nding) a hypothesis map h such that y   h(x) for any data point. 30, 260, 262
hypothesis space Every practical ML method uses a speci c hypothesis space (or model)
H. The hypothesis space of a ML method is a subset of all possible maps from the
feature space to label space. The design choice of the hypothesis space should take into
account available computational resources and statistical aspects. If the computational
infrastructure allows for e cient matrix operations and we expect a linear relation
between feature values and label, a resonable  rst candidate for the hypothesis space
is the space of linear maps (2.4). 2, 3, 11, 16, 17, 19, 30, 34, 35, 48{57, 64, 69, 73{77,
79, 81, 85, 86, 89, 90, 92, 93, 95{103, 105, 107, 108, 110, 112, 113, 117, 124, 128, 132,
147{149, 151, 155, 158, 159, 164, 168, 170{172, 175, 176, 178{181, 183{186, 192, 193,
221, 238, 239, 241, 245, 246, 253, 261, 266, 267
i.i.d. It can be useful to interpret data points z(1); : : : ; z(m) as realizatons of independent
and identically distributed RVs with a common probability distribution. If these RVs
are continous, their joint pdf is p
􀀀
z(1); : : : ; z(m)
 
=
Qm
i=1 p
􀀀
z(i)
 
with p(z) being the
common marginal pdf of the underlying RVs. 24, 25, 46, 47, 55, 62, 65, 66, 88, 92, 99,
100, 108{110, 119, 120, 126, 137, 141, 145, 150, 153, 155, 163{166, 168{171, 173, 174,
183, 184, 187, 201, 208{210, 212, 227, 234, 241, 246, 250, 258, 259, 261, 262, 264, 265
i.i.d. assumption The i.i.d. assumption interprets data points of a dataset as the realizations
of iid RVs. 46, 47, 55, 108{111, 125, 153, 155, 163, 171, 173, 255, 262, 264
label A higher level fact or quantity of interest associated with a data point. If a data point
is an image, its label might be the fact that it shows a cat (or not). Some widely used
synonyms for the term label are "response variable", "output variable" or "target"
[53, 31, 39]. 34, 36, 42, 44, 50, 56, 59, 79, 91, 126, 252, 260
258
label space Consider a ML application that involves data points characterized by features
and labels. The label space of a given ML application or method is constituted by
all potential values that the label of a data point can take on. A popular choice for
the label space in regression problems (or methods) is Y = R. Binary classi cation
problems (or methods) use a label space that consists of two di erent elements, e.g.,
Y = f􀀀1; 1g, Y = f0; 1g or Y = f\cat image"; "no cat image"g 34, 42{44, 48, 52{54,
60, 61, 73, 80, 86, 89, 94, 100, 117, 231, 258
Laplacian matrix The geometry or structure of a similarity graph G can be analyzed using
the properties of special matrices that are associated with G. One such matrix is the
graph Laplacian matrix L whose entries are de ned in (9.13). 228{230
law of large numbers The law of large numbers refers to the convergence of the average
of an increasing number of iid RVs to the mean (or expectation) of their common
probability distribution. 62, 66, 108, 110, 173, 184, 185
learning rate Consider an iterative method for  nding or learning a good choice for a
hypothesis. Such an iterative method repeats similar computational (update) steps
that adjust or modify the current choice for the hypothesis to obtain an improved
hypothesis. A prime example for such an iterative learning method is GD and its
variants (see 5). We refer by learning rate to any parameter of an iterative learning
method that controls the extent by which the current hypothesis might be modi ed
or improved in each iteration. A prime example for such a parameter is the step size
used in GD. Within this book we use the term learning rate mostly as a synonym for
the step size of (a variant of) GD 8, 131{137, 139, 141{143, 145, 146, 173, 190, 266
least absolute deviation regression Least absolute deviation regression uses the average
of the absolute precondition errors to  nd a linear hypothesis. 112
least absolute shrinkage and selection operator (Lasso) The least absolute shrinkage
and selection operator (Lasso) is an instance of SRM for learning the weights
w of a linear map h(x) = wTx. The Lasso minimizes the sum consisting of an average
squared error loss (as in linear regression) and the scaled `1 norm of the weight vector
w. 84, 245
linear classi er A classi er h(x) maps the feature vector x 2 Rn of a datapoint to a
predicted label ^y 2 Y out of a  nite set of label values Y. We can characterize such a
259
classi er equivalently by the decision regions Ra, for every possible label value a 2 Y.
Linear classi ers are such that the boundaries between the regions Ra are hyperplanes
in Rn. 20, 52, 53, 90, 92, 94, 97, 137, 257
linear regression Linear regression aims at learning a linear hypothesis map to predict
a numeric label based on numeric features of a data point. The quality of a linear
hypothesis map is typically measured using the average squared error loss incurred on
a set of labeled data points (the training set). 2, 14, 32, 35, 44, 55, 76, 79, 81, 83{86,
89, 93, 96, 101, 102, 105, 111{114, 116, 117, 123, 124, 126{129, 131, 134, 136, 138, 148,
149, 151, 163, 164, 168, 173, 174, 177, 178, 181, 187{190, 213, 216, 219, 224{226, 235,
237, 240, 243, 244, 247, 259
logistic loss Consider a data point that is characterized by the features x and a binary
label y 2 f􀀀1; 1g. We use a hypothesis h to predict the label y solely from the features
x. The logistic loss incurred by a speci c hypothesis h is de ned as (2.12). 57, 58, 62,
63, 77, 87, 91, 104, 137, 159, 261
logistic regression Logistic regression aims at learning a linear hypothesis map to predict
a binary label based on numeric features of a data point. The quality of a linear
hypothesis map (classi er) is measured using its average logistic loss on some labeled
datapoints (the training set). 44, 52, 63, 86{93, 96, 97, 101, 102, 104, 105, 117, 121,
122, 128, 129, 131, 134, 137{139, 145, 146, 159
loss With a slight abuse of language, we use the term loss either for loss function itself or
for its value for a speci c pair of data point and hypothesis. 2, 9, 21, 31, 32, 34, 57,
58, 62, 63, 76, 77, 79, 92, 108, 112, 125, 129, 147{149, 153, 158, 163, 173, 250, 264
loss function A loss function is a map
L : X   Y   H ! R+ :
􀀀􀀀
x; y
 
; h
 
7! L
􀀀
(x; y); h
 
which assigns a pair consisting of a datapoint, with features x and label y, and a hypothesis
h 2 H the non-negative real number L
􀀀
(x; y); h
 
. The loss value L
􀀀
(x; y); h
 
quanti
 es the discrepancy between the true label y and the predicted label h(x). Smaller
(closer to zero) values L
􀀀
(x; y); h
 
mean a smaller discrepancy between predicted label
and true label of a data point. Figure 2.11 depicts a loss function for a given data
point, with features x and label y, as a function of the hypothesis h 2 H. 2, 3, 17{19,
260
24, 29{32, 34, 35, 44, 56{64, 67, 76, 78, 79, 82, 89{92, 99, 102, 108, 110, 113, 124, 125,
129, 130, 147, 148, 158{161, 171, 177, 192, 226, 255, 260, 261, 264
maximum Given a set of real numbers, the maximum is the largest of those numbers. 72
maximum likelihood Consider data points that are interpreted as iid realizations of RVs
with a common (but unknown) probability distribution. Maximum likelihood methods
 nd a parameter vector w for a probabilistic model p(z;w) such that the probability
(density) of observing the actucal data is maximized. Loosely speaking, we try out all
possible parameter vectors w and determine the resulting probability of observing the
given datapoints if they would be iid with common probability distribution p(z;w).
The maximum likelihood estimator is the parameter vector that results in the highest
probability (density). 47, 57, 88, 99, 100, 120, 210, 255
mean The expectation of a real-valued random variable. 46
mean squared estimation error Consider a ML method that uses a parametrized hypothesis
space. For a given training set, whose data points are interpreted as realizations
of RVs, the ML method learns the parameters incurring the estimation error
 w. The mean squared estimation error is de ned as the expectation E
 


 w

 
2 
of
the squared Euclidean norm of the estimation error. 166, 168
metric A metric refers to a loss function that is used solely for the  nal performance evaluation
of a learnt hypothesis. The metric is typically a loss function that has a \natural"
interpretation (such as the 0=1 loss (2.9)) but is not a good choice to guide the learning
process, e.g., via ERM. For ERM, we typically prefer loss functions that depend
smoothly on the (parameters of the) hypothesis. Examples for such smooth loss functions
include the squared error loss (2.8) and the logistic loss (2.12). 58
minimum Given a set of real numbers, the minimum is the smallest of those numbers. 72
missing data By missing data, we refer to a situation where some feature values of a subset
of data points are unknown. Data imputation techniaues aim at estimating (predicting)
these missing feature values [1]. 37
model We use the term model as a synonym for hypothesis space 2, 17, 32, 34, 44, 48, 78,
79, 147{149, 158, 172, 257, 258, 262, 267
261
multi-label classi cation Multi-label classi cation problems and methods involve data
points that are characterized by several individual labels. 42, 44, 73, 177
nearest neighbour Nearest neighbour methods learn a hypothesis h : X ! Y whose
function value h(x) is solely detemined by the nearest neighbours in the feature space
X 100{102, 105, 106
non i.i.d. data A dataset that cannot be well modelled as realizations of iid RVs. 262
non-i.i.d. See non-i.i.d. data. 111
non-smooth We refer to a function as non-smooth if it is not smooth [102]. 57, 113
outlier Many ML methods are motivated by the i.i.d. assumption which interprets data
points as realizations of iid RVs with a common probability distribution. The i.i.d.
assumption is useful for applications where the statistical properties of the data generation
process are stationary (time-invariant). However, in some applications the data
consists of a majority of \regular" data points that conform with an i.i.d. assumption
and a small number of data points that have fundamentally di erent statistical properties
compared to the regular data points. We refer to a data point that substantially
deviates from the statistical properties of the majority of data points as an outlier.
Di erent methods for outlier detection use di erent measures for this deviation. 57,
59, 82, 83, 156, 225
parameters The parameters of a ML model are tunable (learnable or adjustable) quantities
that allow to choose between di erent hypothesis maps. For example, the linear model
H := fh : h(x) = w1x + w2g consists of all hypothesis maps h(x) = w1x + w2 with
a particular choice for the parameters w1;w2. Another example of parameters are the
weights assigned to the connections of an ANN. 16, 148
positive semi-de nite A symmetric matrix Q = QT 2 Rn n is referred to as positive
semi-de nite if xTQx   0 for every vector x 2 Rn. 5, 9, 100, 133, 189, 222, 229, 235,
251
predictor A predictor is a hypothesis whose function values are numeric, such as real numbers.
Given a data point with features x, the predictor value h(x) 2 R is used as
a prediction (estimate/guess/approximation) for the true numeric label y 2 R of the
data point. 86, 93
262
principal component analysis (PCA) Principal component analysis determines a given
number of new features that are obtained by a linear transformation (map) of the raw
features. 14, 20, 223{231, 234, 235, 263
probabilistic PCA Probabilistic PCA extends basic PCA by using a probabilistic model
for data points. Within this probabilistic model, the task of dimensionality reduction
becomes an estimation problem that can be solved using EM methods. 227
probability density function (pdf) The probability density function (pdf) p(x) of a realvalued
RV x 2 R is a particular representation of its probability distribution. If the
pdf exists, it can be used to compute the probability that x takes on a value from a
(measureable) set B   R via p(x 2 B) =
R
B p(x0)dx0 [10, Ch. 3]. The pdf of a vectorvalued
RV x 2 Rn (if it exists) allows to compute the probability that x falls into a
(measurable) region R via p(x 2 R) =
R
R p(x0)dx0
1 : : : dx0
n [10, Ch. 3]. 120, 258
probability distribution The data generated in some ML applications can be reasonably
well modeled as realizations of a RV. The overall statistical properties (or intrinisic
structure) of such data are then governed by the probability distribution of this RV.
We use the term probability distribution in a highly informal manner and mean the
collection of probabilities assigned to di erent values or value ranges of a RV. The
probability distribution of a binary RV y 2 f0; 1g is fully speci ed by the probabilities
p(y = 0) and p(y = 1)
􀀀
= 1􀀀p(y = 0)
 
. The probability distribution of a realvalued
RV x 2 R might be speci ed by a probability density function p(x) such that
p(x 2 [a; b])   p(a)jb􀀀aj. In the most general case, a probability distribution is de ned
by a probability measure [51, 12]. 21, 24, 25, 46, 55, 57, 62, 65, 66, 92, 107{110, 119,
120, 122, 125, 126, 141, 145, 149, 150, 153, 155, 164, 170, 171, 173, 174, 250, 251, 254,
258, 259, 261{264
random forest A random forest is a set (ensemble) of di erent decision trees. Each of
these decision trees is obtained by  tting a perturbed copy of the original dataset. 182
random variable A random variable is a mapping for function from a set of elementary
events to a set of values. The set of elementary events is equipped with a probability
measure that assigns subsets of elemtary events a probability. A binary random variable
maps elementary events to a set containing two di erent value, such as f􀀀1; 1g or
fcat; no catg. A real-valued random variable maps elementary events to real numbers
R. A vector-valued random variable maps elementary events to the Euclidean space
263
Rn. Probability theory uses the concept of measurable spaces to rigorously de ne and
study the properties of (large) collections of random variables [51, 12]. 6, 9, 24, 25, 33,
46, 47, 55, 62, 65, 66, 84, 87, 92, 99, 108{110, 119, 120, 126, 141, 145, 150, 151, 153,
155, 163{171, 173, 174, 176, 182, 187, 208{210, 227, 234, 241, 242, 246, 250, 252, 258,
259, 261{265, 267
regret The regret of a hypothesis h relative to another hypothesis h0, which serves as a
reference of baseline, is the di erence between the loss incurred by h and the loss
incurred by h0 [21]. The baseline hypothesis h0 is also refered to as an expert.  rst 68
regularization Regularization techniques modify the ERM principle such that the learnt
hypothesis performs well also outside the training set which is used in ERM. One
speci c approach to regularization is by adding a penalty or regularization term to
the objective function of ERM (which is the average loss on the training set). This
regulazation term can be interpreted as an estimate for the increase in the expected
loss (risk) compared to the average loss on the training set. 33, 84, 90, 172, 176, 177,
187, 246
ReLU The recti ed linear unit or \ReLU" is a popular choice for the activation function of
a neuron within an ANN. It is de ned as g(z) = maxf0; zg with z being the weighted
input of the neuron. 98, 104, 105
ridge regression Ridge regression aims at learning the weights w of linear hypothesis map
h(w)(x) = wTx. The quality of a particular choice for the weights w is measured
by the sum of two terms (see (7.4)). The  rst term is the average squared error loss
incurred by h(w) on a set of labeled data points (the training set). The second term
is the scaled squared Euclidean norm  kwk22
with a regularization parameter   > 0.
180, 181, 185{189, 194
risk Consider a hypothesis h that is used to predict the label y of a data point based on
its features x. We measure the quality of a particular prediction using a loss function
L
􀀀
(x; y); h
 
. If we interpret data points as realizations of iid RVs, also the L
􀀀
(x; y); h
 
becomes the realization of a RV. Using such an i.i.d. assumption allows to de ne the
risk of a hypothesis as the expected loss E
 
L
􀀀
(x; y); h
  
. Note that the risk of h
depends on both, the speci c choice for the loss function and the common probability
distribution of the data points. 109, 153, 169, 171, 250
264
sample size The number of individual data points contained in a dataset that is obtained
from realizations of iid RVs. 36, 37, 55, 113
scatterplot A visualization technique that depicts data points by markers in a two-dimensional
plane. 19, 24, 38, 45, 46, 72, 80, 219, 225, 226, 255
semi-supervised learning Semi-supervised learning methods use (large amounts of) unlabeled
data points to support the learning of a hypothesis from (a small number of)
labeled data points [22]. 177, 190
similarity graph Some applications generate data points that are related by a domainspeci
 c notion of similarity. These similarities can be represented conveniently using
a similarity graph G =
􀀀
V := f1; : : : ;mg; E
 
. The node i 2 V represents the i-th data
point. Two nodes are connected by an undirected edge if the corresponding data points
are similar. 228{230, 259
smooth We refer to a real-valued function as smooth if it is di erentiable and its gradient
is continuous [102, 17]. In particular, a di erentiable function f(w) is  -smooth if
the gradient rf(w) is Lipschitz continuous with Lipschitz constant  , i.e., krf(w) 􀀀
rf(w0)k    kw 􀀀 w0k. 104, 112, 113, 130, 133, 262
soft clustering Soft clustering refers to the task of partitioning a given set of data points
into (few) overlapping clusters. Each data point is assigned to several di erent clusters
with varying degree of belonging. Soft clustering amounts to determining such a degree
of belonging (or soft cluster assignment) for each data point and each cluster. 120, 196,
197, 208, 209, 213, 249
spectogram The spectogram of a time signal, e.g., an audio recording, characterizes the
time-frequency distribution of the signal. Loosely speaking, the spectogram quantities
the signal strength at a specitic time and frequence. 38, 39
step size Many ML methods use iterative optimization methods (such as gradient-based
methods) to construct a sequence of increasinbly accurate hypothesis maps h(1); h(2); : : :.
The rth iteration of such an algorithm starts from the current hypothesis h(r) and tries
to modify it to obtain an improved hypothesis h(r+1). Iterative algorithms often use
a step size (hyper-) parameter. The step size controls the amount by which a single
iteration can change or modify the current hypothesis. Since the overall goal of such
265
iteration ML methods is to learn a (approximately) optimal hypothesis we refer to a
step size parameter also as a learning rate. 128, 130, 131
stochastic gradient descent Stochastic gradient descent is obtained from GD by replacing
the gradient of the objective function by a noisy (or stochastic) estimate. 7, 141{
143, 145, 170
structural risk minimization Structural risk minimization is the problem of  nding the
hypothesis that optimally balances the average loss (empirical risk) on a training set
with a regularization term. The regularization term penalizes a hypothesis that is not
robust against (small) perturbations of the data points in the training set. 9, 176, 177,
179{182, 186, 238, 245, 246, 259
subgradient For a real-valued function f : Rn ! R : w 7! f(w), a vector a such that
f(w)   f(w0) +
􀀀
w 􀀀 w0
 T
a is referred to as a subgradient of f at w0. 63, 129
subgradient descent Subgradient descent is a generalization of GD that is obtained by
using sub-gradients (instead of gradients) to construct local approximations of an objective
function such as the empirical risk bL
􀀀
h(w)
  
D
 
as a function of the parameters
w of a hypothesis h(w). 63
support vector machine A binary classi cation method for learning a linear map that
maximally seperates data points the two classes in the feature space (\maximum margin").
Maximizing this separation is equivalent to minimizing a regularized variant of
the hinge loss (2.11). 32, 52, 89{92, 96, 97, 112, 121, 122, 129, 159, 161, 257
test set A set of data points that have neither been used in a training set to learn parameters
of a model nor in a validation set to choose between di erent models (by comparing
validation errors). 160
training error Consider a ML method that aims at learning a hypothesis h 2 H out of a
hypothesis space. We refer to the average loss or empirical risk of a hypothesis h 2 H
on a dataset as training error if it is used to choose between di erent hypotheses.
The principle of ERM is  nd the hypothesis h  2 H with smallest training error.
Overloading the notation a bit, we might refer by training error also to the minimum
empirical risk achieved by the optimal hypothesis h  2 H. 9, 30, 33, 70, 74, 107, 110,
111, 127, 147{151, 154, 156, 157, 160{162, 165, 168, 171{176, 179{181, 213, 246
266
training set A set of data points that is used in ERM to train a hypothesis ^h. The average
loss of ^h on the training set is referred to as the training error. The comparison between
training and validation error informs adaptations of the ML method (such as using a
di erent hypothesis space). 8, 9, 21, 23, 24, 29, 30, 33, 50, 59, 61, 66, 72{74, 76, 77,
83, 84, 87, 91, 92, 100, 101, 104{111, 113, 115{117, 119, 122{127, 134, 135, 137, 142,
143, 146{157, 159{166, 168, 169, 172{176, 178, 179, 181{183, 185, 187, 193, 194, 230,
237, 238, 241, 243, 245, 246, 248, 249, 254, 257, 260, 261, 264, 266, 267
validation error Consider a hypothesis ^h which is obtained by ERM on a training set. The
average loss of ^h on a validation set, which is di erent from the training set, is referred
to as the validation error. 9, 147{149, 152{156, 158{161, 171{174, 176
validation set A set of data points that has not been used as training set in ERM to
train a hypothesis ^h. The average loss of ^h on the validation set is referred to as
the validation error and used to diagnose the ML method. The comparison between
training and validation error informs adaptations of the ML method (such as using
a di erent hypothesis space). 9, 108, 148, 152{156, 159{161, 171{174, 176, 193, 194,
249, 266, 267
Vapnik{Chervonenkis (VC) dimension The VC dimension of an in nite hypothesis
space is a widely-used measure for its size. We refer to [126] for a precise de nition
of VC dimension as well as a discussion of its basic properties and use in ML.
54
variance The variance of a real-valued RV x is de ned as the expectation E
 􀀀
x􀀀Efxg
 2 
of the squared di erence x and its expectation Efxg. We extend this de nition to
vector-valued RVs x as E
 


x 􀀀 Efxg

 
2
2
 
. 9, 46, 166{169, 187, 188
weights We use the term weights synonymously for a  nite set of parameters within a
model. For example, the linear model consists of all linear maps h(x) = wTx that
read in a feature vector x =
􀀀
x1; : : : ; xn
 T
of a data point. Each speci c linear map
is characterized by speci c choices for the parameters for weights w =
􀀀
w1; : : : ;wn
 T
.
16, 17, 123, 170, 187, 193, 194, 213, 214, 231, 248
267
Bibliography
[1] K. Abayomi, A. Gelman, and M. A. Levy. Diagnostics for multivariate imputations.
Journal of The Royal Statistical Society Series C-applied Statistics, 57:273{291, 2008.
[2] M. Abramowitz and I. A. Stegun, editors. Handbook of Mathematical Functions. Dover,
New York, 1965.
[3] C. Andrieu, N. de Freitas, A. Doucet, and M. I. Jordan. An introduction to MCMC
for machine learning. Machine Learning, 50(1-2):5 { 43, 2003.
[4] D. Arthur and S. Vassilvitskii. k-means++: the advantages of careful seeding". In
Proc. of the eighteenth annual ACM-SIAM symposium on Discrete algorithms. Society
for Industrial and Applied Mathematics Philadelphia, 2007.
[5] P. Austin, P. Kaski, and K. Kubjas. Tensor network complexity of multilinear maps.
arXiv, 2018.
[6] R. Baeza-Yates and B. Ribeiro-Neto. Modern Information Retrieval. addwes, 1999.
[7] F. Barata, K. Kipfer, M. Weber, P. Tinschert, E. Fleisch, and T. Kowatsch. Towards
device-agnostic mobile cough detection with convolutional neural networks. In 2019
IEEE International Conference on Healthcare Informatics (ICHI), pages 1{11, 2019.
[8] M. S. Bartlett. An inverse matrix adjustment arising in discriminant analysis. The
Annals of Mathematical Statistics, 22(1):107 { 111, 1951.
[9] M. Belkin, D. Hsu, S. Ma, and S. Mandal. Reconciling modern machine-learning
practice and the classical bias{variance trade-o . Proceedings of the National Academy
of Sciences, 116(32):15849{15854, 2019.
[10] D. Bertsekas and J. Tsitsiklis. Introduction to Probability. Athena Scienti c, 2 edition,
2008.
268
[11] D. P. Bertsekas. Nonlinear Programming. Athena Scienti c, Belmont, MA, 2nd edition,
June 1999.
[12] P. Billingsley. Probability and Measure. Wiley, New York, 3 edition, 1995.
[13] C. M. Bishop. Pattern Recognition and Machine Learning. Springer, 2006.
[14] B. Boashash, editor. Time Frequency Signal Analysis and Processing: A Comprehen-
sive Reference. Elsevier, Amsterdam, The Netherlands, 2003.
[15] S. Boyd and L. Vandenberghe. Convex Optimization. Cambridge Univ. Press, Cambridge,
UK, 2004.
[16] P. J. Brockwell and R. A. Davis. Time Series: Theory and Methods. Springer New
York, 1991.
[17] S. Bubeck. Convex optimization. algorithms and complexity. In Foundations and
Trends in Machine Learning, volume 8. Now Publishers, 2015.
[18] P. Buhlmann and S. van de Geer. Statistics for High-Dimensional Data. Springer,
New York, 2011.
[19] S. Carrazza. Machine learning challenges in theoretical HEP. arXiv, 2018.
[20] R. Caruana. Multitask learning. Machine Learning, 28(1):41{75, 1997.
[21] N. Cesa-Bianchi and G. Lugosi. Prediction, Learning, and Games. Cambridge University
Press, New York, NY, USA, 2006.
[22] O. Chapelle, B. Scholkopf, and A. Zien, editors. Semi-Supervised Learning. The MIT
Press, Cambridge, Massachusetts, 2006.
[23] J. Chen, L. Song, M.Wainwright, and M. Jordan. Learning to explain: An informationtheoretic
perspective on model interpretation. In Proc. 35th Int. Conf. on Mach.
Learning, Stockholm, Sweden, 2018.
[24] H.-F. Cheng, R.Wang, Z. Zhang, F. O'Connell, T. Gray, F. M. Harper, and H. Zhu. Explaining
decision-making algorithms through UI: Strategies to help non-expert stakeholders.
In Proceedings of the 2019 CHI Conference on Human Factors in Computing
Systems, CHI '19, pages 1{12, New York, NY, USA, 2019. Association for Computing
Machinery.
269
[25] I. Cohen and B. Berdugo. Noise estimation by minima controlled recursive averaging
for robust speech enhancement. IEEE Sig. Proc. Lett., 9(1):12{15, Jan. 2002.
[26] D. Cohn, Z. Ghahramani, and M. Jordan. Active learning with statistical models. J.
Artif. Int. Res., 4(1):129{145, March 1996.
[27] T. Cover and P. Hart. Nearest neighbor pattern classi cation. IEEE Transactions on
Information Theory, 13(1):21{27, 1967.
[28] T. M. Cover and J. A. Thomas. Elements of Information Theory. Wiley, New Jersey,
2 edition, 2006.
[29] G. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics
of control, signals and systems 2, (4):303{314, 1989.
[30] T. K. DeBacker and H. M. Crowson. In
uences on cognitive engagement: Epistemological
beliefs and need for closure. British Journal of Educational Psychology,
76(3):535{551, 2006.
[31] Y. Dodge. The Oxford Dictionary of Statistical Terms. Oxford University Press, 2003.
[32] O. Durr, Y. Pauchard, D. Browarnik, R. Axthelm, and M. Loeser. Deep learning on a
raspberry pi for real time face recognition. 01 2015.
[33] B. Efron and R. Tibshirani. Improvements on cross-validation: The 632+ bootstrap
method. Journal of the American Statistical Association, 92(438):548{560, 1997.
[34] R. Eldan and O. Shamir. The power of depth for feedforward neural networks. CoRR,
abs/1512.03965, 2015.
[35] Y. C. Eldar, P. Kuppinger, and H. Bolcskei. Block-sparse signals: Uncertainty relations
and e cient recovery. IEEE Trans. Signal Processing, 58(6):3042{3054, June 2010.
[36] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu. A density-based algorithm for discovering
clusters a density-based algorithm for discovering clusters in large spatial databases
with noise. In Proceedings of the Second International Conference on Knowledge Dis-
covery and Data Mining, pages 226{231, Portland, Oregon, 1996.
[37] A. Esteva, B. Kuprel, R. A. Novoa, J. Ko, S. M. Swetter, H. M. Blau, and S. Thrun.
Dermatologist-level classi cation of skin cancer with deep neural networks. Nature,
542, 2017.
270
[38] J. W. et.al. Association between surgical skin markings in dermoscopic images and
diagnostic performance of a deep learning convolutional neural network for melanoma
recognition. JAMA Dermatol., 155(10):1135{1141, Oct. 2019.
[39] B. Everitt. Cambridge Dictionary of Statistics. Cambridge University Press, 2002.
[40] S. Foucart and H. Rauhut. A Mathematical Introduction to Compressive Sensing.
Springer, New York, 2012.
[41] M. Friendly. A brief history of data visualization. In C. Chen, W. Hardle, and A. Unwin,
editors, Handbook of Computational Statistics: Data Visualization, volume III.
Springer-Verlag, 2006.
[42] R. G. Gallager. Stochastic Processes: Theory for Applications. Cambridge University
Press, 2013.
[43] A. E. Gamal and Y.-H. Kim. Network Information Theory. Cambridge Univ. Press,
2012.
[44] M. Gao, H. Igata, A. Takeuchi, K. Sato, and Y. Ikegaya. Machine learning-based
prediction of adverse drug e ects: An example of seizure-inducing compounds. Journal
of Pharmacological Sciences, 133(2):70 { 78, 2017.
[45] W. Gautschi and G. Inglese. Lower bounds for the condition number of vandermonde
matrices. Numer. Math., 52:241 { 250, 1988.
[46] G. Golub and C. van Loan. An analysis of the total least squares problem. SIAM J.
Numerical Analysis, 17(6):883{893, Dec. 1980.
[47] G. H. Golub and C. F. Van Loan. Matrix Computations. Johns Hopkins University
Press, Baltimore, MD, 3rd edition, 1996.
[48] C. Gomez-Uribe and N. Hunt. The net
ix recommender system: Algorithms, business
value, and innovation. Association for Computing Machinery, 6(4), January 2016.
[49] I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. MIT Press, 2016.
[50] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair,
A. Courville, and Y. Bengio. Generative adversarial nets. In Proc. Neural Inf. Proc.
Syst. (NIPS), 2014.
271
[51] R. Gray. Probability, Random Processes, and Ergodic Properties. Springer, New York,
2 edition, 2009.
[52] R. Gray, J. Kie er, and Y. Linde. Locally optimal block quantizer design. Information
and Control, 45:178 { 198, 1980.
[53] D. Gujarati and D. Porter. Basic Econometrics. Mc-Graw Hill, 2009.
[54] P. Hacker, R. Krestel, S. Grundmann, and F. Naumann. Explainable AI under contract
and tort law: legal incentives and technical challenges. Arti cial Intelligence and Law,
2020.
[55] H. Hagras. Toward human-understandable, explainable ai. Computer, 51(9):28{36,
Sep. 2018.
[56] A. Halevy, P. Norvig, and F. Pereira. The unreasonable e ectiveness of data. IEEE
Intelligent Systems, March/April 2009.
[57] P. Halmos. Naive set theory. Springer-Verlag, 1974.
[58] T. Hastie, R. Tibshirani, and J. Friedman. The Elements of Statistical Learning.
Springer Series in Statistics. Springer, New York, NY, USA, 2001.
[59] T. Hastie, R. Tibshirani, and M. Wainwright. Statistical Learning with Sparsity. The
Lasso and its Generalizations. CRC Press, 2015.
[60] E. Hazan. Introduction to Online Convex Optimization. Now Publishers Inc., 2016.
[61] J. Howard and S. Ruder. Universal language model  ne-tuning for text classi cation.
In Proceedings of the 56th Annual Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 328{339, Melbourne, Australia, July 2018.
Association for Computational Linguistics.
[62] P. Huber. Approximate models. In C. Huber-Carol, N. Balakrishnan, M. Nikulin, and
M. Mesbah, editors, Goodness-of-Fit Tests and Model Validity. Statistics for Industry
and Technology. Birkhauser, Boston, MA, 2002.
[63] P. J. Huber. Robust Statistics. Wiley, New York, 1981.
[64] L. Hya l and R. Rivest. Constructing optimal binary decision trees is np-complete.
Information Processing Letters, 5(1):15{17, 1976.
272
[65] L. Hya l and R. L. Rivest. Constructing optimal binary decision trees is np-complete.
Information Processing Letters, 5(1):15{17, 1976.
[66] G. James, D. Witten, T. Hastie, and R. Tibshirani. An Introduction to Statistical
Learning with Applications in R. Springer, 2013.
[67] K. Jeong, J. Choi, and G. Jang. Semi-local structure patterns for robust face detection.
IEEE Sig. Proc. Letters, 22(9), 2015.
[68] A. Jung. A  xed-point of view on gradient methods for big data. Frontiers in Applied
Mathematics and Statistics, 3, 2017.
[69] A. Jung. Explainable empiricial risk minimization. submitted to IEEE Sig. Proc.
Letters (preprint: https://arxiv.org/pdf/2009.01492.pdf ), 2020.
[70] A. Jung. Networked exponential families for big data over networks. IEEE Access,
8:202897{202909, 2020.
[71] A. Jung, Y. Eldar, and N. Gortz. On the minimax risk of dictionary learning. IEEE
Trans. Inf. Theory, 62(3):1501 { 1515, Mar. 2016.
[72] A. Jung and P. Nardelli. An information-theoretic approach to personalized explainable
machine learning. IEEE Sig. Proc. Lett., 27:825{829, 2020.
[73] A. Jung and Y. SarcheshmehPour. Local graph clustering with network lasso. IEEE
Signal Processing Letters, 28:106{110, 2021.
[74] A. Jung and N. Tran. Localized linear regression in networked data. IEEE Sig. Proc.
Lett., 26(7), Jul. 2019.
[75] J. Kagan. Motives and development. Journal of Personality and Social Psychology,
22(1):51{66, 1972.
[76] S. M. Kay. Fundamentals of Statistical Signal Processing: Estimation Theory. Prentice
Hall, Englewood Cli s, NJ, 1993.
[77] T. Kibble and F. Berkshire. Classical Mechanics. Imperical College Press, 5 edition,
2011.
[78] P. Koehn. Europarl: A parallel corpus for statistical machine translation. In The 10th
Machine Translation Summit, page 79{86., AAMT,, Phuket, Thailand, 2005.
273
[79] D. Koller and N. Friedman. Probabilistic Graphical Models: Principles and Techniques.
Adaptive computation and machine learning. MIT Press, 2009.
[80] A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classi cation with deep convolutional
neural networks. In Neural Information Processing Systems, NIPS, 2012.
[81] B. Kulis and M. I. Jordan. Revisiting k-means: New algorithms via bayesian nonparametrics.
In Proc. of the 29th International Conference on Machine Learning, ICML
2012, Edinburgh, Scotland, UK, June 26 - July 1, 2012. icml.cc / Omnipress, 2012.
[82] C. Lampert. Kernel methods in computer vision. Foundations and Trends in Computer
Graphics and Vision, 2009.
[83] J. Larsen and C. Goutte. On optimal data split for generalization estimation and
model selection. In IEEE Workshop on Neural Networks for Signal Process, 1999.
[84] S. L. Lauritzen. Graphical Models. Clarendon Press, Oxford, UK, 1996.
[85] E. L. Lehmann and G. Casella. Theory of Point Estimation. Springer, New York, 2nd
edition, 1998.
[86] L. Li, W. Chu, J. Langford, and R. Schapire. A contextual-bandit approach to personalized
news article recommendation. In Proc. International World Wide Web Con-
ference, pages 661{670, Raleigh, North Carolina, USA, April 2010.
[87] Q. V. Liao, D. Gruen, and S. Miller. Questioning the ai: Informing design practices
for explainable ai user experiences. In Proceedings of the 2020 CHI Conference on
Human Factors in Computing Systems, CHI '20, pages 1{15, New York, NY, USA,
2020. Association for Computing Machinery.
[88] H. Lutkepohl. New Introduction to Multiple Time Series Analysis. Springer, New
York, 2005.
[89] A. Makhdoumi, S. Salamatian, N. Fawaz, and M. M edard. From the information
bottleneck to the privacy funnel. In 2014 IEEE Information Theory Workshop (ITW
2014), pages 501{505, 2014.
[90] S. G. Mallat. A Wavelet Tour of Signal Processing { The Sparse Way. Academic
Press, San Diego, CA, 3 edition, 2009.
274
[91] K. V. Mardia, J. T. Kent, and J. M. Bibby. Multivariate Analysis. Academic Press,
1979.
[92] J. McInerney, B. Lacker, S. Hansen, K. Higley, H. Bouchard, A. Gruson, and R. Mehrotra.
Explore, exploit, and explain: personalizing explainable recommendations with
bandits. In Proceedings of the 12th ACM Conference on Recommender Systems, 2018.
[93] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas. Communicatione
 cient learning of deep networks from decentralized data. In A. Singh and J. Zhu,
editors, Proceedings of the 20th International Conference on Arti cial Intelligence and
Statistics, volume 54 of Proceedings of Machine Learning Research, pages 1273{1282,
Fort Lauderdale, FL, USA, 20{22 Apr 2017. PMLR.
[94] C. Meyer. Generalized inversion of modi ed matrices. SIAM J. Appied Mathmetmatics,
24(3), 1973.
[95] C. Millard, editor. Cloud Computing Law. Oxford University Press, 2 edition, May
2021.
[96] T. Mitchell. The need for biases in learning generalizations. Technical Report CBM-TR
5-110,, Rutgers University, New Brunswick, New Jersey, USA, 1980.
[97] C. Molnar. Interpretable Machine Learning - A Guide for Making Black Box Models
Explainable. [online] Available: https://christophm.github.io/interpretable-ml-book/.,
2019.
[98] G. Montavon, W. Samek, and K. Muller. Methods for interpreting and understanding
deep neural networks. Digital Signal Processing, 73:1{15, 2018.
[99] K. Mortensen and T. Hughes. Comparing amazon's mechanical turk platform to conventional
data collection methods in the health and medical research literature. J.
Gen. Intern Med., 33(4):533{538, 2018.
[100] R. Muirhead. Aspects of Multivariate Statistical Theory. John Wiley & Sons Inc.,
1982.
[101] N. Murata. A statistical study on on-line learning. In D. Saad, editor, On-line Learning
in Neural Networks, pages 63{92. Cambridge University Press, New York, NY, USA,
1998.
275
[102] Y. Nesterov. Introductory lectures on convex optimization, volume 87 of Applied Opti-
mization. Kluwer Academic Publishers, Boston, MA, 2004. A basic course.
[103] M. E. J. Newman. Networks: An Introduction. Oxford Univ. Press, 2010.
[104] A. Ng. Shaping and Policy search in Reinforcement Learning. PhD thesis, University
of California, Berkeley, 2003.
[105] A. Y. Ng and M. I. Jordan. On discriminative vs. generative classi ers: A comparison of
logistic regression and naive bayes. In T. G. Dietterich, S. Becker, and Z. Ghahramani,
editors, Advances in Neural Information Processing Systems 14, pages 841{848. MIT
Press, 2002.
[106] A. Y. Ng, M. I. Jordan, and Y. Weiss. On spectral clustering: Analysis and an
algorithm. In Adv. Neur. Inf. Proc. Syst., 2001.
[107] S. Oymak, B. Recht, and M. Soltanolkotabi. Sharp time{data tradeo s for linear
inverse problems. IEEE Transactions on Information Theory, 64(6):4129{4158, June
2018.
[108] S. Pan and Q. Yang. A survey on transfer learning. IEEE Transactions on Knowledge
and Data Engineering, 22(10):1345{1359, 2010.
[109] A. Papoulis and S. U. Pillai. Probability, Random Variables, and Stochastic Processes.
Mc-Graw Hill, New York, 4 edition, 2002.
[110] N. Parikh and S. Boyd. Proximal algorithms. Foundations and Trends in Optimization,
1(3):123{231, 2013.
[111] J. Pearl. Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann, 1988.
[112] F. Pedregosa. Scikit-learn: Machine learning in python. Journal of Machine Learning
Research, 12(85):2825{2830, 2011.
[113] L. Pitt and L. G. Valiant. Computational limitations on learning from examples. J.
ACM, 35(4):965{984, Oct. 1988.
[114] T. Poggio, H. Mhaskar, L. Rosasco, B. Miranda, and Q. Liao. Why and when can deepbut
not shallow-networks avoid the curse of dimensionality: A review. International
Journal of Automation and Computing, 14(5):503{519, 2017.
276
[115] H. Poor. An Introduction to Signal Detection and Estimation. Springer, 2 edition,
1994.
[116] M. Ribeiro, S. Singh, and C. Guestrin. \Why should i trust you?": Explaining the
predictions of any classi er. In Proc. 22nd ACM SIGKDD, pages 1135{1144, Aug.
2016.
[117] M. Ribeiro, S. Singh, and C. Guestrin. Anchors: High-precision model-agnostic explanations.
In Proc. AAAI Conference on Arti cial Intelligence (AAAI), 2018.
[118] S. Roweis. EM Algorithms for PCA and SPCA. In Advances in Neural Information
Processing Systems, pages 626{632. MIT Press, 1998.
[119] W. Rudin. Principles of Mathematical Analysis. McGraw-Hill, New York, 3 edition,
1976.
[120] W. Rudin. Real and Complex Analysis. McGraw-Hill, New York, 3rd edition, 1987.
[121] S. J. Russel and P. Norvig. Arti cial Intelligence - A Modern Approach. Prentice Hall,
New York, 3 edition, 2010.
[122] N. P. Santhanam and M. J.Wainwright. Information-theoretic limits of selecting binary
graphical models in high dimensions. IEEE Trans. Inf. Theory, 58(7):4117{4134, Jul.
2012.
[123] Y. SarcheshmehPour, M. Leinonen, and A. Jung. Federated learning from big data
over networks. In Proc. IEEE Int. Conf. on Acoustics, Speech and Signal Processing
(ICASSP), preprint: https://arxiv.org/pdf/2010.14159.pdf, 2021.
[124] F. Sattler, K. Muller, and W. Samek. Clustered federated learning: Model-agnostic
distributed multitask optimization under privacy constraints. IEEE Transactions on
Neural Networks and Learning Systems, 2020.
[125] B. Scholkopf and A. Smola. Learning with Kernels: Support Vector Machines, Regu-
larization, Optimization, and Beyond. MIT Press, Cambridge, MA, USA, Dec. 2002.
[126] S. Shalev-Shwartz and S. Ben-David. Understanding Machine Learning { from Theory
to Algorithms. Cambridge University Press, 2014.
[127] C. E. Shannon. Communication in the presence of noise. 1948.
277
[128] Y. Y. Shkel, R. S. Blum, and H. V. Poor. Secrecy by design with applications to
privacy and compression. IEEE Transactions on Information Theory, 67(2):824{843,
2021.
[129] S.Levine, C. Finn, T. Darrell, and P.Abbeel. End-to-end training of deep visuomotor
policies. J. Mach. Learn. Res., 17, 2016.
[130] V. Smith, C.-K. Chiang, M. Sanjabi, and A. Talwalkar. Federated multi-task learning.
In Advances in Neural Information Processing Systems, volume 30, 2017.
[131] S. Smoliski and K. Radtke. Spatial prediction of demersal  sh diversity in the baltic
sea: comparison of machine learning and regression-based techniques. ICES Journal
of Marine Science, 74(1):102{111, 2017.
[132] A. Sorokin and D. Forsyth. Utility data annotation with amazon mechanical turk. In
2008 IEEE Computer Society Conference on Computer Vision and Pattern Recognition
Workshops, pages 1{8, 2008.
[133] S. Sra, S. Nowozin, and S. J. Wright, editors. Optimization for Machine Learning.
MIT Press, 2012.
[134] G. Strang. Computational Science and Engineering. Wellesley-Cambridge Press, MA,
2007.
[135] G. Strang. Introduction to Linear Algebra. Wellesley-Cambridge Press, MA, 5 edition,
2016.
[136] R. Sutton and A. Barto. Reinforcement learning: An introduction. MIT press, Cambridge,
MA, 2 edition, 2018.
[137] M. E. Tipping and C. Bishop. Probabilistic principal component analysis. Journal of
the Royal Statistical Society, Series B, 21/3:611{622, January 1999.
[138] M. E. Tipping and C. M. Bishop. Probabilistic principal component analysis. J. Roy.
Stat. Soc. Ser. B, 61:611{622, 1999.
[139] N. Tishby and N. Zaslavsky. Deep learning and the information bottleneck principle.
In 2015 IEEE Information Theory Workshop (ITW), pages 1{5, 2015.
278
[140] N. Tran, O. Abramenko, and A. Jung. On the sample complexity of graphical model
selection from non-stationary samples. IEEE Transactions on Signal Processing, 68:17{
32, 2020.
[141] N. Tran, H. Ambos, and A. Jung. Classifying partially labeled networked data via
logistic network lasso. In Proc. IEEE Int. Conf. on Acoustics, Speech and Signal
Processing (ICASSP), pages 3832{3836, Barcelona, Spain, May 2020.
[142] L. G. Valiant. A theory of the learnable. In Proceedings of the Sixteenth Annual ACM
Symposium on Theory of Computing, STOC '84, pages 436{445, New York, NY, USA,
1984. Association for Computing Machinery.
[143] S. A. van de Geer and P. Buhlmann. On the conditions used to prove oracle results
for the Lasso. Electron. J. Statist., 3:1360 { 1392, 2009.
[144] V. N. Vapnik. The Nature of Statistical Learning Theory. Springer, 1999.
[145] O. Vasicek. A test for normality based on sample entropy. Journal of the Royal
Statistical Society. Series B (Methodological), 38(1):54{59, 1976.
[146] R. Vidal. Subspace clustering. IEEE Signal Processing Magazine, 52, March 2011.
[147] U. von Luxburg. A tutorial on spectral clustering. Statistics and Computing, 17(4):395{
416, Dec. 2007.
[148] S. Wachter. Data protection in the age of big data. Nature Electronics, 2(1):6{7, 2019.
[149] S. Wade and Z. Ghahramani. Bayesian Cluster Analysis: Point Estimation and Credible
Balls (with Discussion). Bayesian Analysis, 13(2):559 { 626, 2018.
[150] M. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge:
Cambridge University Press, 2019.
[151] M. J. Wainwright. Information-theoretic limits on sparsity recovery in the highdimensional
and noisy setting. IEEE Trans. Inf. Theory, 55(12):5728{5741, Dec. 2009.
[152] M. J. Wainwright and M. I. Jordan. Graphical Models, Exponential Families, and
Variational Inference, volume 1 of Foundations and Trends in Machine Learning. Now
Publishers, Hanover, MA, 2008.
279
[153] A. Wang. An industrial-strength audio search algorithm. In International Symposium
on Music Information Retrieval, Baltimore, MD, 2003.
[154] W. Wang, M. J. Wainwright, and K. Ramchandran. Information-theoretic bounds on
model selection for Gaussian Markov random  elds. In Proc. IEEE ISIT-2010, pages
1373{1377, Austin, TX, Jun. 2010.
[155] J. Wright, Y. Peng, Y. Ma, A. Ganesh, and S. Rao. Robust principal component
analysis: Exact recovery of corrupted low-rank matrices by convex optimization. In
Neural Information Processing Systems, NIPS 2009, 2009.
[156] L. Xu and M. Jordan. On convergence properties of the EM algorithm for Gaussian
mixtures. Neural Computation, 8(1):129{151, 1996.
[157] X. Yang and Q. Wang. Crowd hybrid model for pedestrian dynamic prediction in a
corridor. IEEE Access, 7, 2019.
[158] K. Young. Bayesian diagnostics for checking assumptions of normality. Journal of
Statistical Computation and Simulation, 47(3{4):167 { 180, 1993.
280
