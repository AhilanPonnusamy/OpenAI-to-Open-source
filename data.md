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
