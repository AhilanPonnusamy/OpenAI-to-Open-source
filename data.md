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
