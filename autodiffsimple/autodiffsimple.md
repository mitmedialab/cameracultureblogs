Automatic differentiation from scratch: forward and reverse modes
=================================================================

Tristan Swedish  
Camera Culture, MIT Media Lab

There is an extremely powerful tool that has gained popularity in recent years that has an unreasonable number of applications, ranging from computational design, robotic control, imaging and graphics, financial analysis, and machine learning. Nope, this post is not about Deep Learning, this post is about [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) (or auto-diff, or AD). In this post, we will develop a basic auto-diff library from scratch in python using only standard library functions. (TL;DR here is the [github](https://github.com/mitmedialab/cameracultureblogs/tree/master/autodiffsimple), with text from the post available as well).

* * *

Why would I want to differentiate computer programs?
====================================================

Computers are really good at simulation. If you’ve played any video games recently, or used any engineering design tools, I’m sure you’ll agree. Simulations are undoubtably very useful, but we naturally encounter the _inverse problem_: “What is a likely input to a simulation, given only the output?”

*   **Design**: Given a simulation of airflow around a _Formula 1_ race car, how should the wing be changed to [improve down-force](https://www.youtube.com/watch?v=hU0Whx7EZic)? 🏎️
*   **Control**: You have a rocket, what sequence of gimbal movements will make it [land upright](https://www.youtube.com/watch?v=ANv5UfZsvZQ)? 🚀
*   **Image Processing**: We can model the blur created by a shaky camera, what does the [“deblurred” scene](https://www.youtube.com/watch?v=IlcLa5JeTrE) look like? 📷
*   **Machine Learning**: With a massive dataset and an architecture for a neural network, what connection weights will [give good results](https://www.youtube.com/watch?v=kSLJriaOumA)? 👤

It can be extremely useful to solve these inverse problems, but how do we go about this? Can we leverage everything we know about the forward simulation code to solve the inverse?

Solving the inverse problems above can require significant domain expertise. Such specialization can be important for performance, but also can be error prone and difficult to adapt to new problems. Auto-diff promises a more general purpose approach, which calculates the gradient of forward simulation code automatically. This makes it possible to use _gradient descent_ to potentially solve these inverse problems, such as the ones above (see related examples applying auto-diff [here](https://www.youtube.com/watch?v=bWw8_m_S3bo), [here](https://www.youtube.com/watch?v=rWmw-ERGyz4), [here](https://people.csail.mit.edu/tzumao/gradient_halide/gradient_halide.pdf), and [here](https://www.youtube.com/watch?v=Ilg3gGewQ5U)). Gradient descent might not always be the “correct” approach, but it provides a useful starting place, and in some cases works surprisingly well. (🤓 : _Optimization and inverse problems are huge fields of their own and auto-diff is no exception to the_ [_TNSTAAFL_](https://en.wikipedia.org/wiki/There_ain%27t_no_such_thing_as_a_free_lunch)_. That said, auto-diff provides a level of generality, making it widely applicable and useful._)

If you’ve heard of auto-diff, it’s probably by using frameworks like PyTorch or Theano, which use it for calculating gradients to train deep neural networks. Until deep-learning frameworks popularized it, the field of AD had been surprisingly obscure. Even now, there is a perception that it is basically magic. The goal of this post is to show how auto-diff really works for both forward and reverse modes, without getting too bogged down in the details. Even with these simple examples, I hope you can appreciate what auto-diff could do for you.

It’s all in the abstraction
===========================

At the core, auto-diff uses a special type of number that makes it possible to differentiate complicated functions. Thus, understanding auto-diff is all about understanding abstraction. The philosophy and motivation for the kind of abstraction I’m talking about is explored in this lecture by Richard Feynman, as part of his famous series on physics: [Feynman Lecture on Algebra](https://www.feynmanlectures.caltech.edu/I_22.html). It’s such a beautiful idea, mathematicians keep discovering new things that are increasingly abstract, but surprisingly, we keep finding use for them ( 🤓 : _and the mathematicians_).

To make this a bit more concrete, let’s write a python function that does some computation:

You’ll notice that we compose our function using a few primitives: `sin`, `**` (or `pow`), `+`, `-`, `*`, `/`. We can compose expressions, and the python interpreter handles everything to make this function do what we expect algebraically. Importantly, operators like `+` are just functions that take two arguments but use a fancy syntax.

We naturally expect the above function to operate on floating point input, but we could pass a new type (`class`) into our function and carefully overload the necessary functions (`+`, `*`, etc) so that the rules of algebra still apply.

Example: Complex Numbers
========================

To better understand this idea, let us consider the example of _complex numbers_. We can write complex numbers like this:

<img class="s t u jp ai" src="https://miro.medium.com/max/140/1\*eNP1clfKTaD5spTKEOGcZA.png" width="70" height="22" role="presentation"/>

where

<img class="s t u jp ai" src="https://miro.medium.com/max/186/1\*zXCYLeKLSUO0ukuAxHJuYw.png" width="93" height="26" role="presentation"/>

As long as we overload the correct operators, such as `*` and `+`, we can define a complex number class that has an algebraic structure that is practically the same as for floating point numbers, except we now bring along the ability to handle the _imaginary component_. With complex numbers, we can compute the result of these operations by using normal algebra. For example here is addition and multiplication:

<img class="s t u jp ai" src="https://miro.medium.com/max/910/1\*5jtkPzX2VyB7dCwkWJON1A.png" width="455" height="29" srcSet="https://miro.medium.com/max/552/1\*5jtkPzX2VyB7dCwkWJON1A.png 276w, https://miro.medium.com/max/910/1\*5jtkPzX2VyB7dCwkWJON1A.png 455w" sizes="455px" role="presentation"/>

<img class="s t u jp ai" src="https://miro.medium.com/max/948/1\*oIrlsGmYCPou1VBJdlybXw.png" width="474" height="29" srcSet="https://miro.medium.com/max/552/1\*oIrlsGmYCPou1VBJdlybXw.png 276w, https://miro.medium.com/max/948/1\*oIrlsGmYCPou1VBJdlybXw.png 474w" sizes="474px" role="presentation"/>

We can then run a complex number through a function that consists of `*` and `+` (and more operations as long as we properly define them), and it would do what we expect, except we get both the real and imaginary valued output.

Dual Numbers
============

There is a kind of number very similar to complex numbers that give us the properties we need to perform auto-diff. _Dual numbers_ have the nice property that when you calculate with them, they bring along their own derivative.

Dual-numbers can be defined in similar way to complex numbers:

<img class="s t u jp ai" src="https://miro.medium.com/max/140/1\*qc1Wqrs3ZeHriLkDYr1cIw.png" width="70" height="22" role="presentation"/>

except

<img class="s t u jp ai" src="https://miro.medium.com/max/146/1\*otDxACCy1GUdYeA3L5guaA.png" width="73" height="26" role="presentation"/>

Now, like with complex numbers, we can replace all our normal numbers with these new magical numbers.

**Here’s the intuition**: when computing with dual numbers, you’re computing a first order approximation of the function for a specific argument. When performing multiplication, higher order terms that preserve the approximation everywhere are discarded. It’s this first order approximation that gives it utility for computing the derivative. Let’s imagine we have two linear polynomials, and we want to calculate the output for all values to the resulting function after applying addition or multiplication. We draw the graphs of these corresponding operations like this:

<img class="s t u jp ai" src="https://miro.medium.com/max/2048/1\*c3RFBKMAbgtarCwgzEIMXw.png" width="1024" height="768" srcSet="https://miro.medium.com/max/552/1\*c3RFBKMAbgtarCwgzEIMXw.png 276w, https://miro.medium.com/max/1104/1\*c3RFBKMAbgtarCwgzEIMXw.png 552w, https://miro.medium.com/max/1280/1\*c3RFBKMAbgtarCwgzEIMXw.png 640w, https://miro.medium.com/max/1400/1\*c3RFBKMAbgtarCwgzEIMXw.png 700w" sizes="700px" role="presentation"/>

Now, we have functions of 𝜖, and show the result of combining two such functions for `+` and `*`. If we want to perfectly model the resulting function for all values of 𝜖, we need to keep higher order terms. Multiplication makes these terms grow (see red text above). If you’ve noticed that this looks like a Taylor series expansion, well you’d be right! What’s nice though, is if we only care about infinitesimal perturbations to our function, we can throw away these high order terms (blue text above).

What about division?
--------------------

There’s a problem though, computing with dual numbers isn’t actually analogous to real numbers since we don’t have a `/` operation. This can be shown by asking, what's the element we can multiply with a dual number to get 1? If we do the reasonable thing, we find:

<img class="s t u jp ai" src="https://miro.medium.com/max/188/1\*LT2D5QdJRG-fo0w-j3NVgw.png" width="94" height="57" role="presentation"/>

This, however, is not well defined for all `(a, b)`, since we have a division by zero when `(0, b)`. Without a single "zero" element, we don't have a well defined multiplicative inverse. 😞

( 🤓 : _In abstract algebra terms, the dual numbers with_ `_+_` _and_ `_*_` _are a ring, while the reals have a multiplicative inverse, making them a field along with_ `_+_` _and_ `_*_`_. This matches our intuition, since dual numbers kinda seem like a polynomial ring, but we lop off any higher-order terms after performing a multiplication. We can keep these higher order terms if we want, and this becomes a Taylor polynomial algebra, but then we need more and more terms for repeated multiplications and this can become unwieldy._)

In practice, maybe this is silly, since with real numbers we don’t divide by zero anyway, so we’ll never want to find the multiplicative inverse of `(0, b)` for a function that works using real numbers as arguments. When using auto-diff frameworks, issues like this can arise, so it’s important to be aware of such things when debugging.

Building more operations
------------------------

We can use the derivation above to define `/` as multiplying by the dual number above. Now, can we do something similar for other functions like `pow`, noting that `/` could also be defined using `pow(x, y)` with `y == -1`? We can do just a bit more math to come up with these definitions.

A general definition for `pow(x, y)` can be found using the binomial theorem, noting that any high order terms go to zero.

<img class="s t u jp ai" src="https://miro.medium.com/max/606/1\*-P4zmVrIsGzQBTgiiBNOsg.png" width="303" height="32" srcSet="https://miro.medium.com/max/552/1\*-P4zmVrIsGzQBTgiiBNOsg.png 276w, https://miro.medium.com/max/606/1\*-P4zmVrIsGzQBTgiiBNOsg.png 303w" sizes="303px" role="presentation"/>

Look, we get the well known power rule using algebra! I love this because we don’t even need to use the traditional definition of the derivative that uses limits. Note that this also generalizes our division operation above. ( 🤓 : `_pow()_` _is particularly interesting, and there's some more detail I didn't cover here, the code section below shows the full rule_.)

Translating this into code, we can overload python’s arithmetic operators as follows:

I’ve omitted the rather repetitive definition of inverse operators (like `__sub__`), since we can get those easily by composing the above operations. As an example, `__truediv__` which overloads `/` is shown as composing multiplication and power. For each operator, we can also automatically "cast" floating point constants into duals. Here is an example of the definition for`__add__`:

We can add more operators, such as `exp()` and `sin()` using the familiar derivative rules ( 🤓 : _in principle, we can derive the rules ourselves using algebra, e.g. equating Euler's formula for_ `_sin()_` _and its power series_). However, as long as we define the output of an operation such that we output the value of the function, as well as its derivative multiplied by the derivative value of the input `Dual`, we can compose these operators and everything works as expected. For example, here is `sin()`:

Implementing Automatic Differentiation
======================================

Forward Mode AD
---------------

Now, we can perform Forward Mode AD practically right away, using the `Dual` numbers class we've already defined.

Let’s draw a graph visualizing our function.

<img class="s t u jp ai" src="https://miro.medium.com/max/1594/1\*9mXmw-gCyZVzyOua4A80vQ.png" width="797" height="495" srcSet="https://miro.medium.com/max/552/1\*9mXmw-gCyZVzyOua4A80vQ.png 276w, https://miro.medium.com/max/1104/1\*9mXmw-gCyZVzyOua4A80vQ.png 552w, https://miro.medium.com/max/1280/1\*9mXmw-gCyZVzyOua4A80vQ.png 640w, https://miro.medium.com/max/1400/1\*9mXmw-gCyZVzyOua4A80vQ.png 700w" sizes="700px" role="presentation"/>

Here, the operations that make up our function are shown as nodes, and their arguments are shown in order (top to bottom). We show red numbers to represent partial derivatives that form the _jacobian_ of each node at our chosen input (`x = 3`). For example, for the division operation in the top right, we compute the partial derivatives with respect to both inputs. We use a dual number with 𝜖 = 1 for the input we would like to compute the partial for, and 𝜖 = 0 for the other input. The partial derivative of the first argument can be written:

<img class="s t u jp ai" src="https://miro.medium.com/max/506/1\*OVzEw0uOVOkrKVX6BR1Ayg.png" width="253" height="60" role="presentation"/>

so now the partial (the red value) is set to `0.5`. The partial with respect to the next argument to the operation is:

<img class="s t u jp ai" src="https://miro.medium.com/max/1018/1\*9x5u5LVB59wOoGIiEapRTg.png" width="509" height="68" srcSet="https://miro.medium.com/max/552/1\*9x5u5LVB59wOoGIiEapRTg.png 276w, https://miro.medium.com/max/1018/1\*9x5u5LVB59wOoGIiEapRTg.png 509w" sizes="509px" role="presentation"/>

using the division rule, so the corresponding entry in the jacobian (red value) is set to `-0.13`.

In order to calculate the derivative with respect to the output, we start at a node and multiply all the red numbers following the path through the tree to the output. If we have a variable like `x` with multiple paths, we sum the product for each path. This may be a slightly weird way to think about computing with dual numbers, but it makes it much easier to understand _reverse mode_ (next section).

Using the dual numbers described above, we can define a function that automatically generates the jacobian for each operation. We run the function in a loop, setting all `.derivative` members to 0, except for the particular input we are interested in:

The function above returns a function that calculates the partial derivative with respect to each input to our function. We note that this requires we run the function multiple times, making our computational complexity pretty poor for functions with many inputs. To solve this problem, we could cache intermediate results for single forward pass and multiply red values _backward_ from the output to each leaf of the graph (totally valid since multiplication is associative). This is called “Reverse Mode AD,” and is the mode used for things with many input parameters like Deep Neural Networks. Reverse Mode is also called the “Adjoint State Method,” which sounds simultaneously boring and scary, but it’s actually super awesome.

Reverse Mode AD
---------------

This algorithm is also called “backpropagation.” If you’ve trained neural networks before, you probably love backprop. We all love backprop. Backprop loves backprop \[[https://arxiv.org/pdf/1606.04474.pdf](https://arxiv.org/pdf/1606.04474.pdf)\].

The trick to implement Reverse Mode is to somehow get the tree structure, like above, automatically. We could think of this as “tracing” through our function evaluation in order to build the above graph. We will accomplish this by overloading, by defining a new type (`class`) that behaves like a number, but also builds a history of the computations used to construct it. Basically, we want a `Variable` that could be used like this:

Variable Class
--------------

In this section, I cover almost all the code you need (excluding most of the overloaded arithmetic operators to stay concise). Amazingly, the meat of the implementation is quite compact, and can be described by the following few boxes of code:

Our `Variable` class constructor takes an operation (or function) and the arguments to the operation as other `Variables`. We also make a place to store the value of the gradient.

The above code uses a @property to make the interface a bit nicer, but actually leads to a bit more code than is strictly necessary. At a high level, we want to create a `forward_op` that calculates the output of our operation, and a `backward_op` that determines what's called the _vector-jacobian product_ of our operation. The main tricky piece is `calc_input_variables` which recursively calls `forward()` on the input variables until a constant value input is encountered, from there, `value` can be calculated. We handle the constant valued operations in the `value` property setter. Note that for simplicity there is no caching of intermediate results, but this would be simple to add in the `value` property.

This completes our bookkeeping for running the operation forward. Below, this is what we need for the backward logic. This is the key idea used in Reverse Mode, so it’s worth understanding what’s going on here.

Below is an example of defining an operation. We could pass in any function handle as an `operation`, but when overloading basic math functions, we create an anonymous (lambda) function.

Clearly, this code is not super efficient. For one, it would be straightforward to cache results when running `forward` so that the function does not need to be called multiple times. But in principle, this is all that's required to make Reverse AD work!

As we can see, our `super_complicated_function` returns what we expect:

Hybrid Mode AD
--------------

There are a few extensions we don’t include in our code that are still good to know about. Using a method called “checkpointing,” we can save a select few intermediate results and recompute values by calling forward on subsections of the computation graph. This is a memory saving technique, but requires additional computation. It can be thought of as a sort of combination of forward and backward modes, so it’s called “Hybrid Mode AD.” ( 🤓 : _It’s known that finding the optimal trade-off between checkpointing and running forward and backward to balance time and memory complexity is NP-hard. This means that we probably won’t find the most efficient way to compute the derivatives for arbitrary programs. Luckily, there are some good heuristics that actually do a pretty good job_.)

Conclusion: Simple Python Library to learn auto-diff
====================================================

Anyway, our library is definitely not very efficient, and more production grade AD libraries often use smart caching of intermediate results and perform various optimizations (🤓 : _such as using a “gradient tape”_). Regardless, our simple little library can do some really neat things. It gracefully handles control flow, and can differentiate pretty complex programs already.

Since we build a computation graph dynamically, our `Variable` class can handle branching and loops.

This “feature” we get for free highlights an important consideration when using auto-diff. We only differentiate the input variables with respect to the branch the particular run of the program took. We get correct derivatives, but conditional statements can make our function non-differentiable in some places. For example the `relu()` function used in neural networks can be written `max(0, x)`, which contains a conditional ( `0 if x < 0 else x`), and it's not really differentiable at `x==0` ( 🤓 _we might say it's subdifferentiable_). Furthermore, if we had any random numbers that are used in these conditionals, our derivatives are only valid for that particular sample from the random number generator. This is a subtle point, but important to keep in mind.

We’ve implemented a simple library that handles composing scalar functions and real numbers, but real AD systems can typically handle other types, such as vectors and matrices using the rules of linear algebra.

And as a final proof of concept, let’s solve a simple inverse problem. I’ve defined a polynomial, and set a target output of the function to `42`. With an initial guess for an input to the function as `3`, we minimize the square error between the output at our initial guess and target output of `42` using our AD library and gradient descent:

Got it! While we can sometimes solve such inverse problem analytically, in general we cannot, but the basic gradient descent algorithm can still be used to try to find the solution.

This post hopefully provides you with a good intuition for using AD in real problems. By carefully constructing types and overloading the right operators, we end up with a rather elegant way to differentiate computer programs. The general principle of abstraction via function overloading can be applied to other “morphisms”, such as those used in [Homomorphic Encryption](https://en.wikipedia.org/wiki/Homomorphic_encryption), or [Auto-vectorization](https://en.wikipedia.org/wiki/Automatic_vectorization), making AD a neat way to gain intuition about these other feats of modern computer science. Furthermore, while I love python, it’s generic programming capabilities aren’t the most flexible, and a better language for real applications is probably writing a C++ Template library like what’s used by [Enoki](https://enoki.readthedocs.io/en/master/demo.html). [JAX](https://github.com/google/jax) is a notable project out of google for AD+NumPy that supports some nifty things like JIT using the Tensorflow XLA compiler. Other well known frameworks that use auto-differentiation include [Tensorflow](https://www.tensorflow.org/guide/eager) and [PyTorch](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html), which are great for linear algebra (and thus deep neural networks).

Anyway, I hope AD is now not so mysterious to you, but is perhaps even more magical. 😄

Further Reading
---------------

There are some other articles that I’ve found super useful, about halfway through writing this post, I found [rufflewind’s post](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation), who covers some additional topics like using “tape” based storage of intermediate values and some memory saving optimizations. A [recent review](http://jmlr.org/papers/volume18/17-468/17-468.pdf) on AD in machine learning is also a more academic resource with more references for further reading.

The code for the embedded examples in the form of a Jupyter Notebook and the full class definitions (with more of the operators overloaded and some pretty-print functions) can be found on [github](https://github.com/mitmedialab/cameracultureblogs/tree/master/autodiffsimple).



