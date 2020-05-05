# The surprising simplicity of automatic differentiation 
Tristan Swedish
Camera Culture
MIT Media Lab

There is an extremely powerful tool that has gained popularity in recent years that has an unreasonable number of applications, particularly to the problem of perception, computational imaging, and machine learning. Nope, this post is not about Deep Learning, this post is about [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) (or AD), the cryptic tool that makes it possible to train neural networks and differentiate pretty general programs. In this post, we will develop a basic AD library from scratch using only standard python functions [github](https://github.com/mitmedialab/cameracultureblogs/autodiffsimple).

## Differentiating Computer Programs

Being able to calculate the derivative of functions defined in computer programs has broad applications. In essence, it allows you to understand how perturbations to the program input change the output. The naive way to do this is to use sampling based methods like finite differences, running the program many times with slightly adjusted inputs to see how the output changes. For a function with N variables or arguments, these methods typically require we call the function N+1 times. This is computationally unattractive, and even worse, we need to know how much to perturb the arguments to the function, leading to numerical problems.

However, AD seeks to calculate these derivatives without running the program many times (typically O(a) for a << N), and to compute the derivatives exactly for a given function input, making numerical stability much less of a problem.

As a motivating example, we will consider the problem of guessing the input to a function defined in python when we’re only given it’s output. This is a common class of problems encountered in robotics, signal processing, and computational imaging. 

Basically, this is an inverse problem: we’d like to figure out the input to a function that produces the outputs we observe. For example, given the output from a rendering engine, we might perform “Inverse Graphics”: we want to take an image and figure out what parameters in our graphics program produce the same image. As such, if we minimize the difference between the given output and the output of our function by choosing different input parameters, we get closer to the solution to the inverse problem.

In general, we can approach this problem as follows:

Step 1. Obtain the source code for the function we’re interested in


Step 2 (what this post is all about). Translate the program so that we can differentiate the whole program output with respect to the inputs


Step 3. Solve the inverse problem using gradient descent

This post focuses on Step 2, demonstrating the simplicity of AD under the hood, performing a beautiful, almost magical transformation of the function so that we can obtain the derivatives by running this transformed version of our original function defined in our program.
## It’s all about Abstraction

A common thread in physics and engineering, is the generalization of things we can calculate with more abstract mathematical objects to better explain real physical stuff. In other words, math gives us the vocabulary to define and solve problems. This is explained really well in Richard Feynman’s famous lectures on physics, maybe my favorite lecture ever: Feynman Lecture on Algebra. It’s such a beautiful idea, mathematicians keep discovering new things that are increasingly abstract, but surprisingly, we keep finding use for them (and the mathematicians...).

Computer Science also loves abstraction and composition, as it allows us to reason about how objects in computer programs can be combined to generate new objects. A relevant branch of mathematics is “Abstract Algebra” and I’m going to use it as inspiration for explaining AD. We’re not going to be very rigorous, but it’s incredibly useful to use Abstract Algebra as a kind of roadmap for what we’re trying to do. You can read more here: [a blog post](https://jrsinclair.com/articles/2019/algebraic-structures-what-i-wish-someone-had-explained-about-functional-programming/), [a book](https://www.fm2gp.com/).

So... what is the (algebraic) structure of our function?

For our example, we want to write generic programs that take objects (or if you will: types), and produce output with objects of the same type. In other words, we could write a python function:

```
def super_complicated_function(x):
	return x * sin(x+6.)**2. / 2. - 2. / 2.**x
```

If we pass in something to this function and can compute a sensible result, we are basically dealing with a structure that looks a lot like a “field” in abstract algebra. These types or objects do what we expect with operators like:  `+`,`-`,`*`,` /`. We can compose expressions, and the python interpreter handles everything to make this function do what we expect algebraically (like obeying operator precedence with `()`). We also note we can’t divide by zero, but we can divide by any other number and the output is not ambiguous. The basic thing that behaves like this are the Real numbers, so we naturally expect the above function to run using floating point types. 

If we pass a new object into our function that overloads these operators in a mathematically sensible way, the rules still apply and the function does not need to be modified. This is typically known as “generic” programming, and in those terms, we can run this function for any other type that also has the corresponding operators used by the function. Python performs duck-typing and automatic type casting, so the function above is actually pretty generic.
Example: Complex Numbers

A familiar example of a type we can define that our function will also accept are complex numbers. Without even running the program we can be confident our calculations will be algebraically correct. We know this because complex numbers form a field with `*` and `+`, thus the underlying algebraic structure is the same as for the reals. We can write complex numbers like this:

![](http://www.sciweavers.org/tex2img.php?eq=a%20%2B%20i%20b&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0)

Where ![](http://www.sciweavers.org/tex2img.php?eq=i%5E2%20%3D%20-1&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0). If we created a complex number class in python that overloads the correct operators, we can then run that number through our function above without any modification, and it would do what we expect, except we get both the real and imaginary valued output.
##Dual Numbers

There is a kind of number very similar to complex numbers that give us the properties we need to perform AD. Dual numbers have the nice property that when you calculate with them, they bring along their own derivative. 

Dual-numbers can be defined in similar way to complex numbers:

![](http://www.sciweavers.org/tex2img.php?eq=a%20%2B%20%5Cepsilon%20b&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0)

Where ![](http://www.sciweavers.org/tex2img.php?eq=%5Cepsilon%5E2%20%3D%200&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0). Note that ![](http://www.sciweavers.org/tex2img.php?eq=%5Cepsilon%5Ek%20%3D%200&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0) when k > 2 as well, which we get by factoring out  ![](http://www.sciweavers.org/tex2img.php?eq=%5Cepsilon%5E2&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0). Now, we can replace all our normal numbers with these new magical numbers. 

Here’s the intuition: when computing with dual numbers, you’re computing a first order approximation of the function for a specific argument. When performing multiplication, higher order terms that preserve the approximation everywhere are discarded. It’s this first order approximation that gives it utility for computing the derivative. Let’s imagine we have two first-order polynomials, and we want to calculate the output for all values to the resulting function after applying addition and multiplication. We could write this out as follows:



In this interpretation, we have a function of ![](http://www.sciweavers.org/tex2img.php?eq=%5Cepsilon&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0), where the output of our function with no perturbation is simply “a”, and perturbed values change linearly. If you’ve noticed that this looks like a Taylor series expansion, well you’d be right! If we want to perfectly model the resulting function for all values of ![](http://www.sciweavers.org/tex2img.php?eq=%5Cepsilon&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0), we need to keep higher order terms. Multiplication makes these terms grow. What’s nice though, is if we only care about infinitesimal ![](http://www.sciweavers.org/tex2img.php?eq=%5Cepsilon&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0) perturbations to our function, we can throw away these high order terms.

There’s a problem though, computing with dual numbers isn’t actually analogous to real numbers since we don’t have a `/`. This can be shown by asking, what’s the element we can multiply with ![](http://www.sciweavers.org/tex2img.php?eq=a%20%2B%20ib&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0) to get 1? If we do the reasonable thing, we find ![](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B1%7D%7Ba%7D%20%2B%20%20-%5Cepsilon%20%5Cfrac%7Bb%7D%7Ba%5E2%7D&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0), which is not well defined for all `(a,b)`, since we have a division by zero when `(0,b)`. Without a single “zero” element, we don’t have a well defined multiplicative inverse. :(

(In abstract algebra terms, the dual numbers with `+` and `*` are a ring, while the reals have a multiplicative inverse, making them a field along with `+` and `*`. This matches our intuition, since dual numbers kinda seem like a polynomial ring, but we lop off any higher-order terms after performing a multiplication. We can keep these higher order terms if we want, and this becomes a Taylor polynomial algebra, but our memory requirements grow significantly for repeated multiplications.)
### What about division?

In practice, maybe this is silly, since with real numbers we don’t divide by zero anyway, so we’ll never want to find the multiplicative inverse of ![](http://www.sciweavers.org/tex2img.php?eq=0%20%2B%20%5Cepsilon%20b&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0) for a well formed function that works on the reals.

We can use the derivation above to define `/` as multiplying by ![](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B1%7D%7Ba%7D%20%2B%20%20-%5Cepsilon%20%5Cfrac%7Bb%7D%7Ba%5E2%7D&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0). Now, noting that `1 / x` is `x**-1`, can we come up with a general definition for `pow(x, y)` that works for all `y` and not just `y == -1`? We can do just a bit more math to come up with these definitions.

### Building More Operations

A general definition for `pow(x,y)` can be found using the binomial theorem, noting that any high order terms go to zero since ![](http://www.sciweavers.org/tex2img.php?eq=%5Cepsilon%5E2&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0):

![](http://www.sciweavers.org/tex2img.php?eq=%28a%20%2B%20%5Cepsilon%20b%29%5En%20%3D%20a%5En%20%2B%20%5Cepsilon%20b%20a%5E%7Bn-1%7D&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0)

Look, we get the well known power rule using algebra! I love this because we don’t even need to use the traditional definition of the derivative that uses limits.

Translating this into code, we can overload python’s arithmetic operators as follows:

```
class Dual:
    def __init__(self, value=0., derivative=0.):
        self.value = value
        self.derivative = derivative
        
    def __add__(self, other):
            return Dual(self.value + other.value, self.derivative + other.derivative)
    
    def __mul__(self, other):
            return Dual(self.value * other.value, self.value*other.derivative + self.derivative * other.value)

     def __pow__(self, other):
             return Dual(self.value**other.value, self.derivative * other.value * self.value**(other.value-1))
            
    def __truediv__(self, other):
        return self * other**(-1.)
 
```

I’ve omitted the rather repetitive definition of inverse operators (like `__sub__`), since we can get those easily by composing the above operations. As an example, `__truediv__` which overloads `/` is shown as composing multiplication and power. 
For each operator, we can also automatically “cast” floating point constants into duals. Here is an example of the definition for `__add__`:

```
def __add__(self, other):
         if isinstance(other, self.__class__):
            return Dual(self.value + other.value, self.derivative + other.derivative)
        else:
            return self + Dual(other)
```

We can add more operators, such as `exp()` and `sin()` using the familiar derivative rules (in principle, we can derive the rules ourselves using algebra, e.g. equating Euler’s formula for `sin()` and its power series). However,  as long as we define the output of an operation such that we output the value of the function, as well as its derivative multiplied by the derivative value of the input `Dual`, we can compose these operators and everything works as expected. For example, here is `sin()` in python:

```
import math

def sin(dual):
    return Dual(math.sin(dual.value), dual.derivative * math.cos(dual.value))
```

Why do we need to import `math`? It would be nice if we could automatically derive the all our math operators from definitions of `+` and `*` using some given algebraic structure, but these are the kinds of things you begin to wish programming languages did when you play with this stuff. (“Hey, this seems like something Haskell should be able to do!” --bystander walking away muttering about Monads...)
## Types of Automatic Differentiation
### Forward Mode AD
Now, we can perform Forward Mode AD practically right away, using the `Dual` numbers class we’ve already defined.

Now, when we run our function, in order to calculate `x’` for the input to our function, we pass in a `Dual`, with a constructor that sets the `.derivative` data member to 1, in order to correctly calculate `f` and `df/dx` when we call the function. Using our dual numbers class, we can input some values into `super_complicated_function` and we get some answers!

```
def super_complicated_function(x):
    return x * sin(x+6.)**2. / 2. - 2. / 2.**x

# symbolic derivative by running the above in mathematica (wolfram alpha)
def d_super_complicated_function(x):
    return 2**(1 - x) * math.log(2) + 0.5 * sin(x + 6)**2 + x * math.sin(x + 6) * math.cos(x + 6)

for x in [-1., 0.5, 3., 2.]:
    print('Function Input: {}'.format(x))
    print('Function Value: {}'.format(super_complicated_function(x)))
    print('Function Symbolic Derivative: {}'.format(d_super_complicated_function(x)))
    print(super_complicated_function(Dual(x, 1.)))
    print('-'*32)

'''
# Printed Results

Function Input: -1.0
Function Value: -4.459767882269113
Function Symbolic Derivative: 3.504367159953579
< Dual value: -4.459767882269113, derivative: 3.504367159953579 >
--------------------------------
Function Input: 0.5
Function Value: -1.4026444100543694
Function Symbolic Derivative: 1.1084382073126584
< Dual value: -1.4026444100543694, derivative: 1.1084382073126582 >
--------------------------------
Function Input: 3.0
Function Value: 0.004762468816939924
Function Symbolic Derivative: -0.8682732520785479
< Dual value: 0.004762468816939924, derivative: -0.8682732520785477 >
--------------------------------
'''
```

Let’s now draw a graph visualizing our function. Here, the operations that make up our function are shown as nodes, and their arguments are shown in order (top to bottom). We show red numbers to represent intermediate derivatives. For example, for the division operation in the top right, we compute ![](http://www.sciweavers.org/tex2img.php?eq=%280.51%20%2B%20%5Cepsilon%201%29%20%2F%202%20%3D%200.5&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0) for the top input and  ![](http://www.sciweavers.org/tex2img.php?eq=0.51%20%2F%20%282%20%2B%20%5Cepsilon%201%29%20%3D%20-0.13&bc=White&fc=Black&im=png&fs=12&ff=modern&edit=0) for the bottom input (the constant value “2”). In order to calculate the derivative with respect to the output, we start at a node and multiply all the red numbers following the path through the tree to the output. If we have a variable like `x` with multiple paths, we sum the product for each path.



We can define a function that automatically generates the red “local derivatives” for each operation. We run the function in a loop, setting all `.derivative` members to 0, except for the particular input we are interested in:

```    
def create_diff_fn(fn):
        def diff_fn(*argv):
            jacobian = []
            Dual_arguments = [Dual(x, 0.) for x in argv]
            for input_arg in Dual_arguments:
                input_arg.derivative = 1.
                result = fn(*Dual_arguments)
                Jacobian.append(result.derivative)
                input_arg.derivative = 0.

            return jacobian
        
        return diff_fn
```

The function above returns a function that calculates the partial derivative with respect to each input to our function. We note that this requires we run the function multiple times, making our computational complexity pretty poor for functions with many inputs. To solve this problem, we can trade memory for time complexity, by caching intermediate results for single forward pass and exploiting the associativity of multiplication found in the chain rule. This is called “Reverse Mode AD,” and is the mode used for things with many input parameters like Deep Neural Networks.
### Reverse Mode AD


Noting the diagram above we note that our derivatives are updated at each operation node. If we step back and examine the chain rule, noting that derivatives multiply, perhaps we can propagate the gradients along the graph in the other direction? This leads to an algorithm called “backpropagation.” If you’ve trained neural networks before, you probably love backprop. We all love backprop. Backprop loves backprop [https://arxiv.org/pdf/1606.04474.pdf].

The basic idea is we encapsulate each primitive operation using forward and backward functions. Interestingly, using dual numbers we can easily compute a backward function for every forward function (see `create_diff_fn` above). Our plan is to dynamically build a graph every time we perform an operation. In order to capture inputs with each operation, we define a new class `Variable` which overloads our familiar operators. Now, the constructor for our `Variable` class keeps track of lower down operations that created it, so we get a nice tree structure. Each time we perform an operation, we calculate the output `value` (blue numbers in the diagram above), and the local gradient (from `create_diff_fn` in red). Then, when we want the gradient of any node, we can multiply all of the local gradients that connect that node to the output. If a node’s output is used multiple times, as in `x`  above, we add this product for every path to the output. As seen above, this is consistent with the chain rule!

Before we share the code for `Variable` that implements this idea, here is an example of how it might be used:

```
# compose a graph of nodes

x = Variable(10.01)
y = Variable(0.05)

z = x * y
m = 1.3
q = m+y
   
L = (q - (z**(m/2.) + z**2. - 1./z))**2

# the forward calculation
print(L.value)

# the backward calculation
L.backward()

# gradients of the inputs are updated
print('x: {} \ny: {}'.format(x.gradient, y.gradient))

```

Here is all the code you need (excluding most of the overloaded arithmetic operators to stay concise):

```
class Variable:
    def __init__(self, operation, input_variables=[]):
        # note the setter for @property value below
        self.value = operation
        self.input_variables = input_variables
        self.gradient = 0.
```

Our `Variable` class represents an operation (or function), it’s inputs as variables, and a @property representing our logic to calculate the output of the operation. We also make a place to store the value of the gradient.

```     
    def calc_input_values(self):
        # calculate the real-valued input to operation
        return [v.value for v in self.input_variables]
    
    def forward(self):  
        # calculate the real-valued output of operation              
        return self.forward_op(*self.calc_input_values())

    @property
    def value(self):
        return self.forward()
    
    @value.setter
    def value(self, value):
        if callable(value):
            self.forward_op = value
        else:
            self.forward_op = lambda : value
        
        self.derivative_op = create_diff_fn(self.forward_op)
```

The above code uses a @property to make the interface a bit nicer, but actually leads to a bit more code than is strictly necessary. At a high level, we want to create a `forward_op` that calculates the output of our operation, and a `backward_op` that determines the jacobian of our operation. The main tricky piece is `calc_input_variables` which recursively calls `forward()` on the input variables until a constant value input is encountered, from there, `value` can be calculated. We handle the constant valued operations in the `value` property setter.

This completes our bookkeeping for running the operation forward. Below, we include all of the backward, or backprop logic:

```
        
    def backward(self, output_gradient=1.):
        # combine gradients from other paths and propagate to children
        self.gradient += output_gradient
        local_gradient = self.derivative_op(*self.calc_input_values())
        for differential, input_variable in zip(local_gradient, self.input_variables):
            input_variable.backward(differential * output_gradient)
```

And this is an example of defining an operation. We could pass in any function handle as an `operation`, but when overloading basic math functions, we create a lambda.
    
```
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Variable(lambda a,b : a+b, (self, other))
        else:
            return Variable(lambda a,b : a+b, (self, Variable(other)))

```

Clearly, this code is not super efficient. For one, it would be straightforward to cache results when running `forward` so that the function does not need to be called multiple times. But in principle, this is all that’s required to make Reverse AD work.

As we can see, our `super_complicated_function` returns what we expect:

```
for x in [-1., 0.5, 3.]:
    print('Function Input: {}'.format(x))
    print('Function Value: {}'.format(super_complicated_function(x)))
    print('Function Symbolic Derivative: {}'.format(d_super_complicated_function(x)))
    x_v = Variable(x)
    L = super_complicated_function(x_v)
    print('Variable Function Output Value: {}'.format(L))
    L.backward()
    print('Input value: {}'.format(x_v))
    print('-'*32)

'''
# Printed Results

Function Input: -1.0
Function Value: -4.459767882269113
Function Symbolic Derivative: 3.504367159953579
Variable Function Output Value: < Variable value: -4.459767882269113, gradient: 0.0 >
Input value: < Variable value: -1.0, gradient: 3.504367159953579 >
--------------------------------
Function Input: 0.5
Function Value: -1.4026444100543694
Function Symbolic Derivative: 1.1084382073126584
Variable Function Output Value: < Variable value: -1.4026444100543694, gradient: 0.0 >
Input value: < Variable value: 0.5, gradient: 1.1084382073126582 >
--------------------------------
Function Input: 3.0
Function Value: 0.004762468816939924
Function Symbolic Derivative: -0.8682732520785479
Variable Function Output Value: < Variable value: 0.004762468816939924, gradient: 0.0 >
Input value: < Variable value: 3.0, gradient: -0.8682732520785479 >
--------------------------------
'''
```
### Hybrid Mode AD

In our construction so far, if we define a function that processes `Dual` types, we are using Forward Mode AD, and functions that process `Variable` types are using Reverse Mode AD. Our library defines a function, `create_diff_fn`,  that translates between these two kinds of functions. We use this function to “wrap” Forward Mode functions written for `Dual`s so that they can process `Variables`, and incorporate them into the Reverse Mode call graph. In this way, we can run Reverse Mode AD on any function that uses operations defined for `Dual`.

By wrapping pieces of our computational graph as standalone functions where we always compute the backward values using Forward Mode, we can trade memory for time complexity. This trade-off is called “Hybrid Mode” AD, and can be very important for performance, especially when encountering large memory requirements for certain functions when using Reverse Mode AD.

It’s known that finding the optimal trade-off between using forward mode and backward mode AD to balance time and memory complexity is NP-hard. This means that we probably won’t find the most efficient way to compute the derivatives for arbitrary programs. Luckily, there are some good heuristics that actually do a pretty good job. Anyway, our library is definitely not very efficient, and more production grade AD libraries often use smart caching of intermediate results and perform various optimizations using JIT compilation. Regardless, our simple little library can do some really neat things. It gracefully handles control flow, and can differentiate pretty complex programs already.

## Conclusion: Simple Python Library

Since we build a computation graph dynamically, our `Variable` class can handle branching and loops.

```
def forward_fn(x):
    for n in range(5):
        if n % 2 == 0:
            x = 3.*x
        else:
            x = x**(1./n) + 1./n
        
    return x

x = Variable(2.)
y = forward_fn(x)
print(y)

y.backward()
print(x)
```

We’ve only implemented a simple library that handles composing scalar functions and real numbers, but real AD systems can build something very similar using vectors and matrices and the rules of linear algebra.

And as a final proof of concept, let’s solve a minimization problem:

```
def fn(x):
    return x**2 + 0.2*(x-2)**4 + 2*x**3

# initialization
x = Variable(2.)
print('---- Initial Value ----')
print('fn(x): {}'.format(L))
print('x: {}'.format(x))

for n in range(30):
    L = fn(x)
    L.backward()
    # gradient descent update
    x.value = x.value - 0.1 * x.gradient
    # clear the gradients
    x.clear_gradient()
    
print('---- Converged Value ----')    
print('fn(x): {}'.format(L))
print('x: {}'.format(x))

# Wolfram alpha minimum: 
# min{x^2 + 0.2 (x - 2)^4 + 2 x^3}≈1.5110 at x≈0.51489
# Output:
'''
---- Initial Value ----
fn(x): < Variable value: 1.5110101392696607, gradient: 1.0 >
x: < Variable value: 2.0, gradient: 0.0 >
---- Converged Value ----
fn(x): < Variable value: 1.5110101392696607, gradient: 1.0 >
x: < Variable value: 0.5149364184363174, gradient: 0.0 >
'''
```

This post hopefully provides you with a good intuition for using AD in real problems. By carefully constructing types and overloading the right operators, we end up with a rather elegant way to differentiate computer programs. The general principle of abstraction via function overload can be applied to other “morphisms”, such as those used in Homomorphic Encryption, making AD actually a neat way to gain intuition about these other feats of modern computer science. Furthermore, while I love python, it’s generic programming capabilities aren’t the most flexible, and a better language for real applications is probably writing a C++ Template library like what’s used by [Enoki](https://enoki.readthedocs.io/en/master/demo.html). A notable python library for AD that I’ve had success with is JAX. Other well known frameworks that use auto-differentiation include Tensorflow and PyTorch, where for efficiency the AD code is implemented at a low level.

Anyway, I hope AD is now not so mysterious to you, but is perhaps even more magical. :)


The code can be found on github.


