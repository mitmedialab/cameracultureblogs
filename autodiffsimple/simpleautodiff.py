#simpleautodiff - example autodiff for forward and reverse modes in python
#Copyright (C) 2020 Tristan Swedish

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>. 

# needed for sin/cos/log
# see https://github.com/mitmedialab/cameracultureblogs/blob/master/autodiffsimple/autodiffsimple.md
import math

class Dual:
    def __init__(self, value=0., derivative=0.):
        self.value = value
        self.derivative = derivative
        
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Dual(self.value + other.value, self.derivative + other.derivative)
        else:
            return self + Dual(other)
        
    __radd__ = __add__
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return other + -self
    
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return Dual(self.value * other.value, self.value*other.derivative + self.derivative * other.value)
        else:
            return self * Dual(other)
        
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        return self * other**(-1.)
    
    def __rtruediv__(self, other):
        return other * self**(-1.)
    
    def __pow__(self, other):
        if isinstance(other, self.__class__):
            if self.value > 0.:
                return Dual(self.value**other.value, self.derivative * other.value * self.value**(other.value-1) + other.derivative * self.value**other.value * math.log(self.value))
            else:
                return Dual(self.value**other.value, self.derivative * other.value * self.value**(other.value-1))
        else:
            return self**Dual(other)
        
    def __rpow__(self, other):
        if isinstance(other, self.__class__):
            return other**self
        else:
            return Dual(other)**self
        
    def __neg__(self):
        return -1. * self
    
    def __repr__(self):
        return '< Dual value: {}, derivative: {} >'.format(self.value, self.derivative)
    
    @staticmethod
    def create_diff_fn(fn):
        def diff_fn(*argv):
            # return a list of functions for each argument
            Jacobian = []
            Dual_arguments = [Dual(x) for x in argv]
            for input_arg in Dual_arguments:
                input_arg.derivative = 1.
                result = fn(*Dual_arguments)
                Jacobian.append(result.derivative)
                input_arg.derivative = 0.

            return Jacobian
        
        return diff_fn
    
    
class Variable:
    def __init__(self, operation, input_variables=[]):
        self.value = operation
        self.input_variables = input_variables
        self.gradient = 0.
        
    def calc_input_values(self):
        return [v.value for v in self.input_variables]
    
    def forward(self):                
        return self.forward_op(*self.calc_input_values())
        
    def backward(self, output_gradient=1.):
        self.gradient += output_gradient
        local_gradient = self.derivative_op(*self.calc_input_values())
        for differential, input_variable in zip(local_gradient, self.input_variables):
            input_variable.backward(differential * output_gradient)
            
    def clear_gradient(self):
        self.gradient = 0.
        for input_variable in self.input_variables:
            input_variable.clear_gradient()
    
    @property
    def value(self):
        return self.forward()
    
    @value.setter
    def value(self, value):
        if callable(value):
            self.forward_op = value
            
        else:
            self.forward_op = lambda : value
        
        self.derivative_op = Dual.create_diff_fn(self.forward_op)
    
    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self.value > other.value
        else:
            return self.value > other
        
    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.value < other.value
        else:
            return self.value < other
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.forward() == other.forward()
        else:
            return self.forward() == other
        
    def __le__(self, other):
        return self == other or self < other
    
    def __ge__(self, other):
        return self == other or self > other
            
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Variable(lambda a,b : a+b, (self, other))
        else:
            return Variable(lambda a,b : a+b, (self, Variable(other)))
        
    __radd__ = __add__
    
    def __sub__(self, other):
        return self + -other
        
    def __rsub__(self, other):
        return other + -self
        
    
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return Variable(lambda a,b : a*b, (self, other))
        else:
            return Variable(lambda a,b : a*b, (self, Variable(other)))
    
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        return self * other**(-1.)
        
    def __rtruediv__(self, other):
        return other * self**(-1.)
    
    def __pow__(self, other):
        if isinstance(other, self.__class__):
            return Variable(lambda a,b : a**b, (self, other))
        else:
            return Variable(lambda a,b : a**b, (self, Variable(other)))
        
    def __rpow__(self, other):
        if isinstance(other, self.__class__):
            return Variable(lambda a,b : a**b, (other, self))
        else:
            return Variable(lambda a,b : a**b, (Variable(other), self))
        
    def __neg__(self):
        return -1.*self
    
    def __repr__(self):
        return '< Variable value: {}, gradient: {} >'.format(self.value, self.gradient)
    
        
def sin(x):
    if isinstance(x, Dual):
        return Dual(math.sin(x.value), x.derivative * math.cos(x.value))
    if isinstance(x, Variable):
        return Variable(lambda a : sin(a), [x])
    else:
        return math.sin(x)
