""" Chapter 1 """

# The First Program in page 3
println("Hello World!") 

# Arithmetic Operations in page 3
println(40+2) #Addition 
println(43-1) #Subtraction
println(6*7) #Multiplication
println(84/2) #Division
println(6^2+6) #Exponentiation 

#Checking the type of a value -> use typeof()
println(typeof(2)) #Int64
println(typeof("2")) #String
println(typeof(42.0)) #Float64
println(typeof("42.0")) #String
println(typeof("Hello World")) #String

""" Chapter 2  """

# Script Mode 
miles = 26.2
println("Miles is: ", miles, " in miles.") 
miles = 26.2 * 1.61
println("In kilometers, it is: ", miles)

# Mathematical Operations on Strings

first_str = "throat"
println("First string is: ", first_str)
second_str = "warbler"
println("First second is: ", second_str)
concantenation_of_string = first_str * second_str
println("Concantenation of strings is: ", concantenation_of_string)
repetition_of_string = first_str^5
println("Repetition of Strings is: ", repetition_of_string)


""" Chapter 3 """

#=
In the context of programming, a function is a named sequence of statements that performs a computation. When you define a 
function, you specify the name and the sequence of statements. Later, you call the function by it's name. 

The expression in the parenthesis of the function is called the argument of the function. It is common to say that a function 
takes in an argument and returns a result. The result is also called the return value. 


There are 2 types of functions:
(i) Inbulit Functions
(ii) New Functions
=#

# parse takes a string and converts it to any number type
println(parse(Int64, "32"))
println(parse(Float64, "3.14159"))

# trunc converts floating-point values to integers; it chops off the fraction part
println(trunc(Int64, 3.9999))
println(trunc(Int64, -2.3))

# float converts integers to floating-point numbers
println(typeof(32))
println(float(32))
println(typeof(float(32)))

# String converts its arguments to a string
println(typeof(32))
println(string(32))
println(typeof(string(32)))

println(typeof(3.14159))
println(string(3.14159))
println(typeof(string(3.14159)))

# Adding new functions

function printlyrics()
    println("I'm a lumberjack, and I'm okay.")
    println("I sleep all night and I work all day")
end

function repeatlyrics()
    printlyrics()
    printlyrics()
end

repeatlyrics()

# Parameters and Arguments 
function printtwice(bruce)
    println(bruce)
    println(bruce)
end
printtwice("Spam")
printtwice(42)

# We can also use variables as an argument
michael = "Eric, the half a bee"
printtwice(michael)


""" Chapter 5 """

#=
The main topic of this chapter is the if statement, which executes different code depending on the status of the program. 
=#

# Floor Division Operator - this operation divides 2 numbers and rounds down to an integer
minutes = 105
println(minutes) # returns 105
println(minutes/60) # return 1.75
println(minutes÷60) # returns 1

# Modulus Operator - this operator divides 2 numbers and returns the remainder
remainder = minutes % 60 

# Boolean Expression - An Boolean expression is an expression that is either true or false
println(5==5)
println(5==6)

#Relational operators
println(5 != 3) # 5 is not equal to 3
println(5 > 3) # 5 is greater than 3
println(3 < 5) # 3 is less than 5
println(5 >= 3) # 5 is greater than or equal to 3
println(3 <= 5) # 3 is less than or equal to 5

# Logical operators

#=

&& -> AND 
|| -> OR
! -> NOT
=#

println(5 > 3 && 5 == 5) # And - returns true as both LHS and RHS are true
println(5 > 3 && 5 != 5) # And - returns false as while LHS is true and RHS is false
println(5 < 3 && 5 != 5) # And - returns false as both LHS and RHS are false
println(5 > 3 || 5 == 5) # Or - returns true as both LHS and RHS are true
println(5 > 3 || 5 != 5) # Or - returns true as while LHS is true and RHS is false
println(5 < 3 || 5 != 5) # Or - returns false as both LHS and RHS are false

# Conditional Execution - Gives us the ability to check the conditions and change the behaviour of the program accordingly
x = 3
if x>0
    println("x is positive")
end

# Alternative Execution - In this style of execution, there are 2 possibilities and the condition determines which one to run
if x%2 == 0
    println("x is even")
else
    println("x is odd")
end


# Chained Conditional - When there are more than 2 possibilities
y = 3
if x < y 
    println("x is less than y")
elseif x > y
    println("x is greater than y")
else
    println("x is equal to y")
end

# Nested Conditional - One conditional can be nested within another
y = 0
if x == y
    println("x is equal to y")
else
    if x < y
        println("x is less than y")
    else
        println("x is greater than y")
    end
end

# Recursion - it is legal for one function to call itself
function countdown(n)
    if n <= 0 
        println("Blastoff!")
    else
        print(n, " ")
        countdown(n-1)
    end
end
countdown(5)

# Keyboard Input

#=
In Julia, we need to print the prompt before calling readline(). The following line will give an error
readline("What...is your name?")
Hence, we need to define a prompt and ask for an input the following way:
=#
#println("What...is your name?"); readline() #WORK THIS OUT!!

""" Chapter 7 """

#= 
This chapter is about iteration, which is the ability to run a block of statements repeatedly. Recursion is a type of
iteration. A For loop is another type of iteration. Here, we will see a while statement. 
=#

#=
You have to first initialise a variable before you can update it. Updating a variable by adding 1 is called an increment; 
substracting 1 is called a decrement
=#
y = 0
println("y is: ", y)
y = y + 1
println("After incrementing, y is: ", y)
x = 5
println("x is: ", x)
x = x - 1
println("After incrementing, x is: ", x)

# While statement - Here is a version of countdown that uses a while statement
function countdown_2(n)
    while n > 0
        print(n, " ")
        n = n - 1
    end
    println("Blast off !")
end
countdown_2(5)

# Break statement - A break statement can be used to jump out of the loop

#=
while true 
    print("> ")
    line = readline()
    if line == "done"
        break
    end
    println(line)
end
println("Done!")
=# #Work this out!!!

""" Chapter 8 """

# A string is a sequence, which means it is an ordered collection of other values

# You can access the characters one at a time with the bracket operator ([]):
fruit = "banana"
println("The string is: ", fruit)
letter = fruit[1]
println("The first element of our string fruit, letter, is: ", letter)
println("The last element of our string fruit, letter, is: ", fruit[end])

# Length is an inbuilt function that returns the number of characters in a string
len = length(fruit)
println("The length of our string fruit is: ", len)

#= Traversal of string - First method - While loop
index = firstindex(fruit)
while index <= len && index >= 0
    letter = fruit[index]
    println("The selected letter is: ", letter)
    global index = index - 1
end
WORK THROUGH THIS=#

# Traversal of string - First method - For loop
for letter in fruit
    println("The selected letter is: ", letter)
end

# String splicing
str = "Julius Caesar"
println("The string is: ", str)
println("The string, after splicing first 6 elements, is: ", str[1:6])
println("The string, after splicing from 8th element till the end, is: ", str[8:end])

""" Chapter 11 """

println("DICTIONARIES")

#=
Another built-in type is called a dictionary. A dictionary is like an array, but more general. In an array, the indices
have to be integers; in a dictionary, they can be almost any type. 

A dictionary contains a collection of indices, which are called keys, and a collection of values. Each key is associated 
with a single value. The association of a key and a value is called a key-value pair, or sometimes an item. 

In mathematical language, a dictionary represents a mapping from keys to values, so you can also say that each key "maps to"
a value. 
=#

# Creating a new dictionary
eng2sp = Dict()
println("The new dictionary, named, eng2sp is: ", eng2sp)

# To add items to the dictionary
eng2sp["one"] = "uno"
println("After adding a key-value pair, eng2sp is: ", eng2sp)
eng2sp["two"] = "dos"
println("After adding a key-value pair, eng2sp is: ", eng2sp)
eng2sp["three"] = "tres"
println("After adding a key-value pair, eng2sp is: ", eng2sp)
println("The value associated to key two is: ", eng2sp["two"])

#= You can transverse the keys of a dictionary in a for statement
function printhist(h)
    for c in keys(h)
        println(c, " ", h[c])
    end
end
h = histogram("parrot")
println("h is: ", h)
printhist(h)
=#

""" Tuples """

println("TUPLES")






""" Chapter 10 """

println("ARRAYS")

#=
We will explore one of Julia's bulit-in types: array

Like a string, an array is a sequence of values. In a string, the values are characters. In an array, they can be 
any type. The elements in an array are called elements or sometimes items. 
=#

# Simplest way to create a new array: Enclosing the elements in square brackets []
array_1 = [10,20,30,40]
println("First array is: ", array_1) 
println(typeof(array_1)) # Array of 4 integers
array_2 = ["cruchy frog", "Ram Bladder", "Lark Vomit"]
println("First array is: ", array_2) 
println(typeof(array_2)) # Array of 3 stringer
array_3 = ["spam", 2.0, 5,  [10,20]] 
println("Third array is: ", array_3) 
println(typeof(array_3)) # Array of 3 stringer# Array of a string, a float, an integer, and another array

# Arrays are mutable 

#=
The syntax for accessing the elements of an array is the same as for accessing the characters of a string - using the 
bracket operator.
=#
cheeses = ["Cheddar", "Edam", "Gouda"]
println("The array cheeses is: ", cheeses)
println("The first element of the array cheeses is: ", cheeses[1])

numbers = [42, 123]
println("The array numbers is: ", numbers)
println("The second element of the array numbers is: ", numbers[2])
# Here is how we can change an element of numbers
numbers[2] = 5
println("After reassignment, the array numbers is: ", numbers)

# Transversing an Array
# The most common way to transverse the elements of an array is with a for loop:
for cheese in cheeses
    println(cheese)
end

# eachindex() -> Creates an iterable object for visiting each index of an AbstractArray A in an efficient manner
println(eachindex(cheeses))
# Transvering with eachindex()
for i in eachindex(numbers)
    numbers[i] = numbers[i] * 2
end
println("Numbers, after transvering, is: ", numbers)

# Array Slices

t = ['a', 'b', 'c', 'd', 'e', 'f']
println("The new array, t, is: ", t)
println("The first 3 elements of the array, t, is: ", t[1:3])
println("The remaining elements of the array, t, is: ", t[3:end])

# A slice operator on the left side of an assignment can update multiple elements
t[2:3] = ['x','y']
println("The array, t, is: ", t)

# push! adds a new element to the end of an array 
t = ['a', 'b', 'c']
println("The new array, t, is: ", t)
push!(t, 'd')
println("The array, t, after pushing is: ", t)

# append! adds the elements of the second array to the end of the first
t1 = ['a', 'b', 'c']
println("The array, t1, is: ", t1)
t2 = ['d', 'e', 'f']
println("The array, t2, is: ", t2)
append!(t1,t2)
println("The array, t1, after appending is: ", t1)

# sort! arranges the elements of the array from low to high 
t = ['d', 'c', 'e', 'b', 'a']
println("The new array, t, is: ", t)
sort!(t)
println("The sorted array, t, is: ", t)

# sum() -> adds all the elements in the array 
t = [1,2,3,4]
println("The new array, t, is: ", t)
println("The sum of array, t, is: ", sum(t))

# Dot syntax
t = [1,2,3,4]
println("The new array, t, is: ", t)
println("The dot syntax of array, t, is: ", t .^ 3)

# Splice -> If you know the index of the element you want, you can use Splice
t = [1,2,3,4]
println("The new array, t, is: ", t)
splice!(t, 2)
println("The array, t1, after splicing is: ", t)

# pop! deletes and returns the last element
t = [1,2,3,4]
println("The new array, t, is: ", t)
pop!(t)
println("The array, t1, after popping is: ", t)


""" Chapter 15 """

#= 
Thus far, we have seen how to use functions to organise code and how to use bulit in 
types to organise data. 
Now, we will learn how to build our own types to organise both code and data.

Now, we are going to define a type. As an example, we will create a type called Point
that will represent a point in two-dimensional space. 

We could do the following:
(i) We could store the coordinates separately in two variables. 
(ii) We could store the coordinates as elements in an array or tuple
(iii) We could create a new type to represent points as objects

Creating a new type is more complicated than the other options, but it has advantages 
that will soon be apparent. 
=#

# A user defined composite type is also called a struct. 
#=
We have constructed a struct (user defined composite type) called Point with 2 attributes/fields: x and y. 
A struct is similar to a class/object in OOP.
=#
struct Point
    x
    y
end

#= 
Tocreate a point, you call Point as if it were a function having arguments the values of fields. 
When a struct is used as a function, like below, it is called a Constructor:
=#
p = Point(3.0,4.0) 
#=
Creating a new object is called instantiation, and the object of an instance is of the type.
When you print an instance, Julia tells you what type it belongs to and what the values of the attributes are.
For our p, which we defined as above, prints to give us:
=#
println("Our struct, p, is: ", p) # The return value is a reference to a Pointobject, which we assign to p. 

# Getting values of Structs
println("The x attribute of p is: ", p.x)
println("The y attribute of p is: ", p.y)

# We can use the dot notation as part of any expression.
distance = sqrt(p.x^2 + p.y^2)
println("The distance of point from the origin is: ", distance)

# Mutable Structs 
#=
Structs are immutable. After the construction of a struct, the fields cannot change value. 
When required, mutable composite typees can be declared with the keyword mutable struct, as seen
below:
=# 
mutable struct MPoint
    x 
    y
end
blank = MPoint(0.0,0.0)
println("Our mutable struct, blank, is: ", blank)
println("The x attribute of blank is: ", blank.x)
println("The y attribute of blank is: ", blank.y)
blank.x = 3.0
blank.y = 4.0
println("Our mutable struct, blank, is now: ", blank)
println("The x attribute of blank is now: ", blank.x)
println("The y attribute of blank is now: ", blank.y)


"""
We will construct another example of a struct, a rectangle. 
This struct represents a rectangle and will be described by 
the following fields: Width, Height, Corner. 
"""
struct Rectangle
    width 
    height
    corner
end
origin = MPoint(0.0, 0.0)
println("The point of origin of the rectangle, ie the corner, is: ", origin)
box = Rectangle(100.0,200.0, origin)
println("The rectangle (with corner (0,0), width 100 and height 200) is: ",box)

# Instances as Arguments - you can pass an instance as an argument in the usual way
#=
Recall that instances are objects that are copies of our class/struct. Therefore, this
tells us that we can pass our instances as arguments in our functions. 

Recall that here, p = Point(3.0,4.0), an instance of our struct/class called point.
We are defining a function called printpoint that takes in an instance of our struct/class 
called point as an argument, and displays it in the conventional mathematical notation.
=#
function printpoint(p)
    println("The mathematical notion of our struct called point ,p, is: ","($(p.x), $(p.y))")
end
printpoint(blank)

# To define the distance function above as a function
function distance_new(p)
    println("The new distance of point from the origin is: ", sqrt(p.x^2 + p.y^2))
end
new_point = Point(7.5,8.3)
distance_new(new_point)

#= 
If a mutable struct is passed as an argument to a function as an argument, the function 
can modify the fields of the object.

Here, movepoint! takes a mutable Point object and 2 numbers, dx and dy, and adds the numbers 
to x and y attributes respectively. 

=#
function movepoint!(p, dx, dy)
    p.x += dx
    p.y += dy
    nothing
end
origin = MPoint(0.0, 0.0)
println("The coordinates of the original point is: ", origin)
movepoint!(origin, 1.0, 2.0)
println("After moving the point, the coordinates of the point is: ", origin)

#=
Not only can functions take in instances as arguments, they can also return instances. 

In the example below, findcenter takes a rectangle as an argument and returns a Point that 
contains the coordinates of the center of the rectangle, which is an object/instance of class
Point:
=#
function findcenter(rect)
    Point(rect.corner.x + rect.width / 2, rect.corner.y + rect.height / 2)
end
center = findcenter(box) # We want to find the center of the rect called box
println("The center of the rect called box is:", center)

""" Chapter 16: Structs and Functions """

#= 
Now, that we know how to create new composite types, the next step is to write functions 
that can take programmer-defined objects (ie structs) and return them as results. 

Here, we will also explore functional programming style and 2 new program development plans.
=#

# As another example, we will define a struct called MyTime, which records the time of the day:
struct MyTime
    hour
    minute 
    second
end

# We will create a new MyTime object called:
time = MyTime(11,59,30)
println("The MyTime object called time is: ", time)

#= 
Now, we will write two functions that add time values. They demonstrate 2 kinds of functions:
(i) Pure Functions
(ii) Modifiers

They also demonstrate a development plan called prototype and patch, which is a way of tackling
a complex problem by starting with a simple prototype and incrementally dealing with complications. 
=#

# Here is a simple prototype of addtime:
#=
This function creates a new MyTime object, initialises its fields, and returns a reference to the
new object.

This is called a Pure Function as 
(i) it does not modify any of the objects passed to it as arguments 
(ii) it has no effect, like displaying a value or getting user input, other than returning a value. 
=#
function addtime(t1, t2)
    MyTime(t1.hour + t2.hour, t1.minute + t2.minute, t1.second + t2.second )
end

# Exercise 16-1
function printtime(t1)
    x = t1.hour
    y = t1.minute
    z = t1.second
    println("Time printed: ", x:y:z)
end
example_of_time = MyTime(11, 59, 30)
printtime(example_of_time) # THIS IS INCORRECT CHECK WHY


# To test the function 
start = MyTime(9, 45, 0)
duration = MyTime(1, 35, 0)
done = addtime(start,duration)
println("The movie ends at: ", done)
printtime(done) # THIS IS INCORRECT CHECK WHY

#=
Note that we have 80s. So, we need an improved version, where whenever the min or sec > 60, we need to carry over
the extra time
=#

function addtime(t1, t2)
    second = t1.second + t2.second
    minute = t1.minute + t2.minute
    hour = t1.hour + t2.hour
    if second >= 60
        second -= 60
        minute += 1
    end
    if minute >= 60
        minute -= 60
        hour += 1
    end
    MyTime(hour, minute, second)
end
done = addtime(start,duration)
println("The movie ends at: ", done)

#= 
Sometimes, it is useful for a function to modify the objects it gets as parameters. In that case, the changes
are visible to the caller. Functions that work this way are called modifiers.

An example is the following function:
=#

function increment!(time,seconds)
    time.second += seconds
    if time.second >= 60
        time.second -= 60
        time.minute += 1
    end
    if time.minute >= 60
        time.minute -= 60
        time.hour += 1
    end
end
# However, it may not necessarily be enough to carry over once. We have to keep doing it till seconds < 60

function increment!(time,seconds)
    time.second += seconds
    while time.second >= 60
        time.second -= 60
        time.minute += 1
    end
    while time.minute >= 60
        time.minute -= 60
        time.hour += 1
    end
end

""" Chapter 17 """

#= 
In Julia, you have the ability to write code that can operate on different types. This is called "generic programming".
Here, a method that offer ways to implement different behaviour for a function depending on the types of their arguments
will be introduced. This is called multiple dispatch. 
=#

# :: operator attaches type annotations to expressions and variables, indicating what types they should have
(1+2) :: Int64
# This helps us check if my program works the way we expect it too

# The :: operator can also be appended to the lefthand side of an assignment, or included part of a declaration:
function returnfloat()
    x::Float64 = 100
    x
end
x = returnfloat()
println("x is: ", x)
println("The type of x is: ", typeof(x))

# A type annotation can also be attached to the header of a function definition
function sinc(x)::Float64
    if x == 0
        return 1
    end
    sin(x)/x
end
# In this case, the return value of sinc is always converted to type Float64
println("The output function of sinc(x), when x = 0.5, is: ", sinc(0.5))


using Printf

# Type declarations can be added to the fields in a struct definition
struct MyTime
    hour :: Int64
    minute :: Int64
    second :: Int64
end

function printtime_2(time)
    @printf("%02d:%02d:%02d", time.hour, time.minute, time.second)
end

start = MyTime(9,45,0)
printtime_2(start)

#=
To add a method to the function printtime2 that only accepts a MyTime object as an argument, all we have to do
is append :: followed by MyTime to the argument time in the function definition
=#
function printtime_2(time::MyTime)
    @printf("%02d:%02d:%02d", time.hour, time.minute, time.second)
end 

# A method is a function with a specific signature: printtime2 has one argument of type MyTime

function printtime_2(time)
    println("I don't know how to print the argument time.")
end 

printtime_2(150)

# A constructor is a special function that is called to create an object
