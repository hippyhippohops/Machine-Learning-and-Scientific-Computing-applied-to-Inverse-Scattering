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

# Mathematical Operations on Strings

first_str = "throat"
println(first_str)
second_str = "warbler"
println(second_str)
concantenation_of_string = first_str * second_str
println(concantenation_of_string)
repetition_of_string = first_str^5
println(repetition_of_string)

""" Chapter 3 """

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
#println("What...is your name?"); readline() #Work this out!!

""" Chapter 7 """

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

""" Chapter 15 """

# We will create a new type to represent points as objects
struct Point
    x
    y
end

# Constructor
p = Point(3.0,4.0) 
println(p)

# Mutable Structs 
mutable struct MPoint
    x 
    y
end
blank = MPoint(0.0,0.0)
println(blank)
blank.x = 3.0
blank.y = 4.0
println(blank)

"""
Represents a rectangle

fields: Width, Height, Corner
"""
struct Rectangle
    width 
    height
    corner
end
origin = MPoint(0.0, 0.0)
box = Rectangle(100.0,200.0, origin)
println(box)

# Instances as Arguments - you can pass an instance as an argument in the usual way
function printpoint(p)
    println("($(p.x), $(p.y))")
end
printpoint(blank)

function movepoint!(p, dx, dy)
    p.x += dx
    p.y += dy
    nothing
end
origin = MPoint(0.0, 0.0)
movepoint!(origin, 1.0, 2.0)
println(origin)
