#Lists

#Creating empty lists
el = []

#Creating unempty lists
car_brands = ['Toyota', 'Honda', 'BMW', 'Ford']
print(car_brands)

#Accessing Elements
first_brand = car_brands[0]
last_brand = car_brands[-1]
print(first_brand, last_brand)

#Slicing
subset = car_brands[1:3]
print(subset)

#Length
length = len(car_brands)
print(length)

#Concatenation
more_brands = ['Mercedes', 'Audi']
combined = car_brands + more_brands
print(combined)

#Modifying Elements
car_brands[1] = 'Chevrolet'
print(car_brands)

#Appending Elements
car_brands.append('Nissan')
print(car_brands)

#Inserting Elements
car_brands.insert(2, 'Tesla')
print(car_brands)

#Removing Elements - Remove
car_brands.remove('BMW')
print(car_brands)

#Removing Elements - Pop
#Note that this changes the "car_brands" list too
popped_brand = car_brands.pop(1)
print(popped_brand, car_brands)

#Sorting
car_brands.sort()
print(car_brands)

#Reversing
car_brands.reverse()
print(car_brands)

#Counting
count = car_brands.count('Toyota')
print("List Counting:", count)

#List comprehensions
#Make lists using iteration and generation at same time / in one line

#Syntax: [expression for item in iterable if condition]
#Expression: Computed expression to be included in the final list
#Item: What will be processed
#Iterable: Object or sequence to loop over
#If condition (Optional): Conditions under which to apply the expression

squares = [x ** 2 for x in range(1, 9)]
print(squares)
squares_lt3 = [x ** 2 for x in range(1, 9) if x < 3]
print(squares_lt3)

#Collect only even numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [n for n in numbers if n % 2 == 0]
print(even_numbers)
odd_numbers = [n for n in numbers if n % 2 == 1]
print(odd_numbers)

#Create a tuple from x, x**2, and x**3
numbers = [1, 2, 3, 4, 5]
numbers_squared_cubed = [(num, num ** 2, num ** 3) for num in numbers]
print(numbers_squared_cubed)

#Create a dictionary from x and x**2 using "dictionary comprehensions"
numbers = [1, 2, 3, 4, 5]
x = [1, 2, 3, 4, 5]
d = {num: num ** 2 for num in x} #Note the braces
print(d)

#Capitalize all the words and put them in a new list 
words = ['alpha', 'beta', 'gamma']
uppercase_words = [word.upper() for word in words]
print(uppercase_words)