#Strings
#In coding, text is often referred to as a string 
#Python has lots of useful and easy methods for manipulating text strings

#Concatenation
str1 = "Hello"
str2 = "World"
string = str1 + " " + str2
print(string)

#String Length
length = len(string)
print(length)

#Slicing Strings
string = "Hello World"

#Print the first letter of the string (indexed to zero)
c1 = string[0]
print(c1)

#Print the second letter in the index (the third in the string)
c3 = string[2]
print(c3)

#Print the 10th letter in the index (the 9th in the string)
c10 = string[10]
print(c10)

#Print the third from the last letter in the string
cminus3 = string[-3]
print(cminus3)

#Print the 3rd letter through the 9th letter
substring = string[2:10] 
print(substring)

#Capitalization (or not)
uppercase = string.upper()
lowercase = string.lower()
print(uppercase, lowercase)

#Splitting / Delimiters
split = string.split(" ") #Space as a delimiter
print(split)

#Formatting
name = "Robert"
age = "36"

formatted_string = "My name is {} and I'm {} years old.".format(name, age)

print(formatted_string)

#F String Interpolation
name = "Robert"
age = "36"

print(f"My name is {name} and I'm {age} years old")

#Replacing
string = "Python is way too hard.  It's so hard I can't believe it!"
new_string = string.replace("hard", "easy")
new_string

#Counting
count = string.count("a")
print(count)
count2 = string.count("lie")
print(count2)