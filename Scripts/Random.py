#Random
import random 

#Set seed for reproducing randomness
random.seed(1663)

#Generate a random integer between two values
random.randint(1,100)

#Select an item from a list at random
languages = ['python','r','swift','javascript','julia']
random.choice(languages)

#Select k items from a list at random
languages = ['python','r','swift','javascript','julia']
random.choices(languages, k=3)

#Select items from a list at random, with weights
languages = ['python','r','swift','javascript','julia']
random.choices(languages, weights=[5,3,1,1,1], k=3)

#Randomly reorder a list
languages = ['python','r','swift','javascript','julia']
random.shuffle(languages)
languages

#Select a sample of k length
languages = ['python','r','swift','javascript','julia']
random.sample(languages, k=3)

#Random float between 0 and 1
random.random()

#Random floats between two integers
random.uniform(0,100)

#Random floats normal distribution given mean and standard deviation
random.normalvariate(100,25)