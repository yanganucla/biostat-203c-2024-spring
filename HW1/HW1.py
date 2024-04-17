# PIC 16A HW1
# Name:Yang An
# Collaborators: None
# Date:04/16/2024

import random # This is only needed in Problem 5

# Problem 1

s = """Tired    : Doing math on your calculator.
Wired    : Doing math in Python.
Inspired : Training literal pythons to carry out long division using an abacus."""

def print_s(s):
    print(s)
# print_s(s)

# you do not have to add docstrings for the rest of these print_s_* functions.

def print_s_lines(s):
    parts = s.split('\n')
    for part in parts:
        # Split the line at the colon ':', and strip to remove any leading/trailing whitespaces
        title, description = part.split(':')
        print(title.strip())     # Print the title (e.g., "Tired")
        print(description.strip())  # Print the description on a new line
# print_s_lines(s)

def print_s_parts(s):
    lines = s.split('\n')
    for line in lines:
        print(line.split()[0])
# print_s_parts(s)

def print_s_some(s):
    lines = s.split('\n')
    max_length = max(len(line) for line in lines)
    for line in lines:
        if len(line) < max_length:
            print(line)
# print_s_some(s)

def print_s_change(s):
    modified_s = s.replace('math', 'data science').replace('long division', 'machine learning')
    print(modified_s)
# print_s_change(s)

# Problem 2 

def make_count_dictionary(L):
    """
    Takes a list L and returns a dictionary D where the keys of D are the unique elements of L.
    Each key's value is the number of times that element appears in L.

    Args:
    L: List of elements (can be heterogeneous with strings, integers, etc.)

    Returns:
    dict: Dictionary with elements of L as keys and their count in L as values.
    """
    D = {}
    for item in L:
        if item in D:
            D[item] += 1
        else:
            D[item] = 1
    return D

# L = ["a", "a", "b", "c"]
# print(make_count_dictionary(L))


# Problem 3

def gimme_an_odd_number():
    """
    Prompts the user to enter integers repeatedly until they enter an odd integer.
    Once an odd integer is entered, the function prints and returns a list of all integers entered.
    """
    numbers = []  # List to store all entered numbers
    while True:  # Loop indefinitely
        x = input("Please enter an integer.")  # Prompt user for input
        if x.isdigit() or (x.startswith('-') and x[1:].isdigit()):  # Check if the input is an integer
            num = int(x)  # Convert the input to an integer
            numbers.append(num)  # Append the number to the list
            if num % 2 != 0:  # Check if the number is odd
                print(numbers)  # Print the list of numbers
                return numbers  # Return the list of numbers

# print(gimme_an_odd_number())  

# Problem 4

def get_triangular_numbers(k):
    """Returns a list of the first k triangular numbers."""
    return [sum(range(1, i + 1)) for i in range(1, k + 1)]
# k = 6
# print(get_triangular_numbers(k))


def get_consonants(s):
    """Returns a list of consonants in the string s, excluding vowels, spaces, commas, and periods."""
    return [char for char in s if char.lower() not in "aeiou ,."]
# s = "make it so, number one"
# print(get_consonants(s))  


def get_list_of_powers(X, k):
    """Returns a list of lists, each containing powers of elements in X from 0 to k."""
    return [[x**i for i in range(k + 1)] for x in X]
# X = [5, 6, 7]
# k = 2
# print(get_list_of_powers(X, k))


def get_list_of_even_powers(X, k):
    """Returns a list of lists, each containing even powers of elements in X from 0 to k."""
    return [[x**i for i in range(k + 1) if i % 2 == 0] for x in X]
# X = [5, 6, 7]
# k = 8
# print(get_list_of_even_powers(X, k))  



# Problem 5

def random_walk(ub, lb):
    """
    Simulates a simple random walk with specified upper and lower bounds.
    
    Parameters:
        ub (int): The upper boundary of the walk.
        lb (int): The lower boundary of the walk.
    
    Returns:
        pos (int): The final position of the walk at termination.
        positions (list of int): Log of the position at each time step, initial position included, final excluded.
        steps (list of int): Log of steps taken, represented by 1 (forward) and -1 (backward).
    """
    pos = 0  # Initial position
    positions = [pos]  # List to store the log of positions
    steps = []  # List to store the log of steps

    while True:
        step = random.choice([1, -1])  # Simulate the coin flip
        steps.append(step)
        pos += step  # Update the position based on the coin flip
        if pos == ub:
            print(f"Upper bound at {pos} reached")
            break
        elif pos == lb:
            print(f"Lower bound at {pos} reached")
            break
        positions.append(pos)  # Log the position after taking the step

    return pos, positions[:-1], steps  # Exclude the final position from the log

# Example of how to use the function
# if __name__ == "__main__":
   # ub = 10
   # lb = -10
   # final_pos, all_positions, all_steps = random_walk(ub, lb)
   # print("Final Position:", final_pos)
   # print("Positions Log:", all_positions)
   # print("Steps Log:", all_steps)

# if __name__ == "__main__":
#     pos, positions, steps = random_walk(5000, -5000)
#     plt.figure(figsize=(12, 8))
#     plt.plot(positions)
#     plt.xlabel('Timestep')
#     plt.ylabel('Position')
#     plt.title('Random Walk')
#     plt.show()

# If you uncomment these two lines, you can run 
# the gimme_an_odd_number() function by
# running this script on your IDE or terminal. 
# Of course you can run the function in notebook as well. 
# Make sure this stays commented when you submit
# your code.
#
# if __name__ == "__main__":
#     gimme_an_odd_number()