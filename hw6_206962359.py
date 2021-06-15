# Skeleton file for HW6 - Spring 2021 - extended intro to CS

# Add your implementation to this file

# You may add other utility functions to this file,
# but you may NOT change the signature of the existing ones.

# Change the name of the file to include your ID number (hw6_ID.py).

# Enter all IDs of participating students as strings, separated by commas.

# For example: SUBMISSION_IDS = ["123456", "987654"] if submitted in a pair or SUBMISSION_IDS = ["123456"] if submitted alone.

SUBMISSION_IDS = []


import random


############
# QUESTION 1
############

# Q1 a
def prefix_suffix_overlap(lst, k):
	result_list = []
	for i in range(len(lst)):
		for j in range(len(lst)):
			if i != j:
				if (lst[i][:k] == lst[j][-k:]):
					result_list.append((i,j))
	return result_list


# Q1 c
class Dict:
	def __init__(self, m, hash_func=hash):
		""" initial hash table, m empty entries """
		self.table = [[] for i in range(m)]
		self.hash_mod = lambda x: hash_func(x) % m

	def __repr__(self):
		L = [self.table[i] for i in range(len(self.table))]
		return "".join([str(i) + " " + str(L[i]) + "\n" for i in range(len(self.table))])

	def insert(self, key, value):
		""" insert key,value into table
			Allow repetitions of keys """
		i = self.hash_mod(key)  # hash on key only
		item = [key, value]  # pack into one item
		self.table[i].append(item)

	def find(self, key):
		""" returns ALL values of key as a list, empty list if none """
		result_lst = []
		i = self.hash_mod(key)
		for curr in self.table[i]:
			if curr[0] == key:
				result_lst.append(curr[1])
		return result_lst



# Q1 d
def prefix_suffix_overlap_hash1(lst, k):
	result_lst = []
	d = Dict(len(lst))

	# O(nk)
	for i in range(len(lst)): # n iterations
		d.insert(lst[i][:k],i) # O(k) - slicing, O(1) - insert(average)

	# O(nk)
	for j in range(len(lst)): # n iterations
		match_indexes = d.find(lst[j][-k:]) # O(k) - slicing, O(1) - find(average)
		for index in match_indexes: # Q1-e: never happens
			if index != j:
				result_lst.append((index,j))

	return result_lst # Q1-e: empty list
		

	
############
# QUESTION 2
############

# Q2 a
def powers_of_2():
	number = 1
	while True:
		yield number # starting 2^0
		number = number * 2


# Q2 b
def pi_approx_monte_carlo():
	power_gen = powers_of_2()
	num_inside, total_amount = 0,0
	while True:
		curr_power = next(power_gen)
		for i in range(curr_power):
			x = random.random()
			y = random.random()
			if x**2 + y**2 <= 1.0:
				num_inside +=1
		total_amount += curr_power
		yield (4*num_inside/total_amount)

# Q2 c
def leibniz():
	n = 0
	while True:
		yield ((-1)**n)/(2*n+1)
		n +=1

def infinite_series(gen):
	sum_of_gen = 0
	for num in gen:
		sum_of_gen += num
		yield sum_of_gen


def pi_approx_leibniz():
	leb_gen = leibniz()
	sum_gen = infinite_series(leb_gen)
	power_gen = powers_of_2()
	while True:
		for i in range(next(power_gen)-1):
			next(sum_gen)
		yield 4*next(sum_gen)


# Q2 d
def unit_slicing():
	power_gen = powers_of_2()
	while True:
		power = next(power_gen)
		lst_res = [(i/power) for i in range(power)]
		yield lst_res
				


def integral(func, a, b):
	power_gen = powers_of_2()
	while True:
		integral_sum = 0
		power = next(power_gen)
		width = (b-a)/ power
		for i in range(power):	  
			integral_sum += func(a+width*i)*((a+width*(i+1))-(a+width*i))
		yield integral_sum

def pi_approx_integral():
	integral_gen = integral((lambda x: (1-x**2)**0.5),-1,1)
	while True:
		yield 2* next(integral_gen)


############
# QUESTION 6
############


# Q6 c
def CYK_d(st, rule_dict, start_var):
	''' can string st be generated from grammar? '''
	n = len(st)

	# table for the dynamic programming algorithm
	table = [[None for j in range(n+1)] for i in range(n)]
	#Initialize the relevant triangular region with empty sets
	for i in range(n):
		for j in range(i+1,n+1):
			table[i][j] = [set(), "none"]

	# Fill the table cells representing substrings of length 1
	fill_length_1_cells_d(table, rule_dict, st)
 
	# Fill the table cells representing substrings of length >=2
	for length in range(2, n+1):
		for i in range(0, n-length+1):
			j = i+length
			fill_cell_d(table, i,j, rule_dict)
			
	if (start_var in table[0][n][0]) == False:
		return -1
	else:
		return table[0][n][1]

def fill_length_1_cells_d(table, rule_dict, st):
	n = len(st)
	for i in range(n):
		for lhs in rule_dict: # lhs is a single variable
			if st[i] in rule_dict[lhs]:
			   # add variable lhs to T[i][i+1]
			   table[i][i+1][0].add(lhs)
		table[i][i+1][1] = 1


def fill_cell_d(table, i, j, rule_dict):
	for k in range(i+1, j): # non trivial partitions of s[i:j]
		for lhs in rule_dict: # lhs is a single variable
			for rhs in rule_dict[lhs]:
				if len(rhs) == 2: # rule like A -> XY (not like A -> a)
					X, Y = rhs[0], rhs[1]
					if X in table[i][k][0] and Y in table[k][j][0]:
						table[i][j][0].add(lhs)
						if table[i][j][1] == "none":
							table[i][j][1] = max(1+table[i][k][1],1+table[k][j][1])
						else:
							table[i][j][1] = min(table[i][j][1], max(1+table[i][k][1],1+table[k][j][1]))


########
# Tester
########

def test():
	import math

	############
	# QUESTION 1
	############

	# Q1 a
	lst = ["abcd", "cdab", "aaaa", "bbbb", "abff"]
	k = 2
	if sorted(prefix_suffix_overlap(lst, k)) != sorted([(0, 1), (1, 0), (4, 1)]):
		print("error in prefix_suffix_overlap")
	# Q1 c
	d = Dict(3)
	d.insert("a", 56)
	d.insert("a", 34)
	if sorted(d.find("a")) != sorted([56, 34]) or d.find("b") != []:
		print("error in Dict.find")

  
	# Q1 d
	lst = ["abcd", "cdab", "aaaa", "bbbb", "abff"]
	k = 2
	if sorted(prefix_suffix_overlap_hash1(lst, k)) != sorted([(0, 1), (1, 0), (4, 1)]):
		print("error in prefix_suffix_overlap_hash1")



	############
	# QUESTION 2
	############

	# Q2 a
	gen = powers_of_2()
	if [next(gen) for i in range(5)] != [1, 2, 4, 8, 16]:
		print('error in powers_of_2')
	

	# Q2 b
	gen = pi_approx_monte_carlo()
	first_apporx = next(gen)
	[next(gen) for i in range(8)]
	tenth_approx = next(gen)
	[next(gen) for i in range(9)]
	twentyth_approx = next(gen)
	if abs(first_apporx - math.pi) < abs(tenth_approx - math.pi) or \
			abs(tenth_approx - math.pi) < abs(twentyth_approx - math.pi) or \
			abs(twentyth_approx - math.pi) > 0.01:
		print('error in pi_approx_monte_carlo')
   
	# Q2 c
	gen = leibniz()
	if [next(gen) for i in range(5)] != [1, -1/3, 1/5, -1/7, 1/9]:
		print('error in leibniz')

	gen = infinite_series(powers_of_2())
	if [next(gen) for i in range(6)] != [1, 3, 7, 15, 31, 63]:
		print('error in infinite_series')

	leibniz_formula = [1, -1/3, 1/5, -1/7, 1/9, -1/11, 1/13, -1/15, 1/17, -1/19, 1/21, -1/23, 1/25, -1/27, 1/29]
	leibniz_sum_powers_of_2 = [4*leibniz_formula[0], 4*sum(leibniz_formula[:3]), 4*sum(leibniz_formula[:7]), 4*sum(leibniz_formula[:15])]
	gen = pi_approx_leibniz()
	first_4_sums = [next(gen) for i in range(4)]
	[next(gen) for i in range(5)]
	tenth_approx = next(gen)
	if first_4_sums != leibniz_sum_powers_of_2 or abs(tenth_approx - math.pi) > 1e-3:
		print('error in pi_approx_leibniz')

	# Q2 d
	gen = unit_slicing()
	if [next(gen) for i in range(4)] != [[0.0], [0.0, 0.5], [0.0, 0.25, 0.5, 0.75], [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]]:
		print('error in unit_slicing')

	b = 10
	true_val = math.log(b)
	gen = integral(lambda x: 1 / x, 1, b)
	first_apporx = next(gen)
	[next(gen) for i in range(8)]
	tenth_approx = next(gen)
	[next(gen) for i in range(9)]
	twentyth_approx = next(gen)
	if abs(first_apporx - true_val) < abs(tenth_approx - true_val) or \
			abs(tenth_approx - true_val) < abs(twentyth_approx - true_val) or \
			abs(twentyth_approx - true_val) > 1e-4:
		print('error in integral')
	
	gen = pi_approx_integral()
	first_apporx = next(gen)
	[next(gen) for i in range(8)]
	tenth_approx = next(gen)
	[next(gen) for i in range(9)]
	twentyth_approx = next(gen)
	if abs(first_apporx - math.pi) < abs(tenth_approx - math.pi) or \
			abs(tenth_approx - math.pi) < abs(twentyth_approx - math.pi) or \
			abs(twentyth_approx - math.pi) > 1e-5:
		print('error in pi_approx_integral')

	############
	# QUESTION 6
	############

	# Q6 c
	rule_dict = {"S": {"AB", "BC"}, "A": {"BA", "a"}, "B": {"CC", "b"}, "C": {"AB", "a"}}
	if CYK_d("baaba", rule_dict, "S") != 4:
		print("Error in CYK_d1")

	if CYK_d("baab", rule_dict, "S") != -1:
		print("Error in CYK_d2")