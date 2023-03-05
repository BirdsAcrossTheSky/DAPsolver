import random
from operator import itemgetter
import time


def func_timer(func):
	"""
	Measure duration of function

	:param func: function that we want to time every time its executed
	:return: function with a timing mechanism
	"""
	def wrapper(*args, **kwargs):
		t0 = time.time()
		rslt = func(*args, **kwargs)
		t = time.time() - t0
		print("Duration of {} was {}s.".format(func.__name__, t))
		return rslt
	return wrapper


def func_counter(func):
	"""
	count number of times function was used

	:param func: function which number of executions we want to know
	:return: function with counting mechanism
	"""
	def wrapper(*args, **kwargs):
		wrapper.cnt += 1
		return func(*args, **kwargs)
	wrapper.cnt = 0
	return wrapper


@func_timer
def read_file(file_name):
	"""
	transform file lines into list of strings and remove redundant contents of generated list (like comments,
	empty lines, spaces)

	:param file_name: string
	:return: str_of_nums_list
	"""

	lines = []
	with open(file_name, 'r') as file:
		while True:
			line = file.readline()
			if not line:
				break
			lines.append(line.split("\t")[0].strip())
	str_of_nums_list = rm_empty_str(lines)
	return str_of_nums_list


@func_timer
def rm_empty_str(lst):
	"""
	removing empty strings from list, that came from empty lines in doc

	params: lst (list)

	returns: l (list)
	"""

	while "" in lst:
		lst.remove("")
	return lst


def separate(table, i, j):
	"""
	extracting numbers from strings and change it to integers

	:param table:
	:param i:
	:param j:
	:return c:
	:return j:
	"""
	c = table[i][j]
	while j + 1 < len(table[i]) and table[i][j + 1] != ' ':
		c = c + table[i][j + 1]
		j = j + 1
	c = int(c)

	return c, j


def parse_to_array(all_values_from_file):
	"""
	transforming list of strings of numbers into list of list that contain numbers, which were part of those strings

	Params: all_values_from_file (list of strings)

	Returns: items (list of lists)
	"""
	items = []
	vect = []
	i = 0

	while i < len(all_values_from_file):
		j = 0

		while j < len(all_values_from_file[i]):
			if all_values_from_file[i][j] != ' ':
				vect.append(separate(all_values_from_file, i, j)[0])
				j = separate(all_values_from_file, i, j)[1] + 1
			else:
				j += 1
		i += 1
		items.append(vect)
		vect = []

	return items


def generate_tables(rr):
	"""
	Generating tables of ordered data to global scope

	:param rr:
	:return:
	"""

	global E
	E = rr[0][0]
	global D
	D = rr[E+1][0]
	global E_table
	E_table = []
	global Ce
	Ce = []
	for i in range(1, E+1):
		Ce.append(rr[i][3])
		E_table.append(rr[i])
	
	global d_ind
	d_ind = []
	global p_ind
	p_ind = []
	P_start_ind = []
	d_ind.append(E+2)
	p_ind.append(d_ind[0]+1)
	P_start_ind.append(d_ind[0]+2)
	for i in range(1, D):
		d_ind.append(d_ind[i-1]+2+rr[p_ind[i-1]][0])
		p_ind.append(d_ind[i]+1)
		P_start_ind.append(d_ind[i]+2)
	
	global D_table
	D_table = []
	global hd
	hd = []
	global p_cnt
	p_cnt = []
	
	for j in range(D):
		D_table.append(rr[d_ind[j]])
		hd.append(rr[d_ind[j]][3])
		p_cnt.append(rr[p_ind[j]][0])

	link = []
	links = []
	global all_links
	all_links = []  # Table of links of all paths
	for z in range(D):  # Loop on every demand
		i = 0
		while i < rr[p_ind[z]][0]:  # Loop on every demand's path
			for k in rr[P_start_ind[z]+i][1:]:	 # Value of path loop
				link.append(k)
			links.append(link)
			link = []
			i += 1
		all_links.append(links)
		links = []

	# all_links[x][y][z] format: all_links[demand number- 1] [path number - 1] [link number - 1]
	return


''' 
Global variables legend:
 E - Links number
 D - Demnand number
 Ce - Link capacity
 hd - Demand capacity
 d_ind - Indexes of demands in file 
 p_ind - Indexes of paths in file
 p_cnt - Number of paths for demand
 all_links - Table of paths
 E_table - Table of links
 D_table - Table of demands
'''


def display():
	"""
	Displaying ordered data

	:return Fx: int
	"""
	print()
	print("Link number E: %s" % E)
	for i in range(1, E+1):
		print("e:%s, node_1: %s , node_2: %s, C(e): %s" % (rr[i][0], rr[i][1], rr[i][2], rr[i][3]))
	print()
	print("Demand number D: %s" % D)
	for j in range(D):
		print("d:%s, node_1: %s , node_2: %s, h(d): %s" % (rr[d_ind[j]][0], rr[d_ind[j]][1], rr[d_ind[j]][2], rr[d_ind[j]][3]))
	print()
	print("Path table P(): ")
	print(all_links)
	print()

	return

def evaluate(x):
	"""
	Function calculating objection function of evaluated chromosome

	:param x:
	:return fx:
	"""
	load = []  # load of a link n+1 l(e,x)
	for i in range(E):
		load.append(0)
		for j in range(D):
			for k in range(p_cnt[j]):
				for l in all_links[j][k]:
					if i+1 == l:
						load[i] += x[j][k]
	y = [] # Overload
	for i in range (E):
		y.append(load[i] - Ce[i])

	fx = max(y)
	return fx


def random_number(z):
	"""
	generate random number from range [0 - z]

	:param z:
	:return x:
	"""

	x = random.randrange(0, z+1)
	return x


def generate_ran_x1():
	"""
	generate random chromosome 1

	:return x: list of lists
	"""
	x = []
	y = []
	for i in range(D):
		hd_used = 0 # Demand used on previous paths
		for j in range(p_cnt[i]):
			if j == p_cnt[i] - 1:
				y.append(hd[i] - hd_used)
			else:
				y.append(random_number(hd[i] - hd_used))
				hd_used += y[j]
		x.append(y)
		y = []

	return x


def generate_ran_x2():
	"""
	generate random chromosome 2

	:return x:
	"""
	x = []
	y = []
	for i in range(D):
		for j in range (p_cnt[i]):
			y.append(0)
		for k in range(hd[i]):
			r = random.randrange(p_cnt[i])
			y[r] += 1
		x.append(y)
		y = []
	return x


def generate_ran_x3():
	"""
	generate random chromosome 3

	:return x:
	"""
	x = []
	y = []
	for i in range(D):
		r = random.randrange(D)
		ri = (r+i)%D  # randomize starting gene
		hd_used = 0  # Demand used on previous path
		for j in range(p_cnt[i]):
			if j == p_cnt[ri] - 1:
				y.append(hd[ri] - hd_used)
			else:
				y.append(random_number(hd[ri] - hd_used))
				hd_used += y[j]
		x.append(y)
		y = []

	return x


# list of functions generating random chromosomes, functions differ with a way they generate chromosome
gen_ran_x = [generate_ran_x1, generate_ran_x2, generate_ran_x3]


def generate_x_of_0():
	"""
	generate chromosome of 0s

	:return:
	"""
	x = []
	y = []
	for i in range(D):
		for j in range(p_cnt[i]):
			y.append(0)
		x.append(y)
		y = []
	return x


def initialize(n):
	"""
	generate random population

	:param n:
	:return generation0:
	"""
	generation0 = []
	for i in range(n):
		generation0.append(gen_ran_x[1]())
	return generation0


def initialize2(n):
	"""
	generate random population

	:param n:
	:return generation0:
	"""
	generation0 = []
	a = int(n/2)
	for i in range(a):
		generation0.append(gen_ran_x[2]())
	for j in range(n-a):
		generation0.append(gen_ran_x[1]())
	return generation0


def initialize_with_0(n):
	"""
	generate population of 0 chromosomes

	:param n:
	:return:
	"""
	generation = []
	for i in range(n):
		generation.append(generate_x_of_0())
	return generation


@func_counter
def crossover(p1, p2):
	"""
	create two chromosomes with genes randomly drawn from its parents (arguments)

	:param p1:
	:param p2:
	:return o1:
	:return o1:
	"""
	o1 = []
	o2 = []
	for i in range(D):
		r1 = random_number(1)
		r2 = random_number(1)
	if r1 == 0:
		o1.append(p1[i])
	else:
		o1.append(p2[i])

	if r2 == 0:
		o2.append(p1[i])
	else:
		o2.append(p2[i])

	return o1, o2


def mutate(x):
	"""


	:param x:
	:return:
	"""
	i = random.randrange(D)
	while hd[i] == 0:  # for hd = 0 case
		i = random.randrange(D)
	j = random.randrange(p_cnt[i])
	while x[i][j] == 0:  # to avoid taking demand from where there is none
		j = random.randrange(p_cnt[i])
	x[i][j] -= 1
	k = random.randrange(p_cnt[i])
	while k == j:  # to avoid come back of a demand to same place it's been taken from
		k = random.randrange(p_cnt[i])
	x[i][k] += 1

	return


def mutate2(x):
	"""
	Generate random chromosome in place of randomly chosen existing one

	:param x:
	:return:
	"""
	i = random.randrange(D)
	x_rand = generate_ran_x2()
	while x_rand[i] == x[i]:
		if p_cnt[i] == 1:
			break
		x_rand = generate_ran_x2()
	x[i] = x_rand[i]  # create similar function for gene?
	return


def mutate22(x):
	"""
	Generate random chromosome in place of randomly chosen existing one


	:param x:
	:return:
	"""
	x_rand = generate_ran_x2()	
	for i in range(D):
		r_mut_gen = random.randrange(mut_rate)
		if r_mut_gen == 0:
			while x_rand[i] == x[i]:
				if p_cnt[i] == 1:
					break
				x_rand = generate_ran_x2()
			x[i] = x_rand[i]
	return


def index_list(population_n): # Funkcja obliczająca F(x) populacji i zapisująca indeks
	"""
	Generate list of Fx for chromosomes of population, sort it and assign indexes NOT OPTIMAL

	:param population_n:
	:return Fx1:
	"""
	Fx1 = [] # [number of chromosome in population][0 - index, 1 - Fx]
	Fx_ind = []
	for i in range(len(population_n)):
		Fx_ind.append(i)  # index assignment
		Fx_ind.append(evaluate(population_n[i]))  # F(x) assignment
		Fx1.append(Fx_ind)
		Fx_ind = []
	return Fx1


def sort_index(population_n):
	"""
	sort pairs Fx-index by Fx

	:param population_n:
	:return:
	"""
	Fx_sorted = sorted(index_list(population_n), key=itemgetter(1))
	# Fx_sorted: [[index_best, Fx_best][index_second_best, Fx_second_best]...]
	return Fx_sorted


def merge_index(Fx1, Fx2):
	"""


	:param Fx1:
	:param Fx2:
	:return:
	"""
	for i in range(len(Fx1)):
		Fx1[i].append(0)  # Add another index equal to 0 if chromosome is from population
	for j in range(len(Fx2)):
		Fx2[j].append(1)  # Add another index equal to 1 if chromosome is from ser\t O(x)

	# Sorted uses stable sort so the order of appending lists matters.
	# To enlarge its share in population, index list of set O(x) should be placed first
	Fx12 = Fx2 + Fx1
	Fx3 = sorted(Fx12, key=itemgetter(1))
	Fx_merged = Fx3[:len(Fx1)]

	return Fx_merged

def display_chromosomes(list_of_x):
	"""
	display chromosomes

	:param list_of_x:
	:return:
	"""
	for s in range(len(list_of_x)):
		print(str(s+1) + ".", "	F(x):", evaluate(list_of_x[s]), "	", list_of_x[s])
	print()
	return


def display_sorted_chromosomes(population_n, Fx):
	"""
	display chromosomes which have been already sorted

	:param population_n:
	:param Fx:
	:return:
	"""
	for s in range(len(population_n)):
		print(str(s+1) + ".", "	F(x):", Fx[s][1], "	", population_n[Fx[s][0]])
	print()
	return
	

def present_chromosome(x):
    """
    present chromosome in readable way

    :param x:
    :return:
    """
    print()
    for i in range(max(p_cnt)):
        for j in range(D):
            try:
                print("   ", x[j][i], "     ", end = "")
            except IndexError:
                print("    -      ", end = "")
                #x[j][i] = '-'
            #continue
        print("\n\n")


def present_chromosome2(x):
    """
    present chromosome in readable way v2

    :param x:
    :return:
    """
    print()
    for i in range(max(p_cnt)):
        for j in range(D):
            try:
                print("", x[j][i], end = "")
            except IndexError:
                print(" -", end = "")
        print("\n\n")



start_time = time.time()
lines_from_file = read_file("network_files/net12_2_for_python_1.txt")
# r = [*filter(None, lines_from_file)]

rr = parse_to_array(lines_from_file)
generate_tables(rr)

# ---------------------- MAIN ------------------
global mut_rate

# ------------------ CONTROL PANEL ----------------------
population_size = 100  # Number of chromosomes in population
M = 8  # 1/M - chance for chromosome to be mutated
mut_rate = 8  # 1/mut_rate - chance for gene mutation in mutating chromosome
# k_param = 0.8 #  descendant number to population size ratio - K = k_param * population_size
# K = int(k_param*population_size/2) # transformation of k_param to K (Number of population descendants)
K = 40  # Number of population descendants = K * 2
F_stb_max = 10  # max number of iterations without improvement - end condition
# ------------------------------------------------------

""" 
Fx - OBJECTIVE FUNCTION
Fx is maximum overload of a link
The objective is to minimize it
"""
population = []		# population[x] - generation x
n = 0
population.append(initialize2(population_size))
F_best = []
Fx_n = sort_index(population[n])
Fx_0_max = Fx_n[-1][1] # Fx of the worst chromosome
worst_index = Fx_n[-1][0]
Fx_0_min = Fx_n[0][1]
F_best.append(Fx_0_min)

"""
TESTING
print("Randomly generated population 0:")
display_sorted_chromosomes(population[n], Fx_n)
print("F(x) of the best chromosome of population:", 	F_best[n])
input("Press Enter to continue...")
print("\n\n\n")
"""
F_stb_cnt = 0 # Stability counter - Number of iteration without Fx_min improvement

while F_stb_cnt < F_stb_max: 


	setOx = [] # Set O(x)

	'''
	# Crossover: Queen of the Bees - Best chromosome crosses with random one
	for j in range(K): 
		for k in range(2):
			setOx.append(generate_x_of_0)  # generating empty chromosomes which will became ascendants of chosen pair
		r1 = random.randrange(1, population_size)
		while population[n][Fx_n[0][0]] == population[n][Fx_n[r1][0]]: 
			r1 = random.randrange(1, population_size)  # So that crossover of same chromosomes is impossible
		setOx[2*j], setOx[2*j+1] = crossover(population[n][Fx_n[0][0]], population[n][Fx_n[r1][0]])
	
	'''
	# Crossover v2 - Probability based on objective function
	#'''
	Z_vec0 = []  # vector of normalized objective function values
	for i in range(population_size):
		Z_vec0.append(abs(Fx_n[i][1] - Fx_n[-1][1]) + 1)

	Z_vec1 = []
	# vector of population indexes - Number of chromosome indexes is grows ^2 with its objective function value
	for j in range(population_size):
		for k in range(Z_vec0[j]**2):
			Z_vec1.append(Fx_n[j][0])

	for k in range(K): 
		for l in range(2):
			setOx.append(generate_x_of_0)  # generating empty chromosomes which will became ascendants of chosen pair
		rc1 = random.randrange(len(Z_vec1))
		rc2 = random.randrange(len(Z_vec1))
		# while population[n][Z_vec1[rc1]] == population[n][Z_vec1[rc2]]:
		# 	rc2 = random.randrange(len(Z_vec1))  # So that crossover of same chromosomes is impossible
		# Not working when all chromosomes are the same
		setOx[2*k], setOx[2*k+1] = crossover(population[n][Z_vec1[rc1]], population[n][Z_vec1[rc2]])

	for k in range(K*2):
		r_mut = random.randrange(M)
		if r_mut == 0:
			mutate22(setOx[k])

	n += 1
	Fx2 = sort_index(setOx)
	Fx_merged = merge_index(Fx_n, Fx2)
	F_best.append(Fx_merged[0][1])  # Fx of best chromosome
	population.append(initialize_with_0(population_size))
	for i in range(population_size):  # Next population is created based on received index list
		if Fx_merged[i][2] == 0:
			population[n][i]=population[n-1][Fx_merged[i][0]]
		else:
			population[n][i]=setOx[Fx_merged[i][0]]
	Fx_n = sort_index(population[n])


	#print("Fx_n_sorted:	", Fx_n_unsorted)
	#Fx_test=sort_index(population[n])
	#print()
	#print("P(n+1):		", Fx_test)
	#print()

	#print()
	#print("POPULACJA", n)
	#print("Posortowane chromosomy populacji", str(n) + ":")
	#display_sorted_chromosomes(population[n], Fx_n)
	#print("F(x) najlepszego chromosomu populacji:	", F_best[n])
	
	if F_best[n] < F_best[n-1]:
		F_stb_cnt = 0
	elif F_best[n] == F_best[n-1]:
		F_stb_cnt += 1
	else:
		print("POPULACJA ZDEEWOLUOWAŁA!")
		F_stb_cnt = 0
		#break


	#input("Press Enter to continue...")
	#print(population[n][0])
	

	#input("Press Enter to continue...")
	#print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

print("Najgorszy uzyskany chromosom:\n")
present_chromosome2(population[0][worst_index])
print()
print("F(x) najgorszego chromosomu populacji:	", Fx_0_max)
print("\n\n")

print("Najlepszy uzyskany chromosom:\n")
present_chromosome2(population[n][0])
print()
print("F(x) najlepszego chromosomu populacji:	", F_best[n])
print("\n\n")

for i in range(n):
	print("Populacja", i + 1, ":          F(x): ", F_best[i])
	

print("Solving time: {:.2f}s".format(time.time() - start_time))
print("Function crossover was executed {} times.".format(crossover.cnt))





