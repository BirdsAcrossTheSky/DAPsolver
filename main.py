import random
from operator import itemgetter
import time


def read_file(file_name):
	"""
	transforming file lines into list of strings and removing redundant contents of generated list (like comments,
	empty lines, spaces)

	params: file_name (string)

	returns: str_of_nums_list
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

	Params: table (list of strings)
			i (index 1)
			j (index 2)

	Returns: int(c) (int)
			 j (int)
	"""
	c = table[i][j]
	while j + 1 < len(table[i]) and table[i][j + 1] != ' ':
		c = c + table[i][j + 1]
		j = j + 1

	return int(c), j


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
	Generating table of ordered data
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
Global variables meaning:
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
	:return:
	"""
	print()
	print("Link number E: %s" % E)
	for i in range(1,E+1):
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

def evaluate(x): #Funkcja obliczająca F(x) dla chromosomu x
	
	load = [] #przepływ przez krawędź n+1 l(e,x)
	for i in range(E):
		load.append(0)
		for j in range(D):
			for k in range(p_cnt[j]):
				for l in all_links[j][k]:
					if i+1 == l:
						load[i] += x[j][k]
	Y = [] # Overload
	for i in range (E):
		Y.append(load[i] - Ce[i])

	Fx = max(Y)
	return Fx

def random_number(z): # Funkcja generująca losową liczbę całkowitą z zakresu [0 - z]
	#if z < 0: # pomaga wykryć niepoprawne użycie
	#	print("NIEPOPRAWNY ARGUMENT (z < 0) FUNKCJI random_number(z)")
	#	return
	x = random.randrange(0, z+1)
	return x

def generate_ran_x(): #Funkcja generująca losowy chromosom
	x = []
	y = []
	for i in range(D):
		hd_used = 0 # Zapotrzebowanie wykorzystane na poprzednie ścieżki
		for j in range(p_cnt[i]):
			if j == p_cnt[i] - 1:
				y.append(hd[i] - hd_used)
			else:
				y.append(random_number(hd[i] - hd_used))
				hd_used += y[j]
		x.append(y)
		y = []

	return x

def generate_ran_x3(): #Funkcja generująca losowy chromosom
	x = []
	y = []
	for i in range(D):
		r = random.randrange(D)
		ri = (r+i)%D # Funkcja losuje od którego genu zaczyna przydzielanie 
		hd_used = 0 # Zapotrzebowanie wykorzystane na poprzednie ścieżki
		for j in range(p_cnt[i]):
			if j == p_cnt[ri] - 1:
				y.append(hd[ri] - hd_used)
			else:
				y.append(random_number(hd[ri] - hd_used))
				hd_used += y[j]
		x.append(y)
		y = []

	return x

def generate_ran_x2():
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

def generate_x_of_0():
	x = []
	y = []
	for i in range(D):
		for j in range(p_cnt[i]):
			y.append(0)
		x.append(y)
		y = []
	return x

def initialize(n): # Funkcja tworząca generacje chromosomów w sposób losowy
    generation0 = []
    for i in range(n):
        generation0.append(generate_ran_x2())
    return generation0

def initialize2(n): # Funkcja tworząca generacje chromosomów w sposób losowy
	generation0 = []
	a = int(n/2)
	for i in range(a):
		generation0.append(generate_ran_x3())
	for j in range(n-a):
		generation0.append(generate_ran_x2())
	return generation0

def initialize_with_0(n):
	generation = []
	for i in range(n):
		generation.append(generate_x_of_0())
	return generation

def crossover(p1, p2):
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
	i = random.randrange(D)
	while hd[i] == 0: # Zabezpieczenie przed hd = 0
		i = random.randrange(D)
	j = random.randrange(p_cnt[i])
	while x[i][j] == 0: # Zabezpieczenie przed przeniesieniem zasobu z miejsca gdzie go nie ma
		j = random.randrange(p_cnt[i])
	x[i][j] -= 1
	k = random.randrange(p_cnt[i])
	while k == j: # Zabezpieczenie przed przeniesieniem zasobu do miejsca z którego został on wzięty
		k = random.randrange(p_cnt[i])
	x[i][k] += 1

	return

def mutate2(x): # Mutacja polegająca na wylosowaniu nowego genu losowego chromosomu
	i = random.randrange(D)
	x_rand = generate_ran_x2()
	while x_rand[i] == x[i]:
		if p_cnt[i] == 1: # OBSERWUJ TO
			break
		x_rand = generate_ran_x2()
	x[i] = x_rand[i] # napisać analogiczną bądź inną dedykowaną funkcje dla pojedyńczego genu
	return

def mutate22(x):
	x_rand = generate_ran_x2()	
	for i in range(D):
		r_mut_gen = random.randrange(mut_rate)
		if r_mut_gen == 0:
			while x_rand[i] == x[i]:
				if p_cnt[i] == 1: # OBSERWUJ TO
					break
				x_rand = generate_ran_x2()
			x[i] = x_rand[i]
	return

def index_list(population_n): # Funkcja obliczająca F(x) populacji i zapisująca indeks
		Fx1 = [] # [numer chromosomu w populacji][0 - indeks, 1 - Fx]
		Fx_ind = []
		for i in range(len(population_n)):
			Fx_ind.append(i) #przydzielenie indeksu
			Fx_ind.append(evaluate(population_n[i])) #przydzielenie F(x)
			Fx1.append(Fx_ind)
			Fx_ind = []
		return Fx1

def sort_index(population_n): # Funkcja sortująca indeksy chromosomów od tych odpowidających najmniejszemu F(x)
	# do odpowiadających F(x) największemu		population_n = population[n]
	# Fx_sorted: [[indeks_best, Fx_best][indeks_second_best, Fx_second_best]...]

	Fx_sorted = sorted(index_list(population_n), key = itemgetter(1))
	return Fx_sorted

def merge_index(Fx1, Fx2):
	for i in range(len(Fx1)): # Dodaje drugi index o wartości 0 jezeli chromosom ze zbioru podstawowego
		Fx1[i].append(0)
	for j in range(len(Fx2)): # Dodaje drugi index o wartości 1 jezeli chromosom ze zbioru O(n)
		Fx2[j].append(1)
	#print("[INDEKS,	 F(X),	 0 DLA P(N) LUB 1 DLA O(N)]")
	#print("P(n):		", Fx1)
	#print("O(n):		", Fx2)
	# Sorted z którego korzystamy jest sortowaniem stabilnym, zatem zachowującym kolejność. Oznacza to, że kolejność dodawania list ma znaczenie. 
	# Aby zwiększyć udział potomków w populacji lista indeksów zbioru O(x) powinna być zapisana przed znakiem dodawania
	Fx12 = Fx2 + Fx1
	Fx3 = sorted(Fx12, key = itemgetter(1))
	#print("O(n) + P(n):	", Fx3)
	Fx_merged = Fx3[:len(Fx1)]
	#print("P(n+1):		", Fx_merged)

	return Fx_merged

def display_chromosomes(list_of_x):
	for s in range(len(list_of_x)):
		print(str(s+1) + ".", "	F(x):", evaluate(list_of_x[s]), "	", list_of_x[s])
	print()
	return

def display_sorted_chromosomes(population_n, Fx):
	for s in range(len(population_n)):
		print(str(s+1) + ".", "	F(x):", Fx[s][1], "	", population_n[Fx[s][0]])
	print()
	return
	

def present_chromosome(x):
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
    print()
    for i in range(max(p_cnt)):
        for j in range(D):
            try:
                print("", x[j][i], end = "")
            except IndexError:
                print(" -", end = "")
                #x[j][i] = '-'
            #continue
        print("\n\n")


start_time = time.time()
lines_from_file = read_file("network_files/net12_2_for_python_1.txt")
# r = [*filter(None, lines_from_file)]

rr = parse_to_array(lines_from_file)
#print(rr) # Wyświetlenie tablicy danych odczytanych z pliku
generate_tables(rr)
#display() #Przystępne wyświetlenie najważniejszych danych odczytanych z pliku

x_test1 = [[0, 3, 0], [2, 0, 2], [2, 3], [1, 0, 1], [1, 2, 0], [2, 2, 0]]
x_test2 = [[0, 0, 3], [0, 4, 0], [5, 0], [0, 2, 0], [0, 0, 3], [0, 0, 4]]

#x_cross1, x_cross2 = crossover(x_test1, x_test2)
#print(x_cross1)
#print(x_cross2)
#x_mut = mutate(x_test1)
#print(x_mut)
#r2 = generate_ran_x2()
#print(r2)

# ---------------------- MAIN ------------------
global mut_rate

# ------------------ STEROWANIE ----------------------
population_size = 100 # Liczba chromosów w populacji
M = 8 #szansa na zmutowanie chromosomu: 1/M
mut_rate = 8 #szansa na zmutowanie każdego genu w mutowanym chromosomie
#k_param = 0.8 # stosunek ilości potomków do wielkości populacji    K_wykl = k_param * population_size
K = 40 # Ilość potomków populacji = K * 2
F_stb_max = 10 # Ile iteracji bez poprawy Fx_min się wykona przed zakończeniem pętli
# ------------------------------------------------------

population = []		# population[x] - generacja x
n = 0
#print("POPULACJA", n)
population.append(initialize2(population_size))
#K = int(k_param*population_size/2) # dostosowanie zmiennej stosunku liczby potomków do programu
F_best = []
Fx_n = sort_index(population[n])
#print(Fx_n)
Fx_0_max = Fx_n[-1][1] # Fx najgorszego chromosomu
worst_index = Fx_n[-1][0]
Fx_0_min = Fx_n[0][1]
F_best.append(Fx_0_min)
#print("Wygenerowana losowo populacja 0:")
#display_sorted_chromosomes(population[n], Fx_n)
#print("F(x) najlepszego chromosomu populacji:", 	F_best[n])
#input("Press Enter to continue...")
#print("\n\n\n")

F_stb_cnt = 0 # Stability counter - Ilość iteracji bez zmiany Fx_min

#print(population[n][0]) # To nie jest najlepszy chromosom w populacji!

while F_stb_cnt < F_stb_max: 


	setOx = [] # zbiór O(x)

	'''# Crossover - Queen of the Bees
	for j in range(K): 
		for k in range(2):
			setOx.append(generate_x_of_0) # należy dodać dwa chromosomy, którym przypisze się potem potomków
		r1 = random.randrange(1, population_size)
		while population[n][Fx_n[0][0]] == population[n][Fx_n[r1][0]]: # Unikniecie sytuacji w której chromosom skrzyżuje się ze swoim bliźniakiem 
			#POTENCJALNIE NIEBEZPIECZNE - Czy można porównywać w ten sposoób dwuwymiarowe listy?
			r1 = random.randrange(1, population_size)
		setOx[2*j], setOx[2*j+1] = crossover(population[n][Fx_n[0][0]], population[n][Fx_n[r1][0]]) # Queen of the bees - Krzyżuje się najlepszy chromosom
		# z chromosomem losowo do niego dobranym
	'''
	# Crossover - prawdopodobieństwo oparte na funkcji celu v1
	#'''
	Z_vec0 = [] # wektor znormalizowanych funkcji celu 
	for i in range(population_size):
		Z_vec0.append(abs(Fx_n[i][1] - Fx_n[-1][1]) + 1)
	#print(Z_vec0)
	Z_vec1 = [] # wektor indeksów populacji - czym lepszy chromosom tym większa liczba jego indeksów
	for j in range(population_size):
		for k in range(Z_vec0[j]**2):
			Z_vec1.append(Fx_n[j][0])
	#print(Z_vec1)
	for k in range(K): 
		for l in range(2):
			setOx.append(generate_x_of_0) # należy dodać dwa chromosomy, którym przypisze się potem potomków
		rc1 = random.randrange(len(Z_vec1))
		rc2 = random.randrange(len(Z_vec1))
		# while population[n][Z_vec1[rc1]] == population[n][Z_vec1[rc2]]: # Unikniecie sytuacji w której chromosom skrzyżuje się ze swoim bliźniakiem
		# W przypadku gdy wszystkie chromosomy są takie same to kod się zacina 
		# 	rc2 = random.randrange(len(Z_vec1))
		setOx[2*k], setOx[2*k+1] = crossover(population[n][Z_vec1[rc1]], population[n][Z_vec1[rc2]])
	#'''
	#print("Potomkowie populacji", str(n) + ":")
	#display_chromosomes(setOx)

	"Z JAKIEGOŚ POWODU JAK R_MUT = 0 DLA JEDNEGO CHROMOSOMU TO RESZTA TEŻ SIĘ MUTUJE (JEŻELI RESZTA JEST TAKA SAMA) NIEDOPRECYZOWANY INDEKS?"
	for k in range(K*2):
		r_mut = random.randrange(M) # IF (1, M) => MUTACJA WYŁĄCZONA
		#print("r_mut", k+1, 	r_mut)
		if r_mut == 0:
			mutate22(setOx[k]); #print("MUTUJE: k+1 =", k+1)
	#print()
	#print("Potomkowie populacji", str(n) + " po mutacjach:")
	#display_chromosomes(setOx)
	
	n += 1
	Fx2 = sort_index(setOx)
	Fx_merged = merge_index(Fx_n, Fx2)
	F_best.append(Fx_merged[0][1]) # Fx najlepszego chromosomu
	population.append(initialize_with_0(population_size))
	for i in range(population_size): # Na podstawie złączonej tablicy indeksów tworzona zostaje nastepna populacja 
		if Fx_merged[i][2] == 0:
			population[n][i]=population[n-1][Fx_merged[i][0]]
		else: #elif merged[i][2] == 1:
			population[n][i]=setOx[Fx_merged[i][0]]
		#else:
		#	print("BŁĄD! Niewłaściwa wartość.")
	#Fx_n_sorted = sort_index(population[n])
	Fx_n = sort_index(population[n])
	#Fx_n_unsorted = index_list(population[n])
	#print("Fx_n:		", Fx_n)
	#print("Fx_n_unsorted:	", Fx_n_unsorted)

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
	

print("Czas wykonania: {:.2f}s".format(time.time() - start_time))





