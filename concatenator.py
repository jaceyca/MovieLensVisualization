trainMovieIDs = set()
testMovieIDs = set()

missing = []

with open("./data/train.txt") as f:
	for line in f:
		lst = line.split("\t")
		trainMovieIDs.add(lst[1])     # this is the movieID

with open("./data/test.txt") as f:
	for line in f:
		lst = line.split("\t")
		testMovieIDs.add(lst[1])

difference = testMovieIDs - trainMovieIDs    # movieIDs that are in the test set but not the train set
print(difference)

with open("./data/test.txt") as f:
	for line in f:
		lst = line.split("\t")
		if lst[1] in difference:
			missing.append(line)

with open("./data/traintest.txt", "w") as f:
	with open ("./data/train.txt") as g:
		for line in g:
			f.write(line)
		f.write("\n")
		for i in missing:
			f.write(i)

print("Done")
