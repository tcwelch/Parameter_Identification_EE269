- The data is available as .mat or .csv (the data is the same between the .csv files or the .mat files are used)
- It has been shuffled and pre-spliced into 5 seperate cross validation
	- These 5 slices are in seperate files for the .csv format and are in
	one file  with slices of corresponding X,Y sets denoted with 1-5 like this: X1,Y1
- I changed the noise to have sigma = 0.01 rather than 0.1 because 0.1 proved to be too large to distinguish anything
- Do not generate new data (i.e. do NOT run generate_data) since we want the dataset to stay constant