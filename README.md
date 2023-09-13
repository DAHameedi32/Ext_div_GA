# Exterior Derivative via Genetic Algorithm

##Introduction
This is a program which can be used to compute the exterior derivative between k-forms represented as matrices. has 2 main dependencies: faer_core and rand.

##How to Use
To use the code you will have to start with a 1-form formatted as a faer vector (1xn matrix). From here you create vectors holding all the 1-forms and from there you can wedge them together as vectors or matrices to create forms of increasing rank. you can then run the GA_main() function in the module GA with appropriate parameters to have the algorithm execute. the function returnd a tuple of a matrix and a maximum fitness value. 
