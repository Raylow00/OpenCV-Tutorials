import sys

# This library handles the arguments and complex parameters easily
import argparse

#print("The name of the script being processed is: '{}'".format(sys.argv[0]))
#print("The number of arguments of the script is: '{}'".format(len(sys.argv)))
#print("The arguments of the script are: '{}'".format(str(sys.argv)))

# We first create the argument parser object
# The created object 'parser' will have the necessary information
# to parse the command-line arguments into data types
parser = argparse.ArgumentParser()

# We add a positional argument using add_argument() including a help
parser.add_argument("first_number", help="This is an example argument", type=int)
parser.add_argument("second_number", help="This is the second number to add", type=int)

# The information about program arguments is stored in 'parser' and used when
# ArgumentParser parses arguments through the parse_args() method
args = parser.parse_args()
print("This is the first argument: ", args.first_number, " and this the second argument: ", args.second_number)

# The arguments can be stored in a dictionary calling vars() function
args_dict = vars(parser.parse_args())
print("args_dict dictionary: '{}'".format(args_dict))
print("First argument from the argument dictionary: '{}'".format(args_dict['first_number']))

