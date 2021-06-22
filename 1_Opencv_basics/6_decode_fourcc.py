import cv2
import argparse

def decode_fourcc(fourcc):
    # Decodes the fourcc value to get the 4 chars identifying it

    fourcc_int = int(fourcc)

    # Print the value of fourcc
    print("int value of fourcc: {}".format(fourcc_int))


    #return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    fourcc_decode = ""
    for i in range(4):
        int_value = fourcc_int >> 8 * i & 0xFF
        print("int value: {}".format(int_value))
        fourcc_decode += chr(int_value)

    return fourcc_decode