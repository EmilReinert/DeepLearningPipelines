"""
clean labled file taken from Motivation Example of Papaer
"""


def test_clean(stack): 
    x = 0
    while x < 10:
        y = 0
        if not stack.empty():
            y = stack.pop()
    x += 1
