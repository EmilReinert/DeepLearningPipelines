"""
defective labled file taken from Motivation Example of Papaer
"""
def test_defective(stack):
    x = 0
    if not stack.empty():
        while x < 10:
            y = 0
            y = stack.pop()
            x += 1