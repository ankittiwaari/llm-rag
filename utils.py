import random
import string

def generate_random_string(length):
    """
    Generates a random string of a specified length using a mix of
    uppercase letters, lowercase letters, and digits.
    """
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string