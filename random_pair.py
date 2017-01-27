import random
def random_pair_int(begin, end):
    assert end > begin
    first, second=random.randint(begin, end), random.randint(begin, end-1)
    if second >= first: second += 1
    return first, second