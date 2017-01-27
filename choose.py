def choose(things, n):
    """
    list, int -> list(list)
    return all combinations of n choices from things
    """
    if n==1: return [[thing] for thing in things]
    choices=[]
    for i, thing in enumerate(things):
        choices+=[[thing]+choice for choice in choose(things[i+1:], n-1)]
    return choices
