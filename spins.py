"""
Functions for generating, manipulating spin configurations

For the (common) case of S=1/2, 0, 1 denote spin up, down
"""
import random
import random_pair
import choose
import numpy as N

def afm_1d_config(nsites):
    """
    int -> list(int)
    AFM spin config in 1D
    """
    spin_config=[None]*nsites
    for i in xrange(nsites):
        if i & 1: spin_config=1
    return spin_config

def configs(nsites, ndim):
    """
    int -> list(int)
    Generate list of all possible spin configs
    associated with a given number of sites
    """
    return [list(config) for config in N.ndindex(tuple([ndim]*nsites))]

def to_config(spin_ups, nsites):
    spin_string=N.zeros([nsites],N.int)
    spin_string[spin_ups]=1
    return list(spin_string)

def conserving_configs(nsites, ndim, totalsz):
    """
    list of all possible spin configs
    that conserve total Sz
    associated with a given number of sites
    """
    nspinup=(2*totalsz+nsites)/2
    return [to_config(choice, nsites) for choice in
            choose.choose(range(nsites), nspinup)]

def flip(spin):
    """
    int -> int
    """
    return (spin+1) & 1

def generate_move(config):
    """
    list(int) -> list(int), tuple(int,)
    Flip a random spin, return new config and flipped pos
    """
    new_config=config[:]
    nsites=len(config)
    touched_site=random.randint(0, nsites-1)
    new_config[touched_site]=flip(new_config[touched_site])
    return new_config, (touched_site,)

def generate_conserving_move(config):
    """
    Flip two spins, to conserve Sz
    """
    new_config=config[:]
    nsites=len(config)

    ups=[i for i in xrange(nsites) if config[i]]
    downs=[i for i in xrange(nsites) if not config[i]]
    
    flip_up=random.choice(ups)
    flip_down=random.choice(downs)
    new_config[flip_up]=flip(new_config[flip_up])
    new_config[flip_down]=flip(new_config[flip_down])

    return new_config, (flip_up, flip_down)
