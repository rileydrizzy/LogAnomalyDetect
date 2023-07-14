"""
doc
"""

#config creation 
from preprocess import test_func
import hydra 


@hydra()
def main(cfg):
    print(cfg)