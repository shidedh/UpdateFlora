import fitz
import numpy as np
import pandas as pd
from tqdm import tqdm

import io
from PIL import Image, ImageDraw, ImageFont, ImageColor

import math
import re


# ------------------------ IMPORT VOLUMES ------------------------ #
vol1_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 1.pdf'
vol2_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 2.pdf'
vol3_path = '../input/NOUVELLE FLORE DU LIBAN ET DE LA SYRIE 3.pdf'

vol1_doc = fitz.open(vol1_path)
vol2_doc = fitz.open(vol2_path)
vol3_doc = fitz.open(vol3_path)

vol1_pages = [vol1_doc[i] for i in range(vol1_doc.page_count)]
vol2_pages = [vol2_doc[i] for i in range(vol2_doc.page_count)]
vol3_pages = [vol3_doc[i] for i in range(vol3_doc.page_count)]

vol1_char_df = pd.read_pickle("../input/char_df/vol1_df.pkl")
vol2_char_df = pd.read_pickle("../input/char_df/vol2_df.pkl")
vol3_char_df = pd.read_pickle("../input/char_df/vol3_df.pkl")

vol1_index = list(range(616, 639))
vol2_index = list(range(703, 725))
vol3_index = list(range(555, 583))
# ---------------------------------------------------------------- #


# ---------------------- Set Global Values ----------------------- #
TARGET_DPI = 300
mat = fitz.Matrix(TARGET_DPI/ 72, TARGET_DPI/ 72)
# ---------------------------------------------------------------- #

# -------------------------- Functions --------------------------- #

def genus_match(row):
    """given a row of the vol1_char_df"""
    word_rspace_removed = row['word'].rstrip()
    return row['word_num'] == 0 and \
           word_rspace_removed.isalpha() and \
           word_rspace_removed[0].isupper() and word_rspace_removed[1:].islower()
           
def epithet_match(row):
    word_rspace_removed = row['word'].rstrip()
    return row['word_num'] == 0 and \
           word_rspace_removed.isalpha() and \
           word_rspace_removed.islower()
# ---------------------------------------------------------------- #


#rightmost point of any bounding box:
# -------- Finding Exact Matches for Genera and Epithets -------- #
def get_center_x0(vol_char_df, page_num, bias = 30):
    """WARNING: Bias = 30 large bias causes miscatagorization in page number in book"""
    df = vol_char_df[vol_char_df['page_num'] == page_num]
    
    right_bound = df['line_bbox'].apply(lambda x : x[2]).max() 
    #leftmost point of any bounding box:
    left_bound = df['line_bbox'].apply(lambda x : x[0]).min()

    return 0.5*(right_bound + left_bound) - bias


def get_col_num(coords, center_x0):
    x0, y0, x1, y1 = coords
    return int(x0 >= center_x0)
# ---------------------------------------------------------------- #