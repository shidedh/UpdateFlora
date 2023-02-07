import fitz
import numpy as np
import pandas as pd
from tqdm import tqdm

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
# ---------------------------------------------------------------- #


# ---------------------- Set Global Values ----------------------- #
TARGET_DPI = 300
mat = fitz.Matrix(TARGET_DPI/ 72, TARGET_DPI/ 72)
# ---------------------------------------------------------------- #


# ------- Functions For Extracting Each CHARACTER and WORD ------- #
#note bbox not adjusted with TARGET_DPI
def page_raw_dict_reformat(vol, page, page_num, word_list):
    """ DOCUMENTATION
    This function appends the characters and relevent information 
    corresponds to it in each page to word_list. Formatted such that 
    word_list is ready to represent each char as a pandas dataframe row 
    with column names:
        ['vol_num', 
         'page_num', 
         'block_num',
         'block_num_absolute', 
         'block_bbox',
         'line_num', 
         'line_wmode', 
         'line_dir', 
         'line_bbox', 
         'span_size',
         'span_flags', 
         'span_font', 
         'span_color', 
         'span_ascender',
         'span_descender', 
         'span_origin', 
         'span_bbox', 
         'word_num', 
         'word',
         'char_num', 
         'char', 
         'char_origin', 
         'char_bbox']

    INPUTS: 
        vol: String corresponding to volume number 
        page: instance of document at index page_num
        page_num: Int corresponging to page number
        word_list: List to which each character will be appended
    """
    text_blocks = [
        block for block in page.get_text("rawdict")['blocks'] 
        if block['type'] == 0
        ]

    #for each block in text blocks
    for b_i in range(len(text_blocks)):
        b = text_blocks[b_i]
        curr_block_num_abs = b['number']                    #true blcok number
        curr_block_num_reletive = b_i                       #excludes image blocks
        curr_block_bbox = b['bbox']

        for l_i in range(len(b['lines'])):
            l = b['lines'][l_i]
            curr_line_num = l_i
            curr_line_wmode = l['wmode']
            curr_line_dir = l['dir']
            curr_line_bbox = l['bbox']
            
            for s in l['spans']:
                span_size = s['size']
                span_flags = s['flags']
                span_font = s['font']
                span_color = s['color']
                span_ascender = s['ascender']
                span_descender = s['descender'] 
                span_chars = s['chars'] 
                span_origin = s['origin'] 
                span_bbox = s['bbox']
                
                word_num = 0
                span_word = ""
                char_in_words = []
                for c in span_chars:
                    char = c['c']
                    if char.isspace() and len(span_word) > 0:
                        #add word to dictionary
                        for c_i in range(len(char_in_words)):
                            char_num = c_i 
                            char_origin = char_in_words[c_i]['origin']
                            char_bbox = char_in_words[c_i]['bbox']
                            curr_char = char_in_words[c_i]['c']

                            char_row = {
                                'vol_num': vol,
                                'page_num': page_num,
                                'block_num': curr_block_num_reletive,
                                'block_num_absolute': curr_block_num_abs,
                                'block_bbox': curr_block_bbox,

                                'line_num': curr_line_num,
                                'line_wmode': curr_line_wmode,
                                'line_dir': curr_line_dir,
                                'line_bbox': curr_line_bbox,

                                'span_size': span_size,
                                'span_flags': span_flags,
                                'span_font': span_font,
                                'span_color': span_color,
                                'span_ascender': span_ascender,
                                'span_descender': span_descender,
                                'span_origin': span_origin,
                                'span_bbox': span_bbox,

                                'word_num': word_num,
                                'word': span_word,

                                'char_num': c_i,
                                'char': curr_char,
                                'char_origin': char_origin,
                                'char_bbox': char_bbox
                            }
                            word_list.append(char_row)

                        word_num += 1
                        span_word = ''
                        char_in_words = []
                    elif not char.isspace():  
                        span_word += char
                        char_in_words.append(c)
                    #only other possibility is that is it a white space in which case we can ignore it

def book_char_df(vol, pages):
    word_list = []
    for page_num in tqdm(range(len(pages))):
        page = pages[page_num]
        page_raw_dict_reformat(vol, page, page_num, word_list)
    return pd.DataFrame(word_list)
# ---------------------------------------------------------------- #

# ----------------------- Extract And Save ----------------------- #
print("\nextracting volume 1")
vol1_df = book_char_df("1", vol1_pages)

print("\nextracting volume 2")
vol2_df = book_char_df("2", vol2_pages)

print("\nextracting volume 3")
vol3_df = book_char_df("3", vol3_pages)

print("\n\nSaving volume 1")
vol1_df.to_csv("../input/char_df/vol1_df.csv")

print("Saving volume 2")
vol2_df.to_csv("../input/char_df/vol2_df.csv")

print("Saving volume 3")
vol3_df.to_csv("../input/char_df/vol3_df.csv")
# ---------------------------------------------------------------- #