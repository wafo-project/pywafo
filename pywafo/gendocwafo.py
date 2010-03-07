'''
Runs epydoc to document pywafo
'''
import os

print('Generating html documentation for wafo in folder html.')

os.system('epydoc.py --html -o html --name wafo --graph all src/wafo')