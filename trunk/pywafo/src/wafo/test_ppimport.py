from scipy.misc.ppimport import ppimport
st = ppimport('scitools')

def main():
    t = st.numpytools.linspace(0,1)
    print(t)
    
if __name__ == '__main__':
    main()