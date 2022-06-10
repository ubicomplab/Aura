from utils import load_extracted_data, load_data
import seaborn as sns
from matplotlib import pyplot as plt

def main():
    d1 = load_data("t1_march10")
    d2 = load_data("t3_march10")
    plt.figure()
    sns.distplot(d1.x, kde=False, norm_hist=True)
    sns.distplot(d2.x, kde=False, norm_hist=True)
    plt.figure()
    sns.distplot(d1.y, kde=False, norm_hist=True)
    sns.distplot(d2.y, kde=False, norm_hist=True)
    plt.figure()
    sns.distplot(d1.z, kde=False, norm_hist=True)
    sns.distplot(d2.z, kde=False, norm_hist=True)


    plt.figure()
    sns.distplot(d1.m_6, kde=False, norm_hist=True)
    sns.distplot(d2.m_6, kde=False, norm_hist=True)
    plt.figure()
    sns.distplot(d1.m_7, kde=False, norm_hist=True)
    sns.distplot(d2.m_7, kde=False, norm_hist=True)
    plt.figure()
    sns.distplot(d1.m_8, kde=False, norm_hist=True)
    sns.distplot(d2.m_8, kde=False, norm_hist=True)



    plt.show()
    print(d1)

if __name__ == "__main__":
    main()