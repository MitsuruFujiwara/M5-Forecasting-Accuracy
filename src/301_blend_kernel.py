
import pandas as pd
import numpy as np

from utils import line_notify, submit

#===============================================================================
# blending kernel predictions
#===============================================================================

def main():
    # load predictions
#    sub1 = pd.read_csv('../input/sub_058724.csv',index_col=0)
#    sub2 = pd.read_csv('../input/sub_057330.csv',index_col=0)
    sub3 = pd.read_csv('../input/sub_054431.csv',index_col=0)
    sub4 = pd.read_csv('../input/sub_054218.csv',index_col=0)
    sub5 = pd.read_csv('../input/sub_053171.csv',index_col=0)
    sub6 = pd.read_csv('../input/sub_051599.csv',index_col=0)

    # average
    sub = 0.5*sub3 + 0.5*sub4
    sub = 0.5*sub + 0.5*sub5
    sub = 0.5*sub + 0.5*sub6
#    sub = 0.5*sub + 0.5*sub5

    # reset index
    sub.reset_index(inplace=True)

    # save csv
    sub.to_csv(submission_file_name,index=False)

    # submission by API
    submit(submission_file_name, comment='model301 blending')

if __name__ == '__main__':
    submission_file_name = '../output/sub_blend.csv'
    main()
