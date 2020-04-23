
# TODO: 

def main():
    # save out of fold prediction
    train_df.loc[:,'demand'] = oof_preds
    train_df = train_df.reset_index()
    train_df[['id', 'demand']].to_csv(oof_file_name, index=False)

    # reshape prediction for submit
    test_df.loc[:,'demand'] = sub_preds
    test_df = test_df.reset_index()
    preds = test_df[['id','d','demand']].reset_index()
    preds = preds.pivot(index='id', columns='d', values='demand').reset_index()

    # split test1 / test2
    preds1 = preds[['id']+COLS_TEST1]
    preds2 = preds[['id']+COLS_TEST2]

    # change column names
    preds1.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]
    preds2.columns = ['id'] + ['F' + str(d + 1) for d in range(28)]

    # replace test2 id
    preds2['id']= preds2['id'].str.replace('_validation','_evaluation')

    # merge
    preds = preds1.append(preds2)

    # save csv
    preds.to_csv(submission_file_name, index=False)

    # submission by API
    submit(submission_file_name, comment='model202 cv: %.6f' % full_rmse)

if __name__ == '__main__':
    main()
