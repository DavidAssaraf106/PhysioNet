labels = ['AF', 'AFL', 'Brady', 'IAVB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR',
                  'PAC', 'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'SNR', 'STach', 'TAb', 'TInv']
results_1 = {'AF': 0.6855404773046326, 'AFL': 0.3391959798994975, 'Brady': 0.38205980066445183,
             'IAVB': 0.39159109645507006, 'IRBBB': 0.039414414414414414, 'LAnFB': 0.16422435573521982,
             'LAD': 0.07835455435847209, 'LBBB': 0.8126195028680688, 'LPR': 0.278372591006424,
             'LQRSV': 0.24208566108007448, 'LQT': 0.39404553415061294, 'NSIVCB': 0.012376237623762377,
             'PR': 0.37722419928825623, 'PAC': 0.7512953367875648, 'PVC': 0.3282275711159737,
             'QAb': 0.03765060240963856, 'RAD': 0.0847457627118644, 'RBBB': 0.5510116229014206,
             'SA': 0.48633440514469456, 'SB': 0.6157112526539278, 'SNR': 0.640810152113623,
             'STach': 0.7962740384615384, 'TAb': 0.1868377021751255, 'TInv': 0.032981530343007916}
results_2 = {'AF': 0.8142006802721088, 'AFL': 0.30405405405405406, 'Brady': 0.32653061224489793,
             'IAVB': 0.37916666666666665, 'IRBBB': 0.1076923076923077, 'LAnFB': 0.44131028207461326,
             'LAD': 0.53515625, 'LBBB': 0.8349328214971209, 'LPR': 0.1746031746031746,
             'LQRSV': 0.18962075848303392, 'LQT': 0.3477822580645161, 'NSIVCB': 0.09003601440576231,
             'PR': 0.3325942350332594, 'PAC': 0.7917888563049853, 'PVC': 0.28938906752411575, 'QAb': 0.0,
             'RAD': 0.3838582677165354, 'RBBB': 0.3729281767955801, 'SA': 0.5096660808435852,
             'SB': 0.6424433912585571, 'SNR': 0.8002177463255308, 'STach': 0.8158368326334733,
             'TAb': 0.14796547472256474, 'TInv': 0.03942181340341656}
results_3 = {'AF': 0.8478802992518704, 'AFL': 0.26223776223776224, 'Brady': 0.2631578947368421,
             'IAVB': 0.6198960653303638, 'IRBBB': 0.03954802259887006, 'LAnFB': 0.4360867558837102,
             'LAD': 0.5638801261829653, 'LBBB': 0.8570075757575758, 'LPR': 0.132013201320132,
             'LQRSV': 0.1934826883910387, 'LQT': 0.32388663967611336, 'NSIVCB': 0.06105006105006105,
             'PR': 0.536309127248501, 'PAC': 0.8235294117647058, 'PVC': 0.16917293233082706,
             'QAb': 0.007680491551459293, 'RAD': 0.25065963060686014, 'RBBB': 0.7069881487535759,
             'SA': 0.511049723756906, 'SB': 0.6336528221512248, 'SNR': 0.8004385964912281,
             'STach': 0.8166363084395871, 'TAb': 0.16257668711656442, 'TInv': 0.0461133069828722}
results_4 = {'AF': 0.8436724565756824, 'AFL': 0.25, 'Brady': 0.26627218934911245, 'IAVB': 0.6153279292557111,
             'IRBBB': 0.028312570781426953, 'LAnFB': 0.42266604737575475, 'LAD': 0.5695208169677927,
             'LBBB': 0.8467360454115421, 'LPR': 0.11784511784511785, 'LQRSV': 0.1540041067761807,
             'LQT': 0.32710280373831774, 'NSIVCB': 0.042997542997543, 'PR': 0.5102717031146454,
             'PAC': 0.7647058823529411, 'PVC': 0.18199233716475097, 'QAb': 0.015337423312883436,
             'RAD': 0.22727272727272727, 'RBBB': 0.6949013157894737, 'SA': 0.5046728971962616,
             'SB': 0.6473095364944059, 'SNR': 0.8074195308237861, 'STach': 0.8233173076923077,
             'TAb': 0.1447935921133703, 'TInv': 0.033112582781456956}
x = np.arange(len(labels))
width = 0.1
scores_1 = results_1.values()
scores_2 = results_2.values()
scores_3 = results_3.values()
scores_4 = results_4.values()
fig, ax = plt.subplots()
plt.figure(figsize=(40, 25))
rects1 = ax.bar(x - 3 * width / 2, scores_1, width, label='DataBase1')
rects2 = ax.bar(x - width / 2, scores_2, width, label='Database2')
rects3 = ax.bar(x + width / 2, scores_3, width, label='Database3')
rects4 = ax.bar(x + 3 * width / 2, scores_4, width, label='Database4')
ax.set_ylabel('F beta Scores')
ax.set_title('F beta Scores by label and Database')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()
fig.tight_layout()
fig.savefig('/home/david/Incremental_study/Results/Recap_non_fs.png')
if experiment_3 not in results_experiments:
    db1_composition = pd.read_csv('/home/david/Incremental_study/Data/database1_new.csv')['Label'].value_counts()
    db2_composition = pd.read_csv('/home/david/Incremental_study/Data/database2.csv')['Label'].value_counts()
    db3_composition = pd.read_csv('/home/david/Incremental_study/Data/database3.csv')['Label'].value_counts()
    db4_composition = pd.read_csv('/home/david/Incremental_study/Data/database4.csv')
    db5_composition = pd.read_csv('/home/david/Incremental_study/Data/database5.csv')[
        'Label'].value_counts()
    db4_composition['Label'] = db4_composition['Label'].apply(lambda x: 'PVC' if x == 'VPB' else x)
    db4_composition['Label'] = db4_composition['Label'].apply(lambda x: 'PAC' if x == 'SVPB' else x)
    db4_composition['Label'] = db4_composition['Label'].apply(lambda x: 'RBBB' if x == 'CRBBB' else x)
    db4_composition = db4_composition['Label'].value_counts()
    db_composition = pd.concat(
        [db1_composition, db2_composition, db3_composition, db4_composition, db5_composition], axis=1)
    db_composition.replace([np.nan], 0, inplace=True)
    print(db_composition)
    ind = np.arange(len(test_pathologies_effective))  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    plt.figure(figsize=(25, 10))
    print(db_composition.iloc[:, 0].values)
    p1 = plt.bar(ind, db_composition.iloc[:, 0].values, width, color='m')
    p2 = plt.bar(ind, db_composition.iloc[:, 1].values, width,
                 bottom=db_composition.iloc[:, 0].values, color='y')
    p3 = plt.bar(ind, db_composition.iloc[:, 2].values, width,
                 bottom=db_composition.iloc[:, 1].values, color='c')
    p4 = plt.bar(ind, db_composition.iloc[:, 3].values, width,
                 bottom=db_composition.iloc[:, 2].values, color='g')
    p5 = plt.bar(ind, db_composition.iloc[:, 4].values, width,
                 bottom=db_composition.iloc[:, 3].values, color='deeppink')
    plt.tight_layout()
    plt.ylabel('Composition of Databases')
    plt.title('Number of Examples by DataBases')
    plt.xticks(ind, db_composition.index.tolist())
    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('Georgia', 'PTB_XL', 'CPSCA', 'CPSCB', 'PTB'))
    plt.savefig('/home/david/Incremental_study/Data/DataBase.png')
