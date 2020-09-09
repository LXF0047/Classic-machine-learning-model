from utils.utils import draw_from_dict

col = ['ack_psh_rst_syn_fin_cnt_0', 'ack_psh_rst_syn_fin_cnt_1', 'ack_psh_rst_syn_fin_cnt_2',
       'ack_psh_rst_syn_fin_cnt_3', 'ack_psh_rst_syn_fin_cnt_4', 'bytes_in', 'bytes_out', 'dst_port', 'hdr_bin_40',
       'hdr_ccnt_0', 'hdr_ccnt_1', 'hdr_ccnt_2', 'hdr_ccnt_3', 'hdr_ccnt_4', 'hdr_ccnt_5', 'hdr_ccnt_6', 'hdr_ccnt_7',
       'hdr_ccnt_8', 'hdr_ccnt_9', 'hdr_ccnt_10', 'hdr_ccnt_11', 'hdr_distinct', 'hdr_mean', 'intervals_ccnt_0',
       'intervals_ccnt_1', 'intervals_ccnt_2', 'intervals_ccnt_3', 'intervals_ccnt_4', 'intervals_ccnt_5',
       'intervals_ccnt_6', 'intervals_ccnt_7', 'intervals_ccnt_8', 'intervals_ccnt_9', 'intervals_ccnt_10',
       'intervals_ccnt_11', 'intervals_ccnt_12', 'intervals_ccnt_13', 'intervals_ccnt_14', 'intervals_ccnt_15',
       'num_pkts_in', 'num_pkts_out', 'pld_bin_inf', 'pld_ccnt_0', 'pld_ccnt_1', 'pld_ccnt_2', 'pld_ccnt_3',
       'pld_ccnt_4', 'pld_ccnt_5', 'pld_ccnt_6', 'pld_ccnt_7', 'pld_ccnt_8', 'pld_ccnt_9', 'pld_ccnt_10', 'pld_ccnt_11',
       'pld_ccnt_12', 'pld_ccnt_13', 'pld_ccnt_14', 'pld_ccnt_15', 'pld_distinct', 'pld_max', 'pld_mean', 'pld_median',
       'pr', 'rev_ack_psh_rst_syn_fin_cnt_0', 'rev_ack_psh_rst_syn_fin_cnt_1', 'rev_ack_psh_rst_syn_fin_cnt_2',
       'rev_ack_psh_rst_syn_fin_cnt_3', 'rev_ack_psh_rst_syn_fin_cnt_4', 'rev_hdr_bin_40', 'rev_hdr_ccnt_0',
       'rev_hdr_ccnt_1', 'rev_hdr_ccnt_2', 'rev_hdr_ccnt_3', 'rev_hdr_ccnt_4', 'rev_hdr_ccnt_5', 'rev_hdr_ccnt_6',
       'rev_hdr_ccnt_7', 'rev_hdr_ccnt_8', 'rev_hdr_ccnt_9', 'rev_hdr_ccnt_10', 'rev_hdr_ccnt_11', 'rev_hdr_distinct',
       'rev_intervals_ccnt_0', 'rev_intervals_ccnt_1', 'rev_intervals_ccnt_2', 'rev_intervals_ccnt_3',
       'rev_intervals_ccnt_4', 'rev_intervals_ccnt_5', 'rev_intervals_ccnt_6', 'rev_intervals_ccnt_7',
       'rev_intervals_ccnt_8', 'rev_intervals_ccnt_9', 'rev_intervals_ccnt_10', 'rev_intervals_ccnt_11',
       'rev_intervals_ccnt_12', 'rev_intervals_ccnt_13', 'rev_intervals_ccnt_14', 'rev_intervals_ccnt_15',
       'rev_pld_bin_128', 'rev_pld_ccnt_0', 'rev_pld_ccnt_1', 'rev_pld_ccnt_2', 'rev_pld_ccnt_3', 'rev_pld_ccnt_4',
       'rev_pld_ccnt_5', 'rev_pld_ccnt_6', 'rev_pld_ccnt_7', 'rev_pld_ccnt_8', 'rev_pld_ccnt_9', 'rev_pld_ccnt_10',
       'rev_pld_ccnt_11', 'rev_pld_ccnt_12', 'rev_pld_ccnt_13', 'rev_pld_ccnt_14', 'rev_pld_ccnt_15',
       'rev_pld_distinct', 'rev_pld_max', 'rev_pld_mean', 'rev_pld_var', 'src_port', 'time_length', 'label', 'id']
lgb = [  47, 29, 34, 13, 27, 498, 567, 650, 44, 95, 83,  0,  3, 74
,  0, 18,  2,  0,  0,  0,  0,  2, 62, 90, 57, 25, 16, 16
,  2,  0, 41,  3,  8,  4, 34,  1,  0,  3, 48, 90, 75,  5
, 60, 33, 19, 26, 15,  4, 23,  8, 14,  1,  1,  0,  4,  3
,  8,  3, 33, 397, 296, 335,  0, 26, 16, 35, 13, 21, 58, 63
, 47, 41,  2, 46,  0,  1,  0,  0,  0,  0,  0,  9, 54, 28
, 11, 16,  0,  0,  0,  9,  1,  3,  4,  8,  0,  0,  2, 24
, 86, 38, 43, 44, 39, 39, 40,  6,  8,  7,  0,  3,  3,  0
,  0,  0,  7, 39, 217, 234, 157, 1834, 969]
ngb = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.51598768e-04
    , 2.63314411e-04, 1.35343229e-03, 5.25595960e-04, 2.83473463e-01
    , 1.36354531e-02, 0.00000000e+00, 2.13967244e-08, 3.73264239e-04
    , 6.70069211e-03, 7.27118223e-04, 0.00000000e+00, 4.53497037e-02
    , 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.85846141e-02
    , 4.63074860e-03, 0.00000000e+00, 1.64099692e-06, 0.00000000e+00
    , 2.06280679e-06, 1.01570793e-06, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 2.07809658e-06, 1.01793201e-06, 9.74382665e-04
    , 2.35486397e-06, 0.00000000e+00, 6.67775162e-04, 1.65296727e-02
    , 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.19392698e-03
    , 9.54405931e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 0.00000000e+00, 4.09433012e-06, 1.48599759e-02
    , 1.79984363e-03, 5.24503560e-03, 0.00000000e+00, 2.03169299e-08
    , 1.26481632e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 3.84176643e-03, 1.18869804e-01, 6.64781588e-03, 3.56996219e-01
    , 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 3.30833398e-06, 0.00000000e+00, 0.00000000e+00
    , 5.13615341e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 1.00185052e-06, 0.00000000e+00, 0.00000000e+00
    , 4.80193005e-04, 0.00000000e+00, 2.37307805e-06, 0.00000000e+00
    , 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.00130202e-03
    , 4.49956054e-02, 5.83246489e-03, 3.27280995e-03, 8.09190964e-03
    , 1.12522880e-02]
rf = [1.00554731e-02, 2.23587344e-03, 6.28147970e-04, 6.58398263e-03
    , 1.30799688e-03, 3.96692241e-02, 2.06457105e-02, 5.01079412e-02
    , 1.29047031e-02, 5.00396200e-03, 1.24194861e-02, 3.07054451e-06
    , 4.44766852e-03, 1.52040964e-02, 4.66545714e-05, 8.01575031e-03
    , 4.33315787e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 5.62519740e-07, 5.56190874e-03, 3.74434538e-02, 3.95485234e-02
    , 6.79765008e-03, 5.75385991e-03, 3.37708599e-05, 5.81683542e-05
    , 1.47313003e-05, 1.95661070e-05, 2.41927192e-05, 9.21659036e-06
    , 9.41800218e-06, 3.05450406e-05, 1.75934605e-05, 2.44840514e-05
    , 5.25313666e-06, 5.20509498e-05, 2.91751088e-03, 2.40816659e-02
    , 2.29835767e-02, 5.80158579e-04, 1.52079117e-02, 1.90777209e-03
    , 2.91763190e-04, 1.90834779e-03, 3.02255218e-04, 3.45540814e-04
    , 4.44310471e-04, 1.52455200e-04, 1.65232069e-03, 1.32181717e-05
    , 1.49874947e-05, 1.73287703e-05, 6.66621199e-06, 1.23999386e-04
    , 1.11559611e-05, 8.08943775e-04, 6.06662528e-03, 1.90074475e-02
    , 3.50426245e-02, 2.25693787e-02, 5.77222108e-03, 1.75274023e-02
    , 6.79643974e-03, 1.78863110e-03, 4.69935710e-03, 4.44309403e-03
    , 3.86978020e-02, 6.53094789e-02, 3.23626409e-02, 6.06011960e-02
    , 7.77111999e-04, 2.83365478e-02, 2.01446948e-03, 8.24339009e-04
    , 2.21388759e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
    , 0.00000000e+00, 2.29471475e-02, 7.87552520e-03, 1.41842881e-02
    , 6.71102081e-03, 6.92653457e-05, 3.81536164e-05, 4.42144033e-05
    , 1.30214970e-05, 2.07151370e-05, 6.33798441e-06, 2.39657764e-06
    , 3.23992015e-06, 1.22677405e-05, 5.36919263e-06, 3.10848875e-06
    , 1.05181734e-04, 1.84099275e-03, 1.60329408e-02, 1.57900219e-02
    , 9.15864064e-03, 3.60228161e-03, 3.28153035e-03, 9.95098252e-04
    , 5.45174693e-04, 1.66512924e-04, 6.99607356e-05, 2.52176882e-05
    , 1.00020361e-05, 4.37101259e-04, 9.90568453e-06, 9.77597506e-06
    , 1.80868253e-05, 1.36244191e-05, 1.26752106e-03, 1.93679707e-02
    , 6.48703761e-02, 4.73119133e-02, 8.20296244e-03, 1.11388132e-02
    , 3.26533896e-02]

cb_d = {'rev_hdr_ccnt_2': 35.786829,
        'dst_port': 15.538599,
       'rev_hdr_ccnt_0': 10.127941,
       'rev_pld_max': 5.185222,
       'hdr_ccnt_6': 4.761911}

lgb_d = dict(zip(col, [int(x) for x in lgb]))
ngb_d = dict(zip(col, [float(x) for x in ngb]))
rf_d = dict(zip(col, [x for x in rf]))
# print(lgb_d)

topk = 30
# xgb_f = [x[0] for x in sorted(xgb.items(), key=lambda item: item[1], reverse=True)[:topk]]
lgb_f = [x[0] for x in sorted(lgb_d.items(), key=lambda item: item[1], reverse=True)[:topk]]
ngb_f = [x[0] for x in sorted(ngb_d.items(), key=lambda item: item[1], reverse=True)[:topk]]
rf_f = [x[0] for x in sorted(rf_d.items(), key=lambda item: item[1], reverse=True)[:topk]]
#
# print(list(set(lgb_f + ngb_f + rf_f)))
# xgb = {'pld_max': 1354, 'rev_ack_psh_rst_syn_fin_cnt_4': 64, 'ack_psh_rst_syn_fin_cnt_3': 94, 'rev_pld_ccnt_1': 189, 'pld_median': 1323, 'src_port': 39306, 'bytes_out': 3880, 'rev_pld_bin_128': 225, 'rev_pld_distinct': 88, 'rev_pld_var': 392, 'dst_port': 4917, 'pld_distinct': 146, 'pld_ccnt_12': 9, 'pld_ccnt_9': 27, 'rev_intervals_ccnt_4': 20, 'pld_ccnt_0': 379, 'ack_psh_rst_syn_fin_cnt_2': 78, 'pld_mean': 1233, 'time_length': 24842, 'rev_pld_mean': 1986, 'bytes_in': 2050, 'pld_ccnt_3': 84, 'rev_pld_ccnt_0': 165, 'ack_psh_rst_syn_fin_cnt_0': 223, 'rev_pld_max': 996, 'rev_hdr_bin_40': 100, 'intervals_ccnt_14': 55, 'intervals_ccnt_0': 553, 'intervals_ccnt_11': 89, 'rev_hdr_ccnt_0': 120, 'hdr_ccnt_0': 505, 'intervals_ccnt_7': 88, 'hdr_ccnt_1': 162, 'rev_pld_ccnt_7': 25, 'rev_intervals_ccnt_15': 49, 'hdr_bin_40': 188, 'rev_intervals_ccnt_1': 52, 'rev_intervals_ccnt_5': 18, 'hdr_ccnt_4': 85, 'ack_psh_rst_syn_fin_cnt_1': 78, 'intervals_ccnt_6': 19, 'intervals_ccnt_1': 285, 'rev_hdr_ccnt_2': 31, 'num_pkts_in': 296, 'rev_pld_ccnt_2': 165, 'rev_ack_psh_rst_syn_fin_cnt_1': 62, 'num_pkts_out': 366, 'rev_ack_psh_rst_syn_fin_cnt_2': 80, 'pld_ccnt_8': 39, 'hdr_mean': 217, 'rev_pld_ccnt_5': 134, 'pld_ccnt_2': 148, 'pld_bin_inf': 27, 'rev_intervals_ccnt_0': 137, 'intervals_ccnt_13': 27, 'intervals_ccnt_15': 255, 'intervals_ccnt_3': 90, 'rev_pld_ccnt_11': 5, 'intervals_ccnt_2': 165, 'hdr_distinct': 17, 'rev_hdr_ccnt_4': 84, 'rev_pld_ccnt_3': 200, 'ack_psh_rst_syn_fin_cnt_4': 89, 'pld_ccnt_14': 8, 'rev_ack_psh_rst_syn_fin_cnt_0': 87, 'pld_ccnt_7': 33, 'intervals_ccnt_4': 88, 'pld_ccnt_1': 193, 'intervals_ccnt_9': 64, 'pld_ccnt_6': 36, 'rev_hdr_ccnt_1': 111, 'pld_ccnt_4': 61, 'intervals_ccnt_10': 40, 'rev_pld_ccnt_14': 11, 'rev_pld_ccnt_8': 23, 'rev_pld_ccnt_13': 4, 'rev_intervals_ccnt_2': 41, 'hdr_ccnt_6': 42, 'rev_pld_ccnt_4': 130, 'rev_hdr_distinct': 12, 'pld_ccnt_5': 33, 'rev_intervals_ccnt_11': 43, 'rev_pld_ccnt_6': 27, 'pld_ccnt_10': 24, 'intervals_ccnt_12': 30, 'rev_hdr_ccnt_3': 4, 'rev_pld_ccnt_15': 26, 'hdr_ccnt_3': 14, 'pld_ccnt_13': 10, 'intervals_ccnt_5': 57, 'rev_intervals_ccnt_3': 35, 'rev_intervals_ccnt_14': 6, 'rev_intervals_ccnt_7': 30, 'rev_intervals_ccnt_10': 21, 'pld_ccnt_11': 5, 'intervals_ccnt_8': 62, 'rev_ack_psh_rst_syn_fin_cnt_3': 18, 'rev_pld_ccnt_12': 2, 'pld_ccnt_15': 7, 'hdr_ccnt_7': 6, 'rev_hdr_ccnt_6': 4, 'rev_intervals_ccnt_6': 1, 'rev_intervals_ccnt_9': 5, 'rev_intervals_ccnt_13': 2, 'rev_pld_ccnt_9': 1}
# draw_from_dict(lgb_d, axis=1)
print(lgb_f)
# print(lgb_f)