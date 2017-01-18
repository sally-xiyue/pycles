import numpy as np

profile_data = {}
cgils_pressure = np.array( [1007.31, 1004.42, 1000.42, 995.164, 988.667, 981.041, 972.139, 961.701, 949.7, 936.138, 921.019,
                            904.348, 886.163, 866.541, 845.553, 823.266, 799.767, 775.177, 749.615, 723.184, 695.999, 668.207,
                            639.947, 611.337, 582.504, 553.594, 524.83, 496.493, 468.859, 442.185, 416.712, 392.542, 369.638,
                            347.948, 327.414, 307.982, 289.599, 272.213, 255.778, 240.246, 225.573, 211.716, 198.635, 186.291,
                            174.647, 163.667, 153.317, 143.357, 133.477, 123.596, 113.712, 103.826, 93.9389, 84.0512, 74.1628,
                            64.2747,54.3862, 44.4976, 34.6094, 24.721, 14.8325, 4.9442]) * 100.0

#CGILS STRATUS CONTROL
profile_data['cgils_ctl_s12'] = {}
profile_data['cgils_ctl_s12']['pressure'] = cgils_pressure
profile_data['cgils_ctl_s12']['temperature'] = np.array([289.349, 289.086, 288.748, 288.319, 287.807, 287.255, 286.804,
                                                     287.157, 289.348, 291.414, 293.025, 293.71, 293.86, 293.631,
                                                     293.022, 292.066, 290.786, 289.204, 287.385, 285.424, 283.377,
                                                     281.241, 279.006, 276.65, 274.234, 271.632, 268.8, 265.745, 262.534,
                                                     259.258, 255.979, 252.756, 249.558, 246.351, 243.122, 239.956,
                                                     236.836, 233.747, 230.742, 227.795, 224.955, 222.219, 219.612,
                                                     217.084, 214.65, 212.354, 210.283, 208.461, 206.949, 205.898,
                                                     205.345, 205.463, 206.109, 207.238, 208.727, 210.692, 213.064,
                                                     215.553, 218.216, 222.066, 228.632, 239.84])
profile_data['cgils_ctl_s12']['vapor_mixing_ratio'] = np.array([0.0098892, 0.0097618, 0.00969193, 0.00963604, 0.00957315,
                                                              0.00945821, 0.00920786, 0.00835516, 0.00671145, 0.00547691,
                                                              0.00459523, 0.00414993, 0.00382056, 0.00358184, 0.00347686,
                                                              0.00347723, 0.00350282, 0.00351751, 0.00343955, 0.00321016,
                                                              0.00289939, 0.00262082, 0.00231533, 0.00200742, 0.00169325,
                                                              0.001426, 0.00122999, 0.00108205, 0.000925172, 0.000766179,
                                                              0.000616963, 0.000484263, 0.000378655, 0.000294027, 0.000224721,
                                                              0.000164931, 0.000116218, 8.39562e-05, 6.25927e-05, 4.68341e-05,
                                                              3.46237e-05, 2.5649e-05, 1.89524e-05, 1.4168e-05, 1.076e-05,
                                                              8.22418e-06, 6.5228e-06, 5.44826e-06, 4.63742e-06, 4.05285e-06,
                                                              3.51932e-06, 3.04643e-06, 2.57466e-06, 2.22345e-06, 2.02099e-06,
                                                              1.93765e-06, 1.93356e-06, 1.97504e-06, 2.07047e-06, 2.22209e-06,
                                                              2.45626e-06, 2.84838e-06])

profile_data['cgils_ctl_s12']['o3_mmr'] = np.array([4.61745e-08, 4.62653e-08, 4.63961e-08, 4.6572e-08, 4.67947e-08, 4.7068e-08, 4.74244e-08,
                                                    4.79982e-08, 4.89756e-08, 5.00399e-08, 5.11424e-08, 5.22069e-08, 5.29626e-08, 5.37341e-08,
                                                    5.45318e-08, 5.53668e-08, 5.61766e-08, 5.67656e-08, 5.73945e-08, 5.8088e-08, 5.85218e-08,
                                                    5.88586e-08, 5.92441e-08, 6.02921e-08, 6.1671e-08, 6.31538e-08, 6.46581e-08, 6.6272e-08,
                                                    6.78748e-08, 6.95746e-08, 7.12968e-08, 7.30793e-08, 7.55817e-08, 7.92614e-08, 8.31281e-08,
                                                    8.72225e-08, 9.15533e-08, 9.78727e-08, 1.05005e-07, 1.12673e-07, 1.20976e-07, 1.29804e-07,
                                                    1.39407e-07, 1.49598e-07, 1.58759e-07, 1.67598e-07, 1.78047e-07, 1.92232e-07, 2.08627e-07,
                                                    2.24752e-07, 2.43633e-07, 3.18806e-07, 4.32655e-07, 5.85249e-07, 8.82672e-07, 1.42564e-06,
                                                    2.27646e-06, 3.61159e-06, 5.81991e-06, 8.58005e-06, 1.1482e-05, 9.30123e-06])


#CGILS STRATOCUMULS CONTROL
profile_data['cgils_ctl_s11'] = {}
profile_data['cgils_ctl_s11']['pressure'] = cgils_pressure
profile_data['cgils_ctl_s11']['temperature'] = np.array([291.128, 290.836, 290.482, 290.032, 289.487, 288.861, 288.181, 287.517,
                                                     287.261, 287.925, 289.324, 291.066, 291.858, 291.913, 291.425, 290.514,
                                                     289.26, 287.74, 285.978, 284.018, 281.991, 279.942, 277.745, 275.389,
                                                     272.989, 270.551, 267.894, 265.016, 261.94, 258.744, 255.449, 252.172,
                                                     248.995, 245.823, 242.623, 239.45, 236.309, 233.213, 230.207, 227.278,
                                                     224.465, 221.789, 219.285, 216.937, 214.758, 212.721, 210.81, 209.005,
                                                     207.362, 206.054, 205.28, 205.318, 205.851, 206.894, 208.356, 210.3,
                                                     212.623, 215.169, 217.823, 221.672, 228.112, 239.096])
profile_data['cgils_ctl_s11']['vapor_mixing_ratio'] = np.array([ 0.0105926, 0.0104483, 0.01038, 0.0103323, 0.0102866, 0.0102299,
                                                           0.0101102, 0.00985921, 0.00924009, 0.00799867, 0.00653776, 0.00484959,
                                                           0.00393407, 0.00372152, 0.0037149, 0.00378886, 0.00381136, 0.00369075,
                                                           0.00346277, 0.00315128, 0.00271089, 0.00228915, 0.00202566, 0.00175284,
                                                           0.00139556, 0.00109764, 0.000906571, 0.0008002, 0.000685203, 0.000542333,
                                                           0.000414636, 0.000310628, 0.000234366, 0.000184272, 0.000147089, 0.000117777,
                                                           9.22989e-05, 7.08848e-05, 5.39181e-05, 4.03834e-05, 3.00164e-05, 2.23073e-05,
                                                           1.66849e-05, 1.26969e-05, 9.69327e-06, 7.47974e-06, 5.84852e-06, 4.77429e-06,
                                                           4.08813e-06, 3.7081e-06, 3.38639e-06, 3.02783e-06, 2.53825e-06, 2.17746e-06,
                                                           1.99484e-06, 1.92276e-06, 1.91291e-06, 1.94374e-06, 2.02213e-06, 2.16778e-06,
                                                           2.38648e-06, 2.73844e-06])

profile_data['cgils_ctl_s11']['o3_mmr'] = np.array([4.6735e-08, 4.68224e-08, 4.69526e-08, 4.71274e-08, 4.7348e-08, 4.76129e-08, 4.79359e-08,
                                                    4.83443e-08, 4.89118e-08, 4.97351e-08, 5.0797e-08, 5.2045e-08, 5.29452e-08, 5.3803e-08,
                                                    5.46618e-08, 5.55479e-08, 5.64202e-08, 5.7151e-08, 5.79334e-08, 5.87828e-08, 5.94247e-08,
                                                    6.00199e-08, 6.06758e-08, 6.19609e-08, 6.35875e-08, 6.53784e-08, 6.72885e-08, 6.93387e-08,
                                                    7.15374e-08, 7.39255e-08, 7.63188e-08, 7.87406e-08, 8.19122e-08, 8.64443e-08, 9.12007e-08,
                                                    9.60567e-08, 1.01148e-07, 1.08904e-07, 1.18266e-07, 1.28124e-07, 1.38057e-07, 1.48628e-07,
                                                    1.62026e-07, 1.76791e-07, 1.91203e-07, 2.06033e-07, 2.22701e-07, 2.40911e-07, 2.61669e-07,
                                                    2.84505e-07, 3.12341e-07, 3.97212e-07, 5.23519e-07, 6.95588e-07, 1.01202e-06, 1.56304e-06,
                                                    2.41166e-06, 3.72114e-06, 5.82051e-06, 8.45051e-06, 1.12784e-05, 9.2692e-06 ])

# CGILS CUMULUS CONTROL
profile_data['cgils_ctl_s6'] = {}
profile_data['cgils_ctl_s6']['pressure'] = cgils_pressure
profile_data['cgils_ctl_s6']['temperature'] = np.array([  298.04, 297.755, 297.401, 296.95, 296.397, 295.744, 295.021,
                                                      294.25, 293.549, 292.824, 292.058, 291.269, 290.461, 289.764,
                                                      289.483, 289.13, 288.54, 287.644, 286.345, 284.861, 283.32,
                                                      281.587, 279.648, 277.592, 275.581, 273.615, 271.469, 268.918,
                                                      265.953, 262.699, 259.305, 255.826, 252.305, 248.776, 245.338,
                                                      241.972, 238.59, 235.182, 231.76, 228.333, 224.956, 221.637, 218.414,
                                                      215.306, 212.363, 209.544, 206.807, 204.1, 201.455, 198.931, 196.866,
                                                      196.093, 196.925, 199.751, 203.153, 206.216, 208.919, 212.221,
                                                      216.025, 220.251, 226.196, 236.265])
profile_data['cgils_ctl_s6']['vapor_mixing_ratio'] = np.array([  0.0157317, 0.0154838, 0.0153586, 0.0152672, 0.0151792, 0.0150708,
                                                             0.0148761, 0.0145738, 0.0141048, 0.0134474, 0.0127099, 0.0119752,
                                                             0.0111656, 0.0101413, 0.00854392, 0.00707964, 0.00566332, 0.00463409,
                                                             0.00397835, 0.00346589, 0.00288315, 0.00243627, 0.00209491, 0.00170026,
                                                             0.00124933, 0.00086599, 0.000648771, 0.000536331, 0.00049462, 0.00048766,
                                                             0.00045086, 0.000396325, 0.000341939, 0.000288786, 0.000235559, 0.000187447,
                                                             0.000140483, 0.000107742, 8.50189e-05, 6.74462e-05, 5.3237e-05, 4.15715e-05,
                                                             3.22764e-05, 2.47352e-05, 1.86647e-05, 1.41949e-05, 1.07417e-05, 8.16138e-06,
                                                             6.15557e-06, 4.6722e-06, 3.60039e-06, 3.01648e-06, 2.78834e-06, 2.6918e-06,
                                                             2.50043e-06, 2.20898e-06, 1.98767e-06, 1.90829e-06, 1.89275e-06, 1.91898e-06,
                                                             2.0406e-06, 2.37104e-06])

profile_data['cgils_ctl_s6']['o3_mmr'] = np.array([ 4.92601e-08, 4.93545e-08, 4.94932e-08, 4.96792e-08, 4.99124e-08, 5.01896e-08,
                                                    5.05255e-08, 5.09403e-08, 5.14613e-08, 5.20777e-08, 5.27941e-08, 5.36222e-08,
                                                    5.44012e-08, 5.53093e-08, 5.64189e-08, 5.76486e-08, 5.8965e-08, 6.04948e-08,
                                                    6.21127e-08, 6.3876e-08, 6.56778e-08, 6.75811e-08, 6.96473e-08, 7.22517e-08,
                                                    7.52788e-08, 7.86392e-08, 8.26171e-08, 8.68405e-08, 9.20066e-08, 9.78419e-08,
                                                    1.03709e-07, 1.09455e-07, 1.15938e-07, 1.24688e-07, 1.33897e-07, 1.42634e-07,
                                                    1.5159e-07, 1.66572e-07, 1.87041e-07, 2.07624e-07, 2.25271e-07, 2.43833e-07,
                                                    2.7506e-07, 3.1091e-07, 3.4908e-07, 3.90834e-07, 4.3532e-07, 4.70482e-07, 5.09847e-07,
                                                    5.62283e-07, 6.29374e-07, 7.54855e-07, 9.38036e-07, 1.20892e-06, 1.62498e-06, 2.21934e-06,
                                                    3.05698e-06, 4.24212e-06, 5.82742e-06, 7.82845e-06, 1.03038e-05, 9.1438e-06])



profile_data['cgils_p2_s12'] = {}
profile_data['cgils_p2_s12']['pressure'] = cgils_pressure
profile_data['cgils_p2_s12']['temperature'] = np.array([ 291.349, 291.086, 290.748, 290.319, 289.807, 289.255, 288.804, 289.17, 291.38, 293.465, 295.095,
                                                         295.802, 295.977, 295.777, 295.2, 294.283, 293.05, 291.524, 289.771, 287.886, 285.925, 283.889,
                                                         281.768, 279.542, 277.272, 274.838, 272.198, 269.36, 266.388, 263.368, 260.355, 257.402, 254.476,
                                                         251.5, 248.134, 244.84, 241.598, 238.393, 235.278, 232.228, 229.29, 226.462, 223.768, 221.158,
                                                         218.646, 216.277, 214.138, 212.249, 210.671, 209.555, 208.937, 208.988, 209.215, 209.702, 210.548,
                                                         211.87, 213.599, 215.445, 217.466, 220.673, 226.596, 237.161])
profile_data['cgils_p2_s12']['vapor_mixing_ratio'] = np.array([ 0.0112509, 0.0111085, 0.0110322, 0.0109727, 0.010906, 0.0107803, 0.0104993, 0.00953267,
                                                                0.00765263, 0.00624227, 0.00523746, 0.00473427, 0.00436517, 0.00410081, 0.00399122,
                                                                0.00400503, 0.00405126, 0.00408894, 0.00402268, 0.0037811, 0.00344296, 0.00314134,
                                                                0.00280511, 0.00246239, 0.00210648, 0.00180346, 0.00158599, 0.00142707, 0.00125195,
                                                                0.00106675, 0.000885872, 0.000718379, 0.000581356, 0.000466407, 0.00035694, 0.000262403,
                                                                0.000185282, 0.000134185, 0.000100314, 7.52891e-05, 5.58362e-05, 4.14975e-05, 3.07609e-05,
                                                                2.30727e-05, 1.7583e-05, 1.3483e-05, 1.07212e-05, 8.9687e-06, 7.6326e-06, 6.65159e-06,
                                                                5.74316e-06, 4.92483e-06, 3.92286e-06, 3.09571e-06, 2.5729e-06, 2.2589e-06, 2.06994e-06,
                                                                1.94876e-06, 1.89017e-06, 1.88754e-06, 1.96174e-06, 2.14669e-06 ])

profile_data['cgils_p2_s12']['o3_mmr'] = np.array([ 4.61745e-08, 4.62653e-08, 4.63961e-08, 4.6572e-08, 4.67947e-08, 4.7068e-08, 4.74244e-08, 4.79982e-08, 4.89756e-08,
                                                    5.00399e-08, 5.11424e-08, 5.22069e-08, 5.29626e-08, 5.37341e-08, 5.45318e-08, 5.53668e-08, 5.61766e-08, 5.67656e-08,
                                                    5.73945e-08, 5.8088e-08, 5.85218e-08, 5.88586e-08, 5.92441e-08, 6.02921e-08, 6.1671e-08, 6.31538e-08, 6.46581e-08,
                                                    6.6272e-08, 6.78748e-08, 6.95746e-08, 7.12968e-08, 7.30793e-08, 7.55817e-08, 7.92614e-08, 8.31281e-08, 8.72225e-08,
                                                    9.15533e-08, 9.78727e-08, 1.05005e-07, 1.12673e-07, 1.20976e-07, 1.29804e-07, 1.39407e-07, 1.49598e-07, 1.58759e-07,
                                                    1.67598e-07, 1.78047e-07, 1.92232e-07, 2.08627e-07, 2.24752e-07, 2.43633e-07, 3.18806e-07, 4.32655e-07, 5.85249e-07,
                                                    8.82672e-07, 1.42564e-06, 2.27646e-06, 3.61159e-06, 5.81991e-06, 8.58005e-06, 1.1482e-05, 9.30123e-06 ])


profile_data['cgils_p2_s11'] = {}
profile_data['cgils_p2_s11']['pressure'] = cgils_pressure
profile_data['cgils_p2_s11']['temperature'] = np.array([293.128, 292.836, 292.482, 292.032, 291.487, 290.861, 290.181, 289.517, 289.278,
                                                        289.964, 291.386, 293.151, 293.968, 294.053, 293.6, 292.729, 291.523, 290.06,
                                                        288.365, 286.484, 284.545, 282.596, 280.514, 278.29, 276.036, 273.762, 271.293,
                                                        268.625, 265.782, 262.836, 259.803, 256.792, 253.881, 250.934, 247.601, 244.302,
                                                        241.041, 237.832, 234.72, 231.689, 228.781, 226.015, 223.426, 220.998, 218.744,
                                                        216.635, 214.656, 212.787, 211.079, 209.707, 208.869, 208.843, 208.957, 209.357,
                                                        210.176, 211.478, 213.158, 215.061, 217.073, 220.279, 226.076, 236.417])
profile_data['cgils_p2_s11']['vapor_mixing_ratio'] = np.array([ 0.0120338, 0.0118728, 0.0117986, 0.0117489, 0.0117023, 0.011644, 0.0115147,
                                                                0.0112354, 0.0105436, 0.00913567, 0.00746987, 0.00554145, 0.00450002, 0.00426498,
                                                                0.00426881, 0.00436888, 0.00441375, 0.00429627, 0.00405592, 0.00371819, 0.00322538,
                                                                0.00274929, 0.00245947, 0.00215535, 0.00174078, 0.00139143, 0.00117108, 0.00105654,
                                                                0.000927655, 0.000755053, 0.000595248, 0.000460711, 0.000359597, 0.000291973, 0.000233391,
                                                                0.000187238, 0.000147083, 0.000113273, 8.64163e-05, 6.49325e-05, 4.84208e-05, 3.60993e-05,
                                                                2.70788e-05, 2.06609e-05, 1.58093e-05, 1.22241e-05, 9.57589e-06, 7.82923e-06, 6.70872e-06,
                                                                6.07764e-06, 5.52669e-06, 4.89821e-06, 3.87212e-06, 3.03555e-06, 2.54215e-06, 2.24303e-06,
                                                                2.0485e-06, 1.91776e-06, 1.84534e-06, 1.8402e-06, 1.90392e-06, 2.06264e-06])

profile_data['cgils_p2_s11']['o3_mmr'] = np.array([4.6735e-08, 4.68224e-08, 4.69526e-08, 4.71274e-08, 4.7348e-08, 4.76129e-08, 4.79359e-08, 4.83443e-08,
                                                   4.89118e-08, 4.97351e-08, 5.0797e-08, 5.2045e-08, 5.29452e-08, 5.3803e-08, 5.46618e-08, 5.55479e-08,
                                                   5.64202e-08, 5.7151e-08, 5.79334e-08, 5.87828e-08, 5.94247e-08, 6.00199e-08, 6.06758e-08, 6.19609e-08,
                                                   6.35875e-08, 6.53784e-08, 6.72885e-08, 6.93387e-08, 7.15374e-08, 7.39255e-08, 7.63188e-08, 7.87406e-08,
                                                   8.19122e-08, 8.64443e-08, 9.12007e-08, 9.60567e-08, 1.01148e-07, 1.08904e-07, 1.18266e-07, 1.28124e-07,
                                                   1.38057e-07, 1.48628e-07, 1.62026e-07, 1.76791e-07, 1.91203e-07, 2.06033e-07, 2.22701e-07, 2.40911e-07,
                                                   2.61669e-07, 2.84505e-07, 3.12341e-07, 3.97212e-07, 5.23519e-07, 6.95588e-07, 1.01202e-06, 1.56304e-06,
                                                   2.41166e-06, 3.72114e-06, 5.82051e-06, 8.45051e-06, 1.12784e-05, 9.2692e-06 ])

profile_data['cgils_p2_s6'] = {}
profile_data['cgils_p2_s6']['pressure'] = cgils_pressure
profile_data['cgils_p2_s6']['temperature'] = np.array([300.04, 299.755, 299.401, 298.95, 298.397, 297.744, 297.021, 296.25,
                                                        295.549, 294.843, 294.101, 293.339, 292.562, 291.899, 291.654, 291.341,
                                                        290.795, 289.95, 288.712, 287.298, 285.834, 284.192, 282.357, 280.42,
                                                        278.54, 276.717, 274.733, 272.375, 269.636, 266.636, 263.515, 260.325,
                                                        257.096, 253.819, 250.253, 246.766, 243.269, 239.754, 236.229, 232.706,
                                                        229.237, 225.832, 222.527, 219.343, 216.327, 213.44, 210.639, 207.869,
                                                        205.163, 202.577, 200.451, 199.616, 200.031, 202.214, 204.974, 207.394,
                                                        209.454, 212.114, 215.275, 218.858, 224.16, 233.587])
profile_data['cgils_p2_s6']['vapor_mixing_ratio'] =np.array([0.0177827, 0.017506, 0.0173688, 0.017271, 0.0171784, 0.017064, 0.0168528,
                                                              0.01652, 0.0159973, 0.0152793, 0.0144719, 0.0136679, 0.0127782, 0.0116392,
                                                              0.00983176, 0.00817067, 0.00655891, 0.00538976, 0.00465179, 0.00407789,
                                                              0.00341607, 0.00291033, 0.00252677, 0.00207359, 0.00154232, 0.00108345,
                                                              0.000824364, 0.00069454, 0.000655469, 0.000663949, 0.000632781, 0.000575092,
                                                              0.000514277, 0.000449512, 0.000367553, 0.000293289, 0.000220568, 0.000169875,
                                                              0.000134712, 0.000107474, 8.53539e-05, 6.70914e-05, 5.24483e-05, 4.04764e-05,
                                                              3.07528e-05, 2.35497e-05, 1.79481e-05, 1.37392e-05, 1.04395e-05, 7.97825e-06,
                                                              6.17322e-06, 5.15048e-06, 4.45282e-06, 3.86001e-06, 3.23345e-06, 2.59546e-06,
                                                              2.13461e-06, 1.88197e-06, 1.72424e-06, 1.62508e-06, 1.62125e-06, 1.78029e-06])



profile_data['cgils_p2_s6']['o3_mmr'] =np.array([4.92601e-08, 4.93545e-08, 4.94932e-08, 4.96792e-08, 4.99124e-08, 5.01896e-08, 5.05255e-08,
                                                 5.09403e-08, 5.14613e-08, 5.20777e-08, 5.27941e-08, 5.36222e-08, 5.44012e-08, 5.53093e-08,
                                                 5.64189e-08, 5.76486e-08, 5.8965e-08, 6.04948e-08, 6.21127e-08, 6.3876e-08, 6.56778e-08,
                                                 6.75811e-08, 6.96473e-08, 7.22517e-08, 7.52788e-08, 7.86392e-08, 8.26171e-08, 8.68405e-08,
                                                 9.20066e-08, 9.78419e-08, 1.03709e-07, 1.09455e-07, 1.15938e-07, 1.24688e-07, 1.33897e-07,
                                                 1.42634e-07, 1.5159e-07, 1.66572e-07, 1.87041e-07, 2.07624e-07, 2.25271e-07, 2.43833e-07,
                                                 2.7506e-07, 3.1091e-07, 3.4908e-07, 3.90834e-07, 4.3532e-07, 4.70482e-07, 5.09847e-07,
                                                 5.62283e-07, 6.29374e-07, 7.54855e-07, 9.38036e-07, 1.20892e-06, 1.62498e-06, 2.21934e-06,
                                                 3.05698e-06, 4.24212e-06, 5.82742e-06, 7.82845e-06, 1.03038e-05, 9.1438e-06])


# Arctic profiles
profile_data['arctic'] = {}
profile_data['arctic']['pressure'] = np.array([1.00575873e5,   1.00278284e5,   9.98399536e4,   9.92184082e4,
                                               9.83804260e4,   9.73009582e4,   9.59621155e4,   9.43535889e4,
                                               9.24715210e4,   9.03178955e4,   8.79002075e4,   8.52306702e4,
                                               8.23262024e4,   7.92071045e4,   7.58969238e4,   7.24222778e4,
                                               6.88113770e4,   6.50942261e4,   6.13018860e4,   5.74658386e4,
                                               5.36173767e4,   4.97872192e4,   4.60047852e4,   4.22979370e4,
                                               3.86919434e4,   3.52094727e4,   3.18696625e4,   2.86879181e4,
                                               2.56749146e4,   2.28361450e4,   2.01715500e4,   1.76832245e4,
                                               1.53775101e4,   1.32585281e4,   1.13282722e4,   9.58641815e3,
                                               8.03020325e3,   6.65455856e3,   5.45598755e3,   4.42831612e3,
                                               3.57421494e3,   2.88478088e3,   2.32831478e3,   1.87929726e3,
                                               1.51679745e3,   1.22420835e3,   9.88086700e2,   7.97505760e2,
                                               6.43681290e2,   5.19531150e2,   4.18798970e2,   3.35839870e2,
                                               2.66320660e2,   2.07438110e2,   1.57348970e2,   1.14929540e2,
                                               7.95491400e1,   5.09767600e1,   2.91778800e1,   9.98834000e0])
profile_data['arctic']['temperature'] = np.array([266.8943176, 267.0693359, 267.3087158, 267.5513916, 267.6365662,
                                                 267.4140015, 266.8858337, 266.1845093, 265.3601990, 264.5728455,
                                                 263.9873962, 263.5353088, 262.6596069, 261.3655701, 259.7712097,
                                                 257.8913879, 255.7699127, 253.3448029, 250.6314545, 247.6959229,
                                                 244.4879303, 240.9742279, 237.2752991, 233.4430847, 229.6099701,
                                                 225.9502258, 222.7632751, 220.5044250, 219.6378784, 220.4424591,
                                                 222.1228943, 223.3230896, 223.6871643, 223.5032501, 223.3511658,
                                                 223.3572693, 222.9019165, 221.5261230, 219.9896393, 218.5794678,
                                                 217.5787201, 217.0293579, 216.6041870, 216.1675110, 215.6414032,
                                                 215.1298370, 214.7327423, 215.0572052, 216.5581055, 219.0173950,
                                                 221.4697571, 223.0003967, 223.6576843, 224.3420258, 226.3999939,
                                                 230.8148956, 237.6799164, 244.1721497, 245.6276703, 238.1793976])
profile_data['arctic']['specific_humidity'] = np.array([2.03999292e-3,   2.07812231e-3,   2.13174864e-3,   2.20352874e-3,
                                                        2.26384269e-3,   2.27144509e-3,   2.23518157e-3,   2.15544183e-3,
                                                        2.05782512e-3,   1.92552052e-3,   1.77560851e-3,   1.63220366e-3,
                                                        1.48773784e-3,   1.32572761e-3,   1.16327132e-3,   1.00650553e-3,
                                                        8.66593564e-4,   7.30944929e-4,   6.02148798e-4,   4.82887407e-4,
                                                        3.72127570e-4,   2.69140044e-4,   1.90601564e-4,   1.34933990e-4,
                                                        9.35404494e-5,   6.32730963e-5,   4.19374412e-5,   2.68939767e-5,
                                                        1.80862729e-5,   1.19694567e-5,   6.16606198e-6,   3.18518985e-6,
                                                        2.30199470e-6,   2.18249524e-6,   2.18069524e-6,   2.18809521e-6,
                                                        2.18759521e-6,   2.18679522e-6,   2.19889516e-6,   2.24759495e-6,
                                                        2.32919457e-6,   2.42719411e-6,   2.53789356e-6,   2.63449306e-6,
                                                        2.73629251e-6,   2.84559190e-6,   2.98549109e-6,   3.12929021e-6,
                                                        3.26818932e-6,   3.40558840e-6,   3.53848748e-6,   3.64988668e-6,
                                                        3.75878587e-6,   3.88308492e-6,   3.96388429e-6,   4.00488396e-6,
                                                        4.01368389e-6,   3.99578403e-6,   3.96328429e-6,   3.93488452e-6])
profile_data['arctic']['vapor_mixing_ratio'] = profile_data['arctic']['specific_humidity'] / (1.0 -
                                                                    profile_data['arctic']['specific_humidity'])
profile_data['arctic']['o3_mr'] = np.array([2.29865000e-5,   2.30413000e-5,   2.31245000e-5,   2.32422000e-5,
                                            2.33976000e-5,   2.36059000e-5,   2.38659000e-5,   2.41841000e-5,
                                            2.45592000e-5,   2.49994000e-5,   2.55411000e-5,   2.61804000e-5,
                                            2.68984000e-5,   2.76856000e-5,   2.80097000e-5,   2.82597000e-5,
                                            2.85266000e-5,   2.87814000e-5,   2.90502000e-5,   3.03922000e-5,
                                            3.20686000e-5,   3.36867000e-5,   3.55035000e-5,   3.99821000e-5,
                                            4.48590000e-5,   5.01222000e-5,   6.56444000e-5,   8.59159000e-5,
                                            1.14569600e-4,   1.61802800e-4,   2.21155900e-4,   2.83780500e-4,
                                            3.56959900e-4,   4.57441900e-4,   6.14007900e-4,   8.55658500e-4,
                                            1.17563900e-3,   1.61028430e-3,   2.15759020e-3,   2.84742160e-3,
                                            3.69521780e-3,   4.24513960e-3,   4.68018010e-3,   5.03616810e-3,
                                            5.41168980e-3,   5.82303240e-3,   6.20196440e-3,   6.48212480e-3,
                                            6.64787910e-3,   6.66918370e-3,   6.50225020e-3,   6.11083460e-3,
                                            5.51963280e-3,   4.80870440e-3,   3.99442720e-3,   3.06811450e-3,
                                            2.24217030e-3,   1.52163050e-3,   9.70994100e-4,   4.19090500e-4])