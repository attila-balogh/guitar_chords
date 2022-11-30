
# Guitar chords classification with PyTorch

<br>
Used dataset: homemade images (5.000+)
<br>
<br>
Time of training (with validation phases) 1h 23m 45s.
<br>
<br>


<img src= "utils/loss.png">
<br>
<img src= "utils/acc.png">
<br>
<img src= "utils/lr.png">
<br>
<img src= "utils/test_cm.png">
Confusion matrix was created from the predictions on the TEST set
<br>
<br>
<br>

### Training details:
<br>



Total parameters: 			      8,910,671<br>
Total trainable parameters: 	8,910,671<br>
<br>
Training dataset size: 		5,092<br>
Validation dataset size: 	1,091<br>
Test dataset size: 			1,092<br>
<br>
Input image size: 	 256 × 256<br>
<br>
Batch size    	 32<br>
Learning rate 	 0.0001<br>
Loss function 	 <function cross_entropy at 0x7fdea31dd680><br>
No. epochs    	 100<br>
<br>
<br>
<br>
EPOCH: 1	Time:  0h  0m 49s<br>
Training	train accuracy 8.42%	train loss: 2.7048<br>
Validation	val accuracy   7.79%	val loss:   2.7045<br>
<br>
EPOCH: 2	Time:  0h  0m 48s<br>
Training	train accuracy 10.92%	train loss: 2.6946<br>
Validation	val accuracy   8.43%	val loss:   2.7000<br>
<br>
EPOCH: 3	Time:  0h  0m 49s<br>
Training	train accuracy 12.69%	train loss: 2.6724<br>
Validation	val accuracy   14.02%	val loss:   2.6806<br>
<br>
EPOCH: 4	Time:  0h  0m 48s<br>
Training	train accuracy 14.69%	train loss: 2.6283<br>
Validation	val accuracy   19.34%	val loss:   2.6484<br>
<br>
EPOCH: 5	Time:  0h  0m 49s<br>
Training	train accuracy 17.05%	train loss: 2.5676<br>
Validation	val accuracy   20.99%	val loss:   2.5885<br>
<br>
EPOCH: 6	Time:  0h  0m 48s<br>
Training	train accuracy 19.11%	train loss: 2.5098<br>
Validation	val accuracy   23.01%	val loss:   2.5114<br>
<br>
EPOCH: 7	Time:  0h  0m 49s<br>
Training	train accuracy 21.01%	train loss: 2.4261<br>
Validation	val accuracy   25.57%	val loss:   2.4036<br>
<br>
EPOCH: 8	Time:  0h  0m 49s<br>
Training	train accuracy 25.26%	train loss: 2.2946<br>
Validation	val accuracy   31.90%	val loss:   2.1926<br>
<br>
EPOCH: 9	Time:  0h  0m 49s<br>
Training	train accuracy 28.30%	train loss: 2.1360<br>
Validation	val accuracy   40.97%	val loss:   1.9493<br>
<br>
EPOCH: 10	Time:  0h  0m 49s<br>
Training	train accuracy 34.76%	train loss: 1.9345<br>
Validation	val accuracy   46.47%	val loss:   1.6774<br>
<br>
EPOCH: 11	Time:  0h  0m 49s<br>
Training	train accuracy 41.65%	train loss: 1.7294<br>
Validation	val accuracy   58.48%	val loss:   1.3483<br>
<br>
EPOCH: 12	Time:  0h  0m 49s<br>
Training	train accuracy 46.92%	train loss: 1.5414<br>
Validation	val accuracy   62.79%	val loss:   1.1868<br>
<br>
EPOCH: 13	Time:  0h  0m 49s<br>
Training	train accuracy 51.77%	train loss: 1.3734<br>
Validation	val accuracy   59.85%	val loss:   1.1222<br>
<br>
EPOCH: 14	Time:  0h  0m 49s<br>
Training	train accuracy 56.28%	train loss: 1.2301<br>
Validation	val accuracy   70.21%	val loss:   0.8524<br>
<br>
EPOCH: 15	Time:  0h  0m 49s<br>
Training	train accuracy 62.16%	train loss: 1.0809<br>
Validation	val accuracy   75.71%	val loss:   0.7147<br>
<br>
EPOCH: 16	Time:  0h  0m 49s<br>
Training	train accuracy 64.41%	train loss: 0.9859<br>
Validation	val accuracy   75.16%	val loss:   0.7100<br>
<br>
EPOCH: 17	Time:  0h  0m 48s<br>
Training	train accuracy 68.28%	train loss: 0.8657<br>
Validation	val accuracy   83.96%	val loss:   0.5122<br>
<br>
EPOCH: 18	Time:  0h  0m 49s<br>
Training	train accuracy 71.17%	train loss: 0.8114<br>
Validation	val accuracy   80.11%	val loss:   0.5180<br>
<br>
EPOCH: 19	Time:  0h  0m 49s<br>
Training	train accuracy 72.94%	train loss: 0.7373<br>
Validation	val accuracy   85.79%	val loss:   0.4112<br>
<br>
EPOCH: 20	Time:  0h  0m 49s<br>
Training	train accuracy 75.98%	train loss: 0.6513<br>
Validation	val accuracy   87.53%	val loss:   0.3782<br>
<br>
EPOCH: 21	Time:  0h  0m 49s<br>
Training	train accuracy 78.77%	train loss: 0.6002<br>
Validation	val accuracy   86.53%	val loss:   0.3729<br>
<br>
EPOCH: 22	Time:  0h  0m 49s<br>
Training	train accuracy 79.54%	train loss: 0.5597<br>
Validation	val accuracy   90.83%	val loss:   0.2807<br>
<br>
EPOCH: 23	Time:  0h  0m 49s<br>
Training	train accuracy 81.87%	train loss: 0.5115<br>
Validation	val accuracy   90.93%	val loss:   0.2549<br>
<br>
EPOCH: 24	Time:  0h  0m 49s<br>
Training	train accuracy 82.95%	train loss: 0.4680<br>
Validation	val accuracy   90.10%	val loss:   0.2540<br>
<br>
EPOCH: 25	Time:  0h  0m 49s<br>
Training	train accuracy 85.02%	train loss: 0.4106<br>
Validation	val accuracy   81.85%	val loss:   0.5116<br>
<br>
EPOCH: 26	Time:  0h  0m 48s<br>
Training	train accuracy 86.23%	train loss: 0.3960<br>
Validation	val accuracy   73.88%	val loss:   0.8186<br>
<br>
EPOCH: 27	Time:  0h  0m 49s<br>
Training	train accuracy 87.57%	train loss: 0.3337<br>
Validation	val accuracy   90.38%	val loss:   0.2484<br>
<br>
EPOCH: 28	Time:  0h  0m 49s<br>
Training	train accuracy 88.24%	train loss: 0.3286<br>
Validation	val accuracy   85.88%	val loss:   0.3976<br>
<br>
EPOCH: 29	Time:  0h  0m 48s<br>
Training	train accuracy 88.04%	train loss: 0.3361<br>
Validation	val accuracy   93.03%	val loss:   0.1776<br>
<br>
EPOCH: 30	Time:  0h  0m 49s<br>
Training	train accuracy 88.71%	train loss: 0.3058<br>
Validation	val accuracy   94.50%	val loss:   0.1707<br>
<br>
EPOCH: 31	Time:  0h  0m 49s<br>
Training	train accuracy 90.02%	train loss: 0.2744<br>
Validation	val accuracy   96.15%	val loss:   0.1116<br>
<br>
EPOCH: 32	Time:  0h  0m 49s<br>
Training	train accuracy 90.85%	train loss: 0.2496<br>
Validation	val accuracy   95.78%	val loss:   0.1195<br>
<br>
EPOCH: 33	Time:  0h  0m 48s<br>
Training	train accuracy 91.95%	train loss: 0.2327<br>
Validation	val accuracy   94.41%	val loss:   0.1733<br>
<br>
EPOCH: 34	Time:  0h  0m 49s<br>
Training	train accuracy 92.52%	train loss: 0.2161<br>
Validation	val accuracy   93.13%	val loss:   0.2052<br>
<br>
EPOCH: 35	Time:  0h  0m 49s<br>
Training	train accuracy 92.05%	train loss: 0.2150<br>
Validation	val accuracy   97.62%	val loss:   0.0845<br>
<br>
EPOCH: 36	Time:  0h  0m 48s<br>
Training	train accuracy 93.01%	train loss: 0.2061<br>
Validation	val accuracy   95.51%	val loss:   0.1480<br>
<br>
EPOCH: 37	Time:  0h  0m 49s<br>
Training	train accuracy 93.83%	train loss: 0.1759<br>
Validation	val accuracy   96.79%	val loss:   0.0895<br>
<br>
EPOCH: 38	Time:  0h  0m 48s<br>
Training	train accuracy 93.64%	train loss: 0.1801<br>
Validation	val accuracy   93.22%	val loss:   0.1694<br>
<br>
EPOCH: 39	Time:  0h  0m 49s<br>
Training	train accuracy 93.83%	train loss: 0.1763<br>
Validation	val accuracy   97.34%	val loss:   0.0845<br>
<br>
EPOCH: 40	Time:  0h  0m 49s<br>
Training	train accuracy 94.52%	train loss: 0.1558<br>
Validation	val accuracy   97.89%	val loss:   0.0781<br>
<br>
EPOCH: 41	Time:  0h  0m 50s<br>
Training	train accuracy 94.70%	train loss: 0.1522<br>
Validation	val accuracy   98.81%	val loss:   0.0460<br>
<br>
EPOCH: 42	Time:  0h  0m 53s<br>
Training	train accuracy 94.97%	train loss: 0.1451<br>
Validation	val accuracy   96.70%	val loss:   0.0961<br>
<br>
EPOCH: 43	Time:  0h  0m 51s<br>
Training	train accuracy 95.15%	train loss: 0.1398<br>
Validation	val accuracy   98.44%	val loss:   0.0670<br>
<br>
EPOCH: 44	Time:  0h  0m 50s<br>
Training	train accuracy 94.70%	train loss: 0.1488<br>
Validation	val accuracy   97.71%	val loss:   0.0748<br>
<br>
EPOCH: 45	Time:  0h  0m 50s<br>
Training	train accuracy 95.78%	train loss: 0.1348<br>
Validation	val accuracy   98.08%	val loss:   0.0631<br>
<br>
EPOCH: 46	Time:  0h  0m 49s<br>
Training	train accuracy 95.54%	train loss: 0.1282<br>
Validation	val accuracy   98.35%	val loss:   0.0560<br>
<br>
EPOCH: 47	Time:  0h  0m 50s<br>
Training	train accuracy 95.93%	train loss: 0.1203<br>
Validation	val accuracy   98.17%	val loss:   0.0651<br>
<br>
EPOCH: 48	Time:  0h  0m 49s<br>
Training	train accuracy 95.99%	train loss: 0.1187<br>
Validation	val accuracy   98.44%	val loss:   0.0477<br>
<br>
EPOCH: 49	Time:  0h  0m 50s<br>
Training	train accuracy 95.56%	train loss: 0.1222<br>
Validation	val accuracy   98.53%	val loss:   0.0491<br>
<br>
EPOCH: 50	Time:  0h  0m 49s<br>
Training	train accuracy 96.33%	train loss: 0.1040<br>
Validation	val accuracy   98.72%	val loss:   0.0460<br>
<br>
EPOCH: 51	Time:  0h  0m 49s<br>
Training	train accuracy 96.21%	train loss: 0.1073<br>
Validation	val accuracy   98.81%	val loss:   0.0456<br>
<br>
EPOCH: 52	Time:  0h  0m 49s<br>
Training	train accuracy 95.99%	train loss: 0.1070<br>
Validation	val accuracy   98.81%	val loss:   0.0426<br>
<br>
EPOCH: 53	Time:  0h  0m 49s<br>
Training	train accuracy 96.78%	train loss: 0.0949<br>
Validation	val accuracy   98.99%	val loss:   0.0461<br>
<br>
EPOCH: 54	Time:  0h  0m 49s<br>
Training	train accuracy 97.13%	train loss: 0.0861<br>
Validation	val accuracy   98.35%	val loss:   0.0564<br>
<br>
EPOCH: 55	Time:  0h  0m 49s<br>
Training	train accuracy 97.09%	train loss: 0.0856<br>
Validation	val accuracy   98.99%	val loss:   0.0458<br>
<br>
EPOCH: 56	Time:  0h  0m 50s<br>
Training	train accuracy 97.51%	train loss: 0.0767<br>
Validation	val accuracy   98.44%	val loss:   0.0497<br>
<br>
EPOCH: 57	Time:  0h  0m 50s<br>
Training	train accuracy 97.19%	train loss: 0.0768<br>
Validation	val accuracy   99.08%	val loss:   0.0398<br>
<br>
EPOCH: 58	Time:  0h  0m 49s<br>
Training	train accuracy 97.33%	train loss: 0.0797<br>
Validation	val accuracy   99.08%	val loss:   0.0372<br>
<br>
EPOCH: 59	Time:  0h  0m 49s<br>
Training	train accuracy 97.53%	train loss: 0.0718<br>
Validation	val accuracy   99.18%	val loss:   0.0365<br>
<br>
EPOCH: 60	Time:  0h  0m 49s<br>
Training	train accuracy 97.58%	train loss: 0.0654<br>
Validation	val accuracy   98.90%	val loss:   0.0466<br>
<br>
EPOCH: 61	Time:  0h  0m 49s<br>
Training	train accuracy 97.62%	train loss: 0.0691<br>
Validation	val accuracy   98.90%	val loss:   0.0460<br>
<br>
EPOCH: 62	Time:  0h  0m 50s<br>
Training	train accuracy 97.55%	train loss: 0.0675<br>
Validation	val accuracy   98.99%	val loss:   0.0438<br>
<br>
EPOCH: 63	Time:  0h  0m 49s<br>
Training	train accuracy 98.10%	train loss: 0.0598<br>
Validation	val accuracy   98.72%	val loss:   0.0467<br>
<br>
EPOCH: 64	Time:  0h  0m 50s<br>
Training	train accuracy 98.13%	train loss: 0.0569<br>
Validation	val accuracy   99.08%	val loss:   0.0420<br>
<br>
EPOCH: 65	Time:  0h  0m 49s<br>
Training	train accuracy 97.92%	train loss: 0.0565<br>
Validation	val accuracy   98.63%	val loss:   0.0530<br>
<br>
EPOCH: 66	Time:  0h  0m 50s<br>
Training	train accuracy 98.33%	train loss: 0.0561<br>
Validation	val accuracy   99.27%	val loss:   0.0405<br>
<br>
EPOCH: 67	Time:  0h  0m 49s<br>
Training	train accuracy 98.25%	train loss: 0.0582<br>
Validation	val accuracy   98.90%	val loss:   0.0431<br>
<br>
EPOCH: 68	Time:  0h  0m 49s<br>
Training	train accuracy 98.23%	train loss: 0.0554<br>
Validation	val accuracy   99.18%	val loss:   0.0422<br>
<br>
EPOCH: 69	Time:  0h  0m 50s<br>
Training	train accuracy 98.41%	train loss: 0.0445<br>
Validation	val accuracy   99.27%	val loss:   0.0434<br>
<br>
EPOCH: 70	Time:  0h  0m 49s<br>
Training	train accuracy 98.64%	train loss: 0.0407<br>
Validation	val accuracy   99.18%	val loss:   0.0362<br>
<br>
EPOCH: 71	Time:  0h  0m 49s<br>
Training	train accuracy 98.64%	train loss: 0.0452<br>
Validation	val accuracy   98.99%	val loss:   0.0424<br>
<br>
EPOCH: 72	Time:  0h  0m 49s<br>
Training	train accuracy 98.47%	train loss: 0.0441<br>
Validation	val accuracy   99.27%	val loss:   0.0472<br>
<br>
EPOCH: 73	Time:  0h  0m 49s<br>
Training	train accuracy 98.53%	train loss: 0.0482<br>
Validation	val accuracy   99.27%	val loss:   0.0381<br>
<br>
EPOCH: 74	Time:  0h  0m 49s<br>
Training	train accuracy 98.90%	train loss: 0.0371<br>
Validation	val accuracy   99.27%	val loss:   0.0359<br>
<br>
EPOCH: 75	Time:  0h  0m 49s<br>
Training	train accuracy 98.64%	train loss: 0.0402<br>
Validation	val accuracy   99.08%	val loss:   0.0430<br>
<br>
EPOCH: 76	Time:  0h  0m 50s<br>
Training	train accuracy 98.63%	train loss: 0.0415<br>
Validation	val accuracy   99.27%	val loss:   0.0337<br>
<br>
EPOCH: 77	Time:  0h  0m 49s<br>
Training	train accuracy 98.88%	train loss: 0.0328<br>
Validation	val accuracy   98.99%	val loss:   0.0380<br>
<br>
EPOCH: 78	Time:  0h  0m 50s<br>
Training	train accuracy 98.76%	train loss: 0.0382<br>
Validation	val accuracy   99.27%	val loss:   0.0359<br>
<br>
EPOCH: 79	Time:  0h  0m 50s<br>
Training	train accuracy 98.76%	train loss: 0.0333<br>
Validation	val accuracy   99.27%	val loss:   0.0380<br>
<br>
EPOCH: 80	Time:  0h  0m 49s<br>
Training	train accuracy 98.66%	train loss: 0.0389<br>
Validation	val accuracy   99.18%	val loss:   0.0334<br>
<br>
EPOCH: 81	Time:  0h  0m 50s<br>
Training	train accuracy 98.90%	train loss: 0.0356<br>
Validation	val accuracy   99.27%	val loss:   0.0377<br>
<br>
EPOCH: 82	Time:  0h  0m 49s<br>
Training	train accuracy 99.08%	train loss: 0.0278<br>
Validation	val accuracy   99.18%	val loss:   0.0370<br>
<br>
EPOCH: 83	Time:  0h  0m 49s<br>
Training	train accuracy 98.92%	train loss: 0.0322<br>
Validation	val accuracy   99.18%	val loss:   0.0336<br>
<br>
EPOCH: 84	Time:  0h  0m 49s<br>
Training	train accuracy 99.08%	train loss: 0.0303<br>
Validation	val accuracy   99.36%	val loss:   0.0339<br>
<br>
EPOCH: 85	Time:  0h  0m 49s<br>
Training	train accuracy 99.12%	train loss: 0.0300<br>
Validation	val accuracy   99.08%	val loss:   0.0366<br>
<br>
EPOCH: 86	Time:  0h  0m 50s<br>
Training	train accuracy 98.98%	train loss: 0.0285<br>
Validation	val accuracy   99.18%	val loss:   0.0410<br>
<br>
EPOCH: 87	Time:  0h  0m 51s<br>
Training	train accuracy 99.08%	train loss: 0.0275<br>
Validation	val accuracy   99.18%	val loss:   0.0387<br>
<br>
EPOCH: 88	Time:  0h  0m 51s<br>
Training	train accuracy 99.39%	train loss: 0.0242<br>
Validation	val accuracy   99.18%	val loss:   0.0359<br>
<br>
EPOCH: 89	Time:  0h  0m 50s<br>
Training	train accuracy 99.19%	train loss: 0.0232<br>
Validation	val accuracy   99.27%	val loss:   0.0373<br>
<br>
EPOCH: 90	Time:  0h  0m 52s<br>
Training	train accuracy 99.23%	train loss: 0.0256<br>
Validation	val accuracy   99.36%	val loss:   0.0350<br>
<br>
EPOCH: 91	Time:  0h  0m 54s<br>
Training	train accuracy 99.29%	train loss: 0.0214<br>
Validation	val accuracy   99.27%	val loss:   0.0345<br>
<br>
EPOCH: 92	Time:  0h  0m 54s<br>
Training	train accuracy 99.00%	train loss: 0.0311<br>
Validation	val accuracy   99.36%	val loss:   0.0361<br>
<br>
EPOCH: 93	Time:  0h  0m 54s<br>
Training	train accuracy 99.31%	train loss: 0.0232<br>
Validation	val accuracy   99.36%	val loss:   0.0357<br>
<br>
EPOCH: 94	Time:  0h  0m 57s<br>
Training	train accuracy 99.14%	train loss: 0.0256<br>
Validation	val accuracy   99.27%	val loss:   0.0365<br>
<br>
EPOCH: 95	Time:  0h  0m 56s<br>
Training	train accuracy 99.41%	train loss: 0.0213<br>
Validation	val accuracy   99.36%	val loss:   0.0356<br>
<br>
EPOCH: 96	Time:  0h  0m 55s<br>
Training	train accuracy 99.18%	train loss: 0.0243<br>
Validation	val accuracy   99.36%	val loss:   0.0344<br>
<br>
EPOCH: 97	Time:  0h  0m 54s<br>
Training	train accuracy 99.16%	train loss: 0.0234<br>
Validation	val accuracy   99.36%	val loss:   0.0351<br>
<br>
EPOCH: 98	Time:  0h  0m 54s<br>
Training	train accuracy 99.31%	train loss: 0.0228<br>
Validation	val accuracy   99.36%	val loss:   0.0348<br>
<br>
EPOCH: 99	Time:  0h  0m 55s<br>
Training	train accuracy 99.35%	train loss: 0.0212<br>
Validation	val accuracy   99.36%	val loss:   0.0346<br>
<br>
EPOCH: 100	Time:  0h  0m 54s<br>
Training	train accuracy 99.25%	train loss: 0.0227<br>
Validation	val accuracy   99.36%	val loss:   0.0351<br>
<br>
<br>
<br>
TIME of training (with validation phases)  1h 23m 45s.<br>
