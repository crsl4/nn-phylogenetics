import json

data = {}
data['ngpu'] = 1
data['lr'] = 0.001
data['batchSize'] = 128 
data['modelRoot'] = 'best_models'
data['dataRoot'] = "../../data/new-lba/"
data['labelFile'] = "labels113683228-0.1-10.0-0.01.in"
data['matFile'] = "sequences113683228-0.1-10.0-0.01.in"
data['nTrainSamples'] = 9500
data['nTestSamples'] = 500
data['nEpochs'] = 100
data['gamma'] = 0.95
data['lrSteps'] 10 

labels_names = ['labels113683228-0.1-10.0-0.01.in',
'labels114134-1.0-10.0-1.0.in', 'labels1346243-1.0-10.0-0.01.in',
'labels18683228-0.1-10.0-1.0.in', 'labels2325654-0.1-40.0-0.1.in',
'labels3245235-1.0-40.0-0.01.in', 'labels3363123-1.0-10.0-0.1.in',
'labels346266-1.0-40.0-1.0.in', 'labels3615253-0.5-40.0-0.1.in',
'labels372783-0.1-40.0-1.0.in', 'labels416173-0.5-40.0-0.01.in',
'labels4473421-0.5-10.0-0.1.in', 'labels45435-1.0-40.0-0.1.in',
'labels467733-0.5-40.0-1.0.in', 'labels4728283-0.5-10.0-0.01.in',
'labels4738282-0.1-2.0-0.01.in', 'labels4919173-0.5-2.0-1.0.in',
'labels5722724-0.5-2.0-0.1.in', 'labels58583625-0.5-2.0-0.01.in',
'labels675223-1.0-2.0-0.01.in', 'labels68113228-0.1-2.0-0.1.in',
'labels68163228-0.1-2.0-1.0.in', 'labels68326728-0.1-10.0-0.1.in',
'labels7842344-1.0-2.0-0.1.in', 'labels88422-1.0-2.0-1.0.in',
'labels976422-0.5-10.0-1.0.in', 'labels976683228-0.1-40.0-0.01.in']

sequence_names =['sequences113683228-0.1-10.0-0.01.in',
'sequences114134-1.0-10.0-1.0.in', 'sequences1346243-1.0-10.0-0.01.in',
'sequences18683228-0.1-10.0-1.0.in', 'sequences2325654-0.1-40.0-0.1.in',
'sequences3245235-1.0-40.0-0.01.in', 'sequences3363123-1.0-10.0-0.1.in',
'sequences346266-1.0-40.0-1.0.in', 'sequences3615253-0.5-40.0-0.1.in',
'sequences372783-0.1-40.0-1.0.in', 'sequences416173-0.5-40.0-0.01.in',
'sequences4473421-0.5-10.0-0.1.in', 'sequences45435-1.0-40.0-0.1.in',
'sequences467733-0.5-40.0-1.0.in', 'sequences4728283-0.5-10.0-0.01.in',
'sequences4738282-0.1-2.0-0.01.in', 'sequences4919173-0.5-2.0-1.0.in',
'sequences5722724-0.5-2.0-0.1.in', 'sequences58583625-0.5-2.0-0.01.in',
'sequences675223-1.0-2.0-0.01.in', 'sequences68113228-0.1-2.0-0.1.in',
'sequences68163228-0.1-2.0-1.0.in', 'sequences68326728-0.1-10.0-0.1.in',
'sequences7842344-1.0-2.0-0.1.in', 'sequences88422-1.0-2.0-1.0.in',
'sequences976422-0.5-10.0-1.0.in', 'sequences976683228-0.1-40.0-0.01.in']


for ii in range(len(sequence_names)):
    data['labelFile'] = labels_names[ii]
    data['matFile'] = labels_names[ii]

    name = "permutation_equivariant_new_lba_"+str(ii+1)+".json"

    with open(name, 'w') as outfile:
        json.dump(data, outfile)

