def get_features(binary_sample):
    features =  list();
    #extract 9 features for 64 time steps
    row, col = binary_sample.shape;

    for j in range(col):

        #declare and initialize variables
        f1 = 0;
        f2 = 0;
        f3 = 0;
        maxpos = 0;
        minpos = row+1;
        tran_cnt = 0;
        cnt = 0;

        #calculate feature for this time step;
        for i in range(row):
            if(binary_sample[i][j] == False):
                f1+=1;
                f2+=(i+1);
                f3+=((i+1)*(i+1));
                maxpos = max(maxpos, i+1);
                minpos = min(minpos, i+1);
                cnt+=1;

            #calculate black-white transitions
            if(i>0 and binary_sample[i-1][j]!=binary_sample[i][j]):
                tran_cnt += 1;

        f1/=row;
        f2/=row;
        f3/=(row*row);
        gradmax = maxpos/(j+1);
        gradmin = minpos/(j+1);

        #make a vector of features
        features.append([f1, f2, f3, maxpos, minpos, gradmax, gradmin, tran_cnt, cnt]);


    #return features;
    return features;