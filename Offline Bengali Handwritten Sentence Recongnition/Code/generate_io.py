import os;
from skimage import io, transform, color, filters
import cv2;
import pickle
import get_feature_vector as gfv
import get_label_vector as glv
import matplotlib.pyplot as plt
import numpy as np;

from sklearn.model_selection import train_test_split

myFile = 'borno.txt';
with open(myFile , 'rb') as f:
    rs_borno = pickle.load(f);

#sort the arrayz
rs_borno.sort();
#print(rs_borno);

def to_string(ar, rs_borno):
    #print(ar);
    ans = "";

    #ans2 = list();
    vowel_dep = ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', 'ং', '্'];
    char_set = list();

    indx = 0 ;
    character = "";
    while(indx<len(ar)):
        ch = rs_borno[ar[indx]];

        if(ch != '্'):
            if(len(character) > 0):
                char_set.append(character);
                character = "";

            if(ch not in vowel_dep):
                character += ch;

            else:
                char_set.append(ch);
        else:
            character += ch;
            if(indx+1 < len(ar)):
                indx+=1;
                character += rs_borno[ar[indx]];
        indx+=1;

    if(len(character)>0):
        char_set.append(character);
    #print(char_set);
    ind = 0;
    while(ind<len(char_set)):
        ch = char_set[ind];
        if(ch == 'ে'):
            if(ind+2<len(char_set) and char_set[ind+2] == 'া'):
                ans += char_set[ind+1];
                ans += 'ো';
                ind+=2;
            elif(ind+2<len(char_set) and char_set[ind+2] == 'ী'):
                ans += char_set[ind+1];
                ans += 'ৌ';
                ind+=2;
            else:
                if(ind+1<len(char_set)):
                    ans += char_set[ind+1];
                    ind+=1;
                ans += ch;
        elif(ch == 'ি' or ch =='ৈ'):
            if(ind+1<len(char_set)):
                ans += char_set[ind+1];
                ind+=1;
            ans += ch;
        else:
            ans += ch;
        ind+=1;
    return ans; #, ans2;



data_directory = 'word_image/';

#wall through all the subdirectories, returns a tuple,
#first element is the sub_directory, second is last directory name, last is list of fies
sub_dirs = [x[0] for x in os.walk(data_directory)];
word_list = [x[1] for x in os.walk(data_directory)];
word_list = word_list[0];
files = [x[2] for x in os.walk(data_directory)];
print(word_list);
print(len(word_list));


train_input = [];
test_input = [];
train_output = [];
test_output = [];
#iterate through all the data_samples
proc = 0;
for (index,dirs) in enumerate(sub_dirs):
    if(index == 0):continue;
    #if(index == 3): break;
    print("Processing word in directory: ",dirs);

    print("clearing input and output");
    input_word = [];
    output_label = [];
    for file in files[index]:

        #fetch the word
        sample = cv2.imread(dirs+"/"+str(file));

        #print("Processing word no: ",file);
        #io.imshow(sample);

        #convert to greyscale and binary
        grey = color.rgb2gray(sample);
        threshold = filters.threshold_otsu(grey);
        binary_sample = grey > threshold;
        #io.imshow(binary_sample);

        #get feature vector
        features = gfv.get_features(binary_sample);

        #append feature and output vectors
        input_word.append(features);
        output_label.append(glv.get_labels(word_list[index-1]));

        proc += 1;

    print("Splitting train and test data sets");
    test_x = list();
    test_y = list();
    train_x1 = list();
    train_x2 = list();
    train_y1 = list();
    train_y2 = list();
    for (i, sample) in enumerate(input_word):
        if(i%3 == 2):
            test_x.append(input_word[i]);
            test_y.append(output_label[i]);
        elif(i%3 == 0):
            train_x1.append(input_word[i]);
            train_y1.append(output_label[i]);
        else:
            train_x2.append(input_word[i]);
            train_y2.append(output_label[i]);

    #train_x, test_x, train_y, test_y = train_test_split(input_word, output_label, test_size = 0.10, random_state = 31);

    print("Extending input and outputs");
    train_input.extend(train_x1);
    train_input.extend(train_x2);
    train_output.extend(train_y1);
    train_output.extend(train_y2);
    test_input.extend(test_x);
    test_output.extend(test_y);

    print(proc, " images processed");

# no_col = 2;
# no_row = 20/no_col;
# if((no_row%no_col)!=0):
#     no_row+=1;

# f = plt.figure(figsize=(25,10));
# f.subplots_adjust(hspace=0.5, wspace=0.2)
# for num,img in enumerate(train_input):
#     if(num==20): break;
#     a = f.add_subplot(no_row,no_col,num+1);
#     #img = rescale(img, 64);
#     tran = np.transpose(img);
#     x = np.linspace(1, 64, 64)


#     for item in tran:
#         a.plot(x, item, label='feature')

#     a.set_title(to_string(train_output[num], rs_borno));

# plt.legend()

# plt.show()
#save the input and output array in a file
myFilei = "train_input.txt";
myFileo = "train_output.txt";
myFiletsti = "test_input.txt";
myFiletsto = "test_output.txt";
with open(myFilei, 'wb') as f:
    pickle.dump(train_input, f);
    print("Train input SuccessFully Saved");
with open(myFileo, 'wb') as f:
    pickle.dump(train_output, f);
    print("Train output SuccessFully Saved");
with open(myFiletsti, 'wb') as f:
    pickle.dump(test_input, f);
    print("Test input SuccessFully Saved");
with open(myFiletsto, 'wb') as f:
    pickle.dump(test_output, f);
    print("Test output SuccessFully Saved");